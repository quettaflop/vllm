# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused DSV4.1 shifted mHC (post + next pre + collapse RMSNorm) for sm_120, one kernel per sublayer.

The sm_120 counterpart of DeepGEMM's ``sm100_mega_mhc`` (which needs tcgen05/TMEM). It replaces the TileLang pair
``mhc_fused_tilelang`` (post + split-K pre-norm GEMM) and ``mhc_pre_big_fuse_with_norm_tilelang`` (split
reduction, Sinkhorn, delayed collapse, RMSNorm) for decode-sized token counts.

One CTA per 128-wide hidden slice, one warp per token (<= 8 tokens):
  1. the CTA's slice of ``fn`` (24 x 4 x 128 fp32) streams into shared memory with cp.async *before*
     ``griddepcontrol.wait``, so it overlaps the previous kernel under PDL;
  2. post: new_r[j] = post[j] * x + sum_k comb[k][j] * residual[k] (fp32, TileLang's order), written as BF16;
     the slice's 24 projection partials and the residual square sum, in fp32 FMAs;
  3. the delayed collapse uses the carried pre-mix, so it needs no projection: o = sum_j pre_in[j] * bf16(new_r[j]),
     rounded to BF16, and its square sum;
  4. partials go to global memory; one arrival counter; every CTA normalizes its own slice with the total square sum
     (fixed summation order), and the last CTA to arrive reduces the projection, applies the RMS scale and runs
     the sigmoid / Sinkhorn epilogue (16 lanes, shuffles). Counters reset themselves.
Arithmetic is fp32 SIMT like the TileLang path (DeepGEMM's sm100 kernel uses TF32 UMMA). The post-mapped residual
is bit-identical to TileLang; the other outputs differ only by summation order.
Enable with VLLM_SM120_MHC=1.
"""

import os

import torch

CUDA_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cstdint>

#define DEVI __device__ __forceinline__
constexpr int HC = 4, MIX = 24, HS = 128, THREADS = 256, MAXT = 8, NPART = MIX + 2;
constexpr int KS = HC * HS;     // K per CTA slice (512)
constexpr int KP = KS + 4;      // padded smem row (bank-conflict-free mma fragments)

struct MhcArgs {
  const __nv_bfloat16* x; const __nv_bfloat16* res; const float* post_in; const float* comb_in; const float* fn;
  const float* scale; const float* base; const float* pre_in; const __nv_bfloat16* nw;
  __nv_bfloat16* res_out; float* post_out; float* comb_out; __nv_bfloat16* y; float* pre_out;
  float* part; unsigned* ctr; unsigned long long* dbg;
  int T, H, nsplit, sk_iters;
  float rms_eps, pre_eps, sk_eps, post_mult, norm_eps;
};

DEVI void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
DEVI void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
DEVI float bf(__nv_bfloat16 v) { return __bfloat162float(v); }
DEVI float rcp_fast(float v) { float r; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(v)); return r; }
// One MUFU each (the IEEE-rounded forms compile to guarded slow-path regions, ~60-80 cycles each on one thread)
DEVI float sigmoidf_(float v) { return rcp_fast(1.f + __expf(-v)); }
DEVI unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
#define STAMP(i) do { if (a.dbg && threadIdx.x == 0) a.dbg[blockIdx.x * 16 + (i)] = gtime(); } while (0)
DEVI uint32_t tf32(float v) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(v)); return r; }
DEVI void mma_tf32(float* d, const uint32_t* a, const uint32_t* b) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// 3xTF32: hi*hi + hi*lo + lo*hi, close to fp32 accuracy
DEVI void split(float v, uint32_t& hi, uint32_t& lo) { hi = tf32(v); lo = tf32(v - __uint_as_float(hi)); }

__global__ void __launch_bounds__(THREADS, 1) sm120_mhc_kernel(const MhcArgs a) {
  extern __shared__ __align__(16) float smem[];
  float* sW = smem;                 // [MIX][KP]   fn slice, k = j * HS + h
  float* sR = sW + MIX * KP;        // [MAXT][KP]  post-mapped residual (fp32), zero rows for t >= T
  float* red = sR + MAXT * KP;      // [8 warps][2 tiles][32 lanes][4]
  __shared__ int s_last;
  __shared__ float s_sq[MAXT], s_sqo[MAXT];
  __shared__ float s_par[3 + MIX];  // scale[3], base[MIX]: static, loaded before the dependency wait
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, c4 = lane & 3;
  const int H = a.H, T = a.T, h0 = blockIdx.x * HS;
  STAMP(0);
  // 1. weights first: fn is static, so it streams in while the previous kernel finishes (PDL)
  constexpr int CH = KS / 4;
  for (int i = tid; i < MIX * CH; i += THREADS) {
    int n = i / CH, rem = i - n * CH, j = rem / (HS / 4), c = rem - j * (HS / 4);
    __pipeline_memcpy_async(sW + n * KP + j * HS + c * 4, a.fn + (size_t)n * HC * H + (size_t)j * H + h0 + c * 4, 16);
  }
  __pipeline_commit();
  if (tid < 3) s_par[tid] = a.scale[tid];
  else if (tid < 3 + MIX) s_par[tid] = a.base[tid - 3];
  pdl_wait();
  STAMP(1);
  // 2. post (warp = token): residual out, fp32 copy for the projection, collapse with the carried pre-mix
  float okeep[HS / 32];
  if (warp < T) {
    float pm[HC], cm[HC][HC], pin[HC];
#pragma unroll
    for (int j = 0; j < HC; j++) {
      pm[j] = a.post_in[warp * HC + j];
      pin[j] = a.pre_in[warp * HC + j];
#pragma unroll
      for (int k = 0; k < HC; k++) cm[k][j] = a.comb_in[(warp * HC + k) * HC + j];
    }
    float sq = 0.f, sqo = 0.f;
#pragma unroll
    for (int p = 0; p < HS / 32; p++) {
      int hl = p * 32 + lane, h = h0 + hl;
      float xv = bf(a.x[(size_t)warp * H + h]);
      float rv[HC];
#pragma unroll
      for (int k = 0; k < HC; k++) rv[k] = bf(a.res[((size_t)warp * HC + k) * H + h]);
      float o = 0.f;
#pragma unroll
      for (int j = 0; j < HC; j++) {
        float nr = pm[j] * xv;
#pragma unroll
        for (int k = 0; k < HC; k++) nr += cm[k][j] * rv[k];
        __nv_bfloat16 rb = __float2bfloat16(nr);
        a.res_out[((size_t)warp * HC + j) * H + h] = rb;
        sq += nr * nr;
        o += pin[j] * bf(rb);
        sR[warp * KP + j * HS + hl] = nr;
      }
      float orr = bf(__float2bfloat16(o));
      okeep[p] = orr;
      sqo += orr * orr;
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) {
      sq += __shfl_xor_sync(0xffffffffu, sq, o);
      sqo += __shfl_xor_sync(0xffffffffu, sqo, o);
    }
    if (lane == 0) { s_sq[warp] = sq; s_sqo[warp] = sqo; }
  } else if (warp < MAXT) {
    for (int k = lane; k < KS; k += 32) sR[warp * KP + k] = 0.f;
  }
  __pipeline_wait_prior(0);
  __syncthreads();
  STAMP(2);
  // 3. projection on tensor cores: D[n][t] = sum_k W[n][k] R[t][k], 3xTF32, K split over the 8 warps
  float d0[4] = {0.f, 0.f, 0.f, 0.f}, d1[4] = {0.f, 0.f, 0.f, 0.f};
  const int ksteps = KS / 8 / 8;  // per warp
#pragma unroll
  for (int ks = 0; ks < ksteps; ks++) {
    int k0 = (warp * ksteps + ks) * 8;
    uint32_t bh[2], bl[2];
    split(sR[g * KP + k0 + c4], bh[0], bl[0]);
    split(sR[g * KP + k0 + c4 + 4], bh[1], bl[1]);
    uint32_t ah[4], al[4];
    // tile 0: rows 0..15
    split(sW[g * KP + k0 + c4], ah[0], al[0]);
    split(sW[(g + 8) * KP + k0 + c4], ah[1], al[1]);
    split(sW[g * KP + k0 + c4 + 4], ah[2], al[2]);
    split(sW[(g + 8) * KP + k0 + c4 + 4], ah[3], al[3]);
    mma_tf32(d0, ah, bh); mma_tf32(d0, ah, bl); mma_tf32(d0, al, bh);
    // tile 1: rows 16..23 (rows 24..31 are zero)
    split(sW[(16 + g) * KP + k0 + c4], ah[0], al[0]);
    split(sW[(16 + g) * KP + k0 + c4 + 4], ah[2], al[2]);
    ah[1] = ah[3] = al[1] = al[3] = 0u;
    mma_tf32(d1, ah, bh); mma_tf32(d1, ah, bl); mma_tf32(d1, al, bh);
  }
  float* rw = red + ((warp * 2) * 32 + lane) * 4;
#pragma unroll
  for (int i = 0; i < 4; i++) { rw[i] = d0[i]; rw[32 * 4 + i] = d1[i]; }
  __syncthreads();
  // cross-warp reduction in a fixed order; thread -> (tile, lane, reg) = one output element
  {
    int tile = tid >> 7, l = (tid >> 2) & 31, r = tid & 3;
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < 8; w++) v += red[((w * 2 + tile) * 32 + l) * 4 + r];
    int gg = l >> 2, cc = l & 3;
    int n = tile * 16 + gg + ((r & 2) ? 8 : 0), t = 2 * cc + (r & 1);
    if (n < MIX && t < T) a.part[((size_t)t * NPART + n) * a.nsplit + blockIdx.x] = v;
    if (tid < T) {
      a.part[((size_t)tid * NPART + MIX) * a.nsplit + blockIdx.x] = s_sq[tid];
      a.part[((size_t)tid * NPART + MIX + 1) * a.nsplit + blockIdx.x] = s_sqo[tid];
    }
  }
  STAMP(3);
  // 4. arrive and wait for every slice (all CTAs resident: grid <= SMs)
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    unsigned old = atomicAdd(&a.ctr[0], 1u);
    s_last = old == (unsigned)(a.nsplit - 1);
    volatile unsigned* arrive = a.ctr;
    while (*arrive < (unsigned)a.nsplit) __nanosleep(32);
    __threadfence();
  }
  __syncthreads();
  STAMP(4);
  // 5. after the barrier every CTA holds all partials. CTA t (< T) runs token t's epilogue on warp 7 while warps
  // 0..T-1 normalize this CTA's slice; with T == 8 warp 7 normalizes first, then runs the epilogue.
  constexpr int EPI = 7;
  __shared__ float s_mix[NPART];
  auto normalize = [&]() {
    const float* col = a.part + ((size_t)warp * NPART + MIX + 1) * a.nsplit;
    float v0 = lane < a.nsplit ? __ldcg(col + lane) : 0.f;  // independent loads (nsplit <= 64)
    float v1 = lane + 32 < a.nsplit ? __ldcg(col + lane + 32) : 0.f;
    float tot = v0 + v1;
#pragma unroll
    for (int o = 16; o; o >>= 1) tot += __shfl_xor_sync(0xffffffffu, tot, o);
    float rn = rsqrtf(tot / (float)H + a.norm_eps);
#pragma unroll
    for (int p = 0; p < HS / 32; p++) {
      int h = h0 + p * 32 + lane;
      a.y[(size_t)warp * H + h] = __float2bfloat16(okeep[p] * rn * bf(a.nw[h]));
    }
  };
  if (warp < T) normalize();
  const int et = blockIdx.x;
  if (warp == EPI && et < T) {
    // stage token et's contiguous [NPART][nsplit] block (independent 16-byte loads, one L2 round trip)
    float* stage = sW;
    const int total = NPART * a.nsplit, nvec = total >> 2;
    const float* src = a.part + (size_t)et * NPART * a.nsplit;
    for (int i = lane; i < nvec; i += 32)
      reinterpret_cast<float4*>(stage)[i] = __ldcg(reinterpret_cast<const float4*>(src) + i);
    for (int i = (nvec << 2) + lane; i < total; i += 32) stage[i] = __ldcg(src + i);
    __syncwarp();
    if (lane < MIX + 1) {
      const float* col = stage + lane * a.nsplit;
      float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
      int si = 0;
      for (; si + 4 <= a.nsplit; si += 4) { s0 += col[si]; s1 += col[si + 1]; s2 += col[si + 2]; s3 += col[si + 3]; }
      for (; si < a.nsplit; si++) s0 += col[si];
      s_mix[lane] = (s0 + s1) + (s2 + s3);
    }
    __syncwarp();
    if (a.dbg && lane == 0) a.dbg[blockIdx.x * 16 + 8] = gtime();
    if (lane == 0) {
      const int t = et;
      float rms = rsqrtf(s_mix[MIX] / (float)(HC * H) + a.rms_eps);
      const float* bs = s_par + 3;
      float sc0 = s_par[0], sc1 = s_par[1], sc2 = s_par[2];
      float pre_v[HC], post_v[HC];
#pragma unroll
      for (int j = 0; j < HC; j++) {
        pre_v[j] = sigmoidf_(s_mix[j] * rms * sc0 + bs[j]) + a.pre_eps;
        post_v[j] = sigmoidf_(s_mix[HC + j] * rms * sc1 + bs[HC + j]) * a.post_mult;
      }
      float m[HC][HC];
#pragma unroll
      for (int j = 0; j < HC; j++) {
        float rmax = -INFINITY;
#pragma unroll
        for (int k = 0; k < HC; k++) {
          m[j][k] = s_mix[2 * HC + j * HC + k] * rms * sc2 + bs[2 * HC + j * HC + k];
          rmax = fmaxf(rmax, m[j][k]);
        }
        float rs = 0.f;
#pragma unroll
        for (int k = 0; k < HC; k++) { m[j][k] = __expf(m[j][k] - rmax); rs += m[j][k]; }
        float rinv = rcp_fast(rs);
#pragma unroll
        for (int k = 0; k < HC; k++) m[j][k] = m[j][k] * rinv + a.sk_eps;
      }
      for (int it = 0; it < a.sk_iters; it++) {
        if (it > 0) {
#pragma unroll
          for (int j = 0; j < HC; j++) {
            float rinv = rcp_fast((m[j][0] + m[j][1]) + (m[j][2] + m[j][3]) + a.sk_eps);
#pragma unroll
            for (int k = 0; k < HC; k++) m[j][k] *= rinv;
          }
        }
#pragma unroll
        for (int k = 0; k < HC; k++) {
          float cinv = rcp_fast((m[0][k] + m[1][k]) + (m[2][k] + m[3][k]) + a.sk_eps);
#pragma unroll
          for (int j = 0; j < HC; j++) m[j][k] *= cinv;
        }
      }
#pragma unroll
      for (int j = 0; j < HC; j++) {
        a.pre_out[t * HC + j] = pre_v[j];
        a.post_out[t * HC + j] = post_v[j];
#pragma unroll
        for (int k = 0; k < HC; k++) a.comb_out[t * HC * HC + j * HC + k] = m[j][k];
      }
      if (a.dbg) a.dbg[blockIdx.x * 16 + 9] = gtime();
    }
  }
  pdl_trigger();
  __syncthreads();
  STAMP(5);
  if (tid == 0) {
    unsigned d = atomicAdd(&a.ctr[1], 1u);
    if (d == (unsigned)(a.nsplit - 1)) { a.ctr[0] = 0u; a.ctr[1] = 0u; __threadfence(); }
  }
  if (tid == 0 && a.dbg) a.dbg[blockIdx.x * 16 + 6] = gtime();
}

void run(const torch::Tensor& x, const torch::Tensor& res, const torch::Tensor& post_in, const torch::Tensor& comb_in,
         const torch::Tensor& fn, const torch::Tensor& scale, const torch::Tensor& base, const torch::Tensor& pre_in,
         const torch::Tensor& nw, const torch::Tensor& res_out, const torch::Tensor& post_out,
         const torch::Tensor& comb_out, const torch::Tensor& y, const torch::Tensor& pre_out,
         const torch::Tensor& part, const torch::Tensor& ctr, int64_t sk_iters, double rms_eps, double pre_eps,
         double sk_eps, double post_mult, double norm_eps, int64_t pdl, const torch::Tensor& dbg) {
  MhcArgs a;
  a.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  a.res = reinterpret_cast<const __nv_bfloat16*>(res.data_ptr());
  a.post_in = post_in.data_ptr<float>();
  a.comb_in = comb_in.data_ptr<float>();
  a.fn = fn.data_ptr<float>();
  a.scale = scale.data_ptr<float>();
  a.base = base.data_ptr<float>();
  a.pre_in = pre_in.data_ptr<float>();
  a.nw = reinterpret_cast<const __nv_bfloat16*>(nw.data_ptr());
  a.res_out = reinterpret_cast<__nv_bfloat16*>(res_out.data_ptr());
  a.post_out = post_out.data_ptr<float>();
  a.comb_out = comb_out.data_ptr<float>();
  a.y = reinterpret_cast<__nv_bfloat16*>(y.data_ptr());
  a.pre_out = pre_out.data_ptr<float>();
  a.part = part.data_ptr<float>();
  a.ctr = reinterpret_cast<unsigned*>(ctr.data_ptr<int32_t>());
  a.dbg = dbg.numel() ? reinterpret_cast<unsigned long long*>(dbg.data_ptr<int64_t>()) : nullptr;
  a.T = (int)x.size(0);
  a.H = (int)x.size(1);
  a.nsplit = a.H / HS;
  a.sk_iters = (int)sk_iters;
  a.rms_eps = (float)rms_eps;
  a.pre_eps = (float)pre_eps;
  a.sk_eps = (float)sk_eps;
  a.post_mult = (float)post_mult;
  a.norm_eps = (float)norm_eps;
  TORCH_CHECK(a.T >= 1 && a.T <= MAXT && a.H % HS == 0, "sm120_mhc: T in [1, 8], H % 128 == 0");
  TORCH_CHECK(a.nsplit <= 64 && a.T <= a.nsplit, "sm120_mhc: hidden size out of range");
  size_t smem = (size_t)(MIX * KP + MAXT * KP + 8 * 2 * 32 * 4) * sizeof(float);
  TORCH_CHECK(cudaFuncSetAttribute(sm120_mhc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) ==
                  cudaSuccess, "sm120_mhc: shared memory");
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(a.nsplit);
  cfg.blockDim = dim3(THREADS);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, sm120_mhc_kernel, a) == cudaSuccess, "sm120_mhc: launch");
}
"""

CPP_SRC = (
    "void run(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, "
    "const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, "
    "const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, "
    "const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, "
    "int64_t, double, double, double, double, double, int64_t, const torch::Tensor&);"
)

ENABLED = os.environ.get("VLLM_SM120_MHC", "0") == "1"
PDL = os.environ.get("VLLM_SM120_MHC_PDL", "1") == "1"
MAX_TOKENS = 8
HS = 128
_mod = None
_counters: dict = {}


def _get():
    global _mod
    if _mod is None:
        from torch.utils.cpp_extension import load_inline

        old = os.environ.get("TORCH_CUDA_ARCH_LIST")
        os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"
        try:
            _mod = load_inline(
                name="vllm_sm120_mhc_v10",
                cpp_sources=CPP_SRC,
                cuda_sources=CUDA_SRC,
                functions=["run"],
                extra_cuda_cflags=["-O3"],
                verbose=False,
            )
        finally:
            if old is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = old
    return _mod


def supports(x: torch.Tensor, residual: torch.Tensor, capture_aux: bool) -> bool:
    return (
        ENABLED
        and not capture_aux
        and x.is_cuda
        and x.dim() == 2
        and 1 <= x.shape[0] <= MAX_TOKENS
        and residual.dim() == 3
        and residual.shape[1] == 4
        and x.shape[1] % HS == 0
        and x.shape[1] // HS <= 64
        and x.is_contiguous()
        and residual.is_contiguous()
        and _is_sm120(x.device)
    )


_SM120: dict = {}


def _is_sm120(device: torch.device) -> bool:
    v = _SM120.get(device.index)
    if v is None:
        v = _SM120[device.index] = torch.cuda.get_device_capability(device)[0] == 12
    return v


def _counter(fn: torch.Tensor) -> torch.Tensor:
    # One self-resetting counter pair per call site (each sublayer has its own fn), zeroed once at creation.
    key = (fn.data_ptr(), fn.device.index)
    c = _counters.get(key)
    if c is None:
        c = torch.zeros(2, dtype=torch.int32, device=fn.device)
        _counters[key] = c
    return c


_NO_DBG: dict = {}


def mhc_shifted_post_pre(
    x,
    residual,
    post_layer_mix,
    comb_res_mix,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    pre_mix,
    norm_weight,
    norm_eps,
    dbg=None,
):
    """Same outputs as the TileLang path: residual, post (T, 4, 1), comb (T, 4, 4), layer input, next pre-mix."""
    T, hc, H = residual.shape
    new_residual = torch.empty_like(residual)
    post = torch.empty(T, hc, dtype=torch.float32, device=x.device)
    comb = torch.empty(T, hc * hc, dtype=torch.float32, device=x.device)
    y = torch.empty(T, H, dtype=torch.bfloat16, device=x.device)
    next_pre = torch.empty(T, hc, dtype=torch.float32, device=x.device)
    part = torch.empty(T, 26, H // HS, dtype=torch.float32, device=x.device)
    _get().run(
        x,
        residual,
        post_layer_mix.reshape(T, hc),
        comb_res_mix.reshape(T, hc, hc),
        fn,
        hc_scale,
        hc_base,
        pre_mix,
        norm_weight,
        new_residual,
        post,
        comb,
        y,
        next_pre,
        part,
        _counter(fn),
        int(sinkhorn_repeat),
        float(rms_eps),
        float(hc_pre_eps),
        float(hc_sinkhorn_eps),
        float(hc_post_mult_value),
        float(norm_eps),
        int(PDL),
        dbg
        if dbg is not None
        else _NO_DBG.setdefault(
            x.device, torch.empty(0, dtype=torch.int64, device=x.device)
        ),
    )
    return new_residual, post.unsqueeze(-1), comb.view(T, hc, hc), y, next_pre
