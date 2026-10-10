# SPDX-License-Identifier: Apache-2.0
"""Small-M bf16 GEMM for sm_120 decode: y[M, N] = x[M, K] @ w[N, K].T, M <= 32, bf16 or fp32 output.

Replaces cuBLAS for DSV4.1's bf16 projections at decode sizes, where cuBLAS picks sm_80 WMMA kernels
(``cutlass_80_wmma_tensorop_*``) with a separate split-K reduce, and for some shapes only 2 CTAs (30-38 us for a
0.7 MB weight). Design as small_m_tc: swap-AB ``mma.sync.m16n8k16`` bf16 (weights are the 16-row A operand,
tokens the 8-column B operand), 128 weight rows per CTA, K split over blockIdx.y, a cp.async weight pipeline whose
first stages are issued *before* ``griddepcontrol.wait`` (PDL), then a PDL-chained fixed-order split-K reduce.
Batched/strided for the grouped wo_a projection. Deterministic and batch-invariant (the split depends only on K).
Enable with VLLM_SMALL_M_BF16=1.
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
constexpr int BNT = 128, NTHR = 256, KC = 32, WROW = KC * 2 + 16;  // 80-byte smem rows: conflict-free fragments
#ifndef BF_STAGES
#define BF_STAGES 4
#endif

DEVI void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
DEVI void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
DEVI void mma_bf16(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// grid: x = 128-row weight tiles, y = K slices, z = batch. partial: [batch][S][PR][N] fp32
template <int NT>
__global__ void __launch_bounds__(NTHR, 1) bf16_tc(
    const __nv_bfloat16* __restrict__ x, long ldx, long bsx,
    const __nv_bfloat16* __restrict__ w, long ldw, long bsw,
    float* __restrict__ partial, int M, int N, int KS, int PR, int XROW) {
  constexpr int MT = NT * 8;
  extern __shared__ __align__(16) uint8_t smem[];
  uint8_t* Ws = smem;
  uint8_t* Xs = Ws + BF_STAGES * BNT * WROW;
  const int tid = threadIdx.x, n0 = blockIdx.x * BNT, k0 = blockIdx.y * KS, bz = blockIdx.z;
  const __nv_bfloat16* wb = w + bz * bsw;
  const __nv_bfloat16* xb = x + bz * bsx;
  const int nst = KS / KC;
  auto load_stage = [&](int st) {
    uint8_t* base = Ws + (st % BF_STAGES) * BNT * WROW;
    for (int i = tid; i < BNT * 4; i += NTHR) {
      int row = i >> 2, ch = i & 3, gr = n0 + row;
      uint8_t* dst = base + row * WROW + ch * 16;
      if (gr < N) __pipeline_memcpy_async(dst, wb + (size_t)gr * ldw + k0 + st * KC + ch * 8, 16);
      else *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
    }
  };
  // weights first: they do not depend on the previous kernel
  for (int st = 0; st < BF_STAGES - 1; st++) {
    if (st < nst) load_stage(st);
    __pipeline_commit();
  }
  pdl_wait();
  // the activation K-slice (tiny) into smem
  const int cpr = KS / 8;
  for (int i = tid; i < MT * cpr; i += NTHR) {
    int t = i / cpr, ch = i - t * cpr;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (t < M) v = *reinterpret_cast<const uint4*>(xb + (size_t)t * ldx + k0 + ch * 8);
    *reinterpret_cast<uint4*>(Xs + t * XROW + ch * 16) = v;
  }
  pdl_trigger();
  const int lane = tid & 31, warp = tid >> 5, g = lane >> 2, c = lane & 3, r0 = warp * 16;
  float acc[NT][4];
#pragma unroll
  for (int t = 0; t < NT; t++)
#pragma unroll
    for (int i = 0; i < 4; i++) acc[t][i] = 0.f;
  for (int st = 0; st < nst; st++) {
    __pipeline_wait_prior(BF_STAGES - 2);
    __syncthreads();
    if (st + BF_STAGES - 1 < nst) load_stage(st + BF_STAGES - 1);
    __pipeline_commit();
    const uint8_t* Wb = Ws + (st % BF_STAGES) * BNT * WROW;
#pragma unroll
    for (int kk = 0; kk < KC / 16; kk++) {
      const uint8_t* wr = Wb + kk * 32 + 4 * c;
      uint32_t a0 = *reinterpret_cast<const uint32_t*>(wr + (r0 + g) * WROW);
      uint32_t a1 = *reinterpret_cast<const uint32_t*>(wr + (r0 + g + 8) * WROW);
      uint32_t a2 = *reinterpret_cast<const uint32_t*>(wr + (r0 + g) * WROW + 16);
      uint32_t a3 = *reinterpret_cast<const uint32_t*>(wr + (r0 + g + 8) * WROW + 16);
#pragma unroll
      for (int t = 0; t < NT; t++) {
        const uint8_t* xr = Xs + (t * 8 + g) * XROW + (st * KC + kk * 16) * 2 + 4 * c;
        mma_bf16(acc[t], a0, a1, a2, a3, *reinterpret_cast<const uint32_t*>(xr),
                 *reinterpret_cast<const uint32_t*>(xr + 16));
      }
    }
  }
  __pipeline_wait_prior(0);
  float* base = partial + ((size_t)bz * gridDim.y + blockIdx.y) * PR * N;
#pragma unroll
  for (int t = 0; t < NT; t++) {
    int tok = t * 8 + 2 * c, nA = n0 + r0 + g, nB = nA + 8;
    if (nA < N) {
      if (tok < M) base[(size_t)tok * N + nA] = acc[t][0];
      if (tok + 1 < M) base[(size_t)(tok + 1) * N + nA] = acc[t][1];
    }
    if (nB < N) {
      if (tok < M) base[(size_t)tok * N + nB] = acc[t][2];
      if (tok + 1 < M) base[(size_t)(tok + 1) * N + nB] = acc[t][3];
    }
  }
}

// out[b][m][n] (row stride ldo, batch stride bso) = sum_s partial[b][s][m][n], fixed order
template <typename T>
__global__ void bf16_reduce(const float* __restrict__ partial, T* __restrict__ out, long ldo, long bso,
                            int M, int N, int S, int PR) {
  pdl_wait();
  pdl_trigger();
  int bz = blockIdx.y;
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)M * N) return;
  int m = i / N, n = i - (size_t)m * N;
  const float* p = partial + (size_t)bz * S * PR * N + (size_t)m * N + n;
  float a = 0.f;
#pragma unroll 8
  for (int s = 0; s < S; s++) a += __ldcg(p + (size_t)s * PR * N);
  if constexpr (sizeof(T) == 4) out[bz * bso + (size_t)m * ldo + n] = a;
  else out[bz * bso + (size_t)m * ldo + n] = __float2bfloat16(a);
}

static void launch(const void* fn, dim3 grid, dim3 block, size_t smem, cudaStream_t st, void** args, bool pdl) {
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid; cfg.blockDim = block; cfg.dynamicSmemBytes = smem; cfg.stream = st;
  cfg.attrs = attr; cfg.numAttrs = 1;
  TORCH_CHECK(cudaLaunchKernelExC(&cfg, fn, args) == cudaSuccess, "small_m_bf16: launch");
}

// x: [B][M][K] via (ldx, bsx); w: [B][N][K] via (ldw, bsw); out via (ldo, bso)
void run(const torch::Tensor& x, int64_t ldx, int64_t bsx, const torch::Tensor& w, int64_t ldw, int64_t bsw,
         const torch::Tensor& out, int64_t ldo, int64_t bso, const torch::Tensor& partial,
         int64_t B, int64_t M, int64_t N, int64_t K, int64_t S, int64_t pdl) {
  int KS = (int)(K / S);
  TORCH_CHECK(KS * S == K && KS % KC == 0, "small_m_bf16: K / S must be a multiple of 32");
  int MT = M <= 8 ? 8 : M <= 16 ? 16 : 32, NT = MT / 8, PR = MT;
  int XROW = ((KS * 2 + 127) / 128) * 128 + 16;
  size_t smem = (size_t)BF_STAGES * BNT * WROW + (size_t)MT * XROW;
  auto st = at::cuda::getCurrentCUDAStream();
  const void* fn = NT == 1 ? (const void*)bf16_tc<1> : NT == 2 ? (const void*)bf16_tc<2> : (const void*)bf16_tc<4>;
  TORCH_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) == cudaSuccess,
              "small_m_bf16: shared memory ", smem);
  const void* xp = x.data_ptr();
  const void* wp = w.data_ptr();
  float* pp = partial.data_ptr<float>();
  long lx = ldx, bx = bsx, lw = ldw, bw = bsw;
  int m = (int)M, n = (int)N;
  void* a1[] = {(void*)&xp, &lx, &bx, (void*)&wp, &lw, &bw, (void*)&pp, &m, &n, &KS, &PR, &XROW};
  launch(fn, dim3((N + BNT - 1) / BNT, (unsigned)S, (unsigned)B), dim3(NTHR), smem, st, a1, pdl);
  void* op = out.data_ptr();
  long lo = ldo, bo = bso;
  int s_ = (int)S;
  void* a2[] = {(void*)&pp, &op, &lo, &bo, &m, &n, &s_, &PR};
  dim3 rg((unsigned)((M * N + 255) / 256), (unsigned)B);
  if (out.scalar_type() == at::kFloat) launch((const void*)bf16_reduce<float>, rg, dim3(256), 0, st, a2, pdl);
  else launch((const void*)bf16_reduce<__nv_bfloat16>, rg, dim3(256), 0, st, a2, pdl);
}
"""

CPP_SRC = ("void run(const torch::Tensor&, int64_t, int64_t, const torch::Tensor&, int64_t, int64_t, "
           "const torch::Tensor&, int64_t, int64_t, const torch::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t, "
           "int64_t);")

ENABLED = os.environ.get("VLLM_SMALL_M_BF16", "0") == "1"
PDL = os.environ.get("VLLM_SMALL_M_BF16_PDL", "1") == "1"
MAX_M = 32
_mod = None


def _get():
    global _mod
    if _mod is None:
        from torch.utils.cpp_extension import load_inline

        old = os.environ.get("TORCH_CUDA_ARCH_LIST")
        os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"
        try:
            _mod = load_inline(name="vllm_small_m_bf16_v1", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC,
                               functions=["run"], extra_cuda_cflags=["-O3"], verbose=False)
        finally:
            if old is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = old
    return _mod


def split_for(N: int, K: int, B: int = 1) -> int:
    """K split, measured at M=6 on the DSV4.1 shapes: 16 slices, 32 when there are at most two 128-row tiles
    (tiny N). Falls back to the nearest valid split (K / S a multiple of 32). Depends only on (N, K, B)."""
    tiles = (N + 127) // 128 * B
    want = 32 if tiles <= 2 else 16
    cands = [s for s in range(1, 257) if K % s == 0 and (K // s) % 32 == 0]
    return min(cands, key=lambda s: (abs(s - want), -s)) if cands else 1


def _sm120(t: torch.Tensor) -> bool:
    return torch.cuda.get_device_capability(t.device)[0] == 12


def supports(x: torch.Tensor, w: torch.Tensor) -> bool:
    return (ENABLED and x.is_cuda and x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16 and x.dim() == 2
            and w.dim() == 2 and 1 <= x.shape[0] <= MAX_M and x.shape[1] == w.shape[1] and x.shape[1] % 32 == 0
            and x.stride(1) == 1 and x.stride(0) % 8 == 0 and x.data_ptr() % 16 == 0 and w.is_contiguous()
            and _sm120(x))


def mm(x: torch.Tensor, w: torch.Tensor, out_dtype: torch.dtype = torch.bfloat16, S: int | None = None) -> torch.Tensor:
    """x [M, K] @ w[N, K].T -> [M, N] in out_dtype (bf16 or fp32)."""
    M, K = x.shape
    N = w.shape[0]
    S = S or split_for(N, K)
    PR = 8 if M <= 8 else 16 if M <= 16 else 32
    partial = torch.empty(S * PR * N, dtype=torch.float32, device=x.device)
    out = torch.empty(M, N, dtype=out_dtype, device=x.device)
    _get().run(x, x.stride(0), 0, w, K, 0, out, N, 0, partial, 1, M, N, K, S, int(PDL))
    return out


def supports_grouped(x: torch.Tensor, w: torch.Tensor) -> bool:
    # x [M, G, K] (contiguous), w [G, N, K] (contiguous)
    return (ENABLED and x.is_cuda and x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16 and x.dim() == 3
            and w.dim() == 3 and 1 <= x.shape[0] <= MAX_M and x.shape[1] == w.shape[0] and x.shape[2] == w.shape[2]
            and x.shape[2] % 32 == 0 and x.is_contiguous() and w.is_contiguous() and x.data_ptr() % 16 == 0
            and _sm120(x))


def grouped_into(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor) -> None:
    """out[M, G, N] = per group g: x[:, g, :] @ w[g].T (out contiguous [M, G, N], bf16)."""
    M, G, K = x.shape
    N = w.shape[1]
    assert out.is_contiguous() and out.shape == (M, G, N)
    S = split_for(N, K, G)
    PR = 8 if M <= 8 else 16 if M <= 16 else 32
    partial = torch.empty(G * S * PR * N, dtype=torch.float32, device=x.device)
    _get().run(x, G * K, K, w, K, N * K, out, G * N, N, partial, G, M, N, K, S, int(PDL))
