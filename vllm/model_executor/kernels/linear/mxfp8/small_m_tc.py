# SPDX-License-Identifier: Apache-2.0
"""Small-M MXFP8 GEMM for sm_120a (DeepSeek V4.1 decode), from ~/kernel-play/mxfp8/tc (2026-10-03).

Block-scaled tensor-core mma (m16n8k32, UE8M0) with swap-AB and a split-K over blockIdx.y, reading the same
operands as FlashInfer's CUTLASS mm_mxfp8: fp8 weight [N, K] and activations [M, K] with 128x4-swizzled UE8M0
scales. FlashInfer picks a 128-row tile and no split-K at M <= 64, which launches 5-40 CTAs on 188 SMs; this
kernel spreads K over S slices so the weight streams from every SM. The reduction order depends only on K, so
outputs are batch-invariant. Enable with VLLM_MXFP8_SMALL_M_TC=1; applies to M <= VLLM_MXFP8_SMALL_M_MAX (64).
"""
import os

import torch

CUDA_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <cstdint>

#define DEVI __device__ __forceinline__

DEVI int sf_off(int row, int kblk, int nKTiles) {
  return ((row >> 7) * nKTiles + (kblk >> 2)) * 512 + (row & 31) * 16 + ((row >> 5) & 3) * 4 + (kblk & 3);
}

DEVI void mma_bs(uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                 uint32_t b0, uint32_t b1,
                 float& d0, float& d1, float& d2, float& d3,
                 uint8_t sfa, uint8_t sfb) {
  float c0 = d0, c1 = d1, c2 = d2, c3 = d3;
  asm volatile(
    "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},{%14},{%15,%16},{%17},{%18,%19};\n"
    : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
      "f"(c0), "f"(c1), "f"(c2), "f"(c3),
      "r"((uint32_t)sfa), "h"((uint16_t)0), "h"((uint16_t)0),
      "r"((uint32_t)sfb), "h"((uint16_t)0), "h"((uint16_t)0));
}

constexpr bool TC_NOMMA = false;  // debug: skip the block-scaled mma to isolate faults
constexpr bool TC_NOPIPE = false; // debug: synchronous loads instead of cp.async
constexpr int WROW = 48;      // smem row stride for W (32 useful bytes, stride 48 -> no bank conflicts)
constexpr int XROW = 48;      // smem row stride for X
#ifndef TC_BLOCK_ROWS
#define TC_BLOCK_ROWS 128
#endif
#ifndef TC_NOSCALE
#define TC_NOSCALE 0
#endif

// BNT weight rows per block (BNT/16 warps), NT token groups of 8, STAGES pipeline depth.
template <int NT, int STAGES, int BNT>
__global__ void __launch_bounds__(BNT * 2) mxfp8_tc(
    const uint8_t* __restrict__ xq, const uint8_t* __restrict__ xs,
    const uint8_t* __restrict__ wq, const uint8_t* __restrict__ ws,
    float* __restrict__ partial,
    int M, int N, int K, int PR, int KSB, int nKTiles) {
  constexpr int MT = NT * 8;
  constexpr int XSTAGE = MT * XROW;
  constexpr int NTHR = BNT * 2;
  constexpr int WSTAGE = BNT * WROW;

  extern __shared__ uint8_t smem[];
  uint8_t* Ws  = smem;                              // STAGES * WSTAGE
  uint8_t* Xs  = Ws + STAGES * WSTAGE;              // STAGES * XSTAGE
  uint8_t* sWs = Xs + STAGES * XSTAGE;              // BNT * KSB
  uint8_t* sXs = sWs + BNT * KSB;                   // MT * KSB

  int tid = threadIdx.x;
  int n0 = blockIdx.x * BNT;
  int kb0 = blockIdx.y * KSB;

  auto load_stage = [&](int stage) {
    int buf = stage % STAGES;
    int row = tid >> 1, half = tid & 1;
    int g = n0 + row;
    uint8_t* wdst = Ws + buf * WSTAGE + row * WROW + half * 16;
    if (TC_NOPIPE) {
      *reinterpret_cast<uint4*>(wdst) = (g < N)
          ? *reinterpret_cast<const uint4*>(wq + (size_t)g * K + (size_t)(kb0 + stage) * 32 + half * 16)
          : make_uint4(0, 0, 0, 0);
    } else if (g < N) {
      __pipeline_memcpy_async(wdst, wq + (size_t)g * K + (size_t)(kb0 + stage) * 32 + half * 16, 16);
    } else {
      *reinterpret_cast<uint4*>(wdst) = make_uint4(0, 0, 0, 0);
    }
    for (int i = tid; i < MT * 2; i += NTHR) {
      int t = i >> 1, xh = i & 1;
      uint8_t* xdst = Xs + buf * XSTAGE + t * XROW + xh * 16;
      if (TC_NOPIPE) {
        *reinterpret_cast<uint4*>(xdst) = (t < M)
            ? *reinterpret_cast<const uint4*>(xq + (size_t)t * K + (size_t)(kb0 + stage) * 32 + xh * 16)
            : make_uint4(0, 0, 0, 0);
      } else if (t < M) {
        __pipeline_memcpy_async(xdst, xq + (size_t)t * K + (size_t)(kb0 + stage) * 32 + xh * 16, 16);
      } else {
        *reinterpret_cast<uint4*>(xdst) = make_uint4(0, 0, 0, 0);
      }
    }
  };

  // Issue the weight/activation prologue first so those DRAM fetches overlap the scale preload.
  if (!TC_NOPIPE) {
    for (int s = 0; s < STAGES - 1; s++) {
      if (s < KSB) load_stage(s);
      __pipeline_commit();
    }
  }

  // --- scales for the whole K-slice, de-swizzled into smem (once per block) ---
  for (int i = tid; i < BNT * KSB; i += NTHR) {
    int row = i / KSB, kb = i - row * KSB;
    int g = n0 + row;
    sWs[i] = (g < N && !TC_NOSCALE) ? ws[sf_off(g, kb0 + kb, nKTiles)] : (uint8_t)0x7F;
  }
  for (int i = tid; i < MT * KSB; i += NTHR) {
    int t = i / KSB, kb = i - t * KSB;
    sXs[i] = (t < M) ? xs[sf_off(t, kb0 + kb, nKTiles)] : (uint8_t)0x7F;
  }

  float acc[NT][4];
  #pragma unroll
  for (int t = 0; t < NT; t++)
    #pragma unroll
    for (int i = 0; i < 4; i++) acc[t][i] = 0.f;

  __syncthreads();

  int lane = tid & 31, warp = tid >> 5;
  int g = lane >> 2, c = lane & 3;
  int r0 = warp * 16;

  for (int kb = 0; kb < KSB; kb++) {
    int cur = kb % STAGES;
    if (TC_NOPIPE) {
      load_stage(kb);
      __syncthreads();
    } else {
      // One barrier per step: wait for stage kb, then (after the barrier proves the previous
      // step's readers are done) issue the load that overwrites the buffer they used.
      __pipeline_wait_prior(STAGES - 2);
      __syncthreads();
      int future = kb + STAGES - 1;
      if (future < KSB) load_stage(future);
      __pipeline_commit();
    }

    const uint8_t* Wb = Ws + cur * WSTAGE;
    const uint8_t* Xb = Xs + cur * XSTAGE;
    uint32_t a0 = *reinterpret_cast<const uint32_t*>(Wb + (r0 + g) * WROW + 4 * c);
    uint32_t a1 = *reinterpret_cast<const uint32_t*>(Wb + (r0 + g + 8) * WROW + 4 * c);
    uint32_t a2 = *reinterpret_cast<const uint32_t*>(Wb + (r0 + g) * WROW + 16 + 4 * c);
    uint32_t a3 = *reinterpret_cast<const uint32_t*>(Wb + (r0 + g + 8) * WROW + 16 + 4 * c);
    uint8_t sfa = sWs[(r0 + g + (c & 1) * 8) * KSB + kb];

    #pragma unroll
    for (int t = 0; t < NT; t++) {
      uint32_t b0 = *reinterpret_cast<const uint32_t*>(Xb + (t * 8 + g) * XROW + 4 * c);
      uint32_t b1 = *reinterpret_cast<const uint32_t*>(Xb + (t * 8 + g) * XROW + 16 + 4 * c);
      uint8_t sfb = sXs[(t * 8 + g) * KSB + kb];
      if (TC_NOMMA)
        acc[t][0] += (float)(a0 + b0 + sfa + sfb);
      else
        mma_bs(a0, a1, a2, a3, b0, b1, acc[t][0], acc[t][1], acc[t][2], acc[t][3], sfa, sfb);
    }
    if (TC_NOPIPE) __syncthreads();
  }

  if (!TC_NOPIPE) __pipeline_wait_prior(0);  // drain any in-flight stage loads before exit

  // --- write partials: D[weight row][token] ---
  float* base = partial + (size_t)blockIdx.y * PR * N;
  #pragma unroll
  for (int t = 0; t < NT; t++) {
    int tok = t * 8 + 2 * c;
    int nA = n0 + r0 + g, nB = nA + 8;
    if (nA < N) {
      if (tok < M)     base[(size_t)tok * N + nA] = acc[t][0];
      if (tok + 1 < M) base[(size_t)(tok + 1) * N + nA] = acc[t][1];
    }
    if (nB < N) {
      if (tok < M)     base[(size_t)tok * N + nB] = acc[t][2];
      if (tok + 1 < M) base[(size_t)(tok + 1) * N + nB] = acc[t][3];
    }
  }
}

__global__ void reduce_splitk(const float* __restrict__ partial, __nv_bfloat16* __restrict__ out,
                              int M, int N, int S, int PR) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)M * N) return;
  int m = i / N, n = i % N;
  float acc = 0.f;
  for (int s = 0; s < S; s++) acc += partial[((size_t)s * PR + m) * N + n];
  out[i] = __float2bfloat16(acc);
}

void run(const torch::Tensor& xq, const torch::Tensor& xs,
         const torch::Tensor& wq, const torch::Tensor& ws,
         const torch::Tensor& partial, const torch::Tensor& out,
         int64_t M, int64_t N, int64_t K, int64_t S, int64_t PR) {
  int nKTiles = (int)((K / 32 + 3) / 4);
  int KSB = (int)((K / 32) / S);
  TORCH_CHECK(KSB * S * 32 == K, "K must be divisible by 32*S");
  int MT = (M <= 8) ? 8 : (M <= 16) ? 16 : (M <= 32) ? 32 : 64;
  int NT = MT / 8;
  auto stream = at::cuda::getCurrentCUDAStream();

  // kb0 is 0: each block derives its slice from blockIdx.y * KSB inside the kernel.
  #define LAUNCH2(NT_, BNT_) \
    do { \
      dim3 grid((int)((N + BNT_ - 1) / BNT_), (unsigned)S); \
      size_t smem = TC_STAGES * ((size_t)BNT_ * WROW + (size_t)NT_ * 8 * XROW) \
                    + (size_t)BNT_ * KSB + (size_t)NT_ * 8 * KSB; \
      TORCH_CHECK(cudaFuncSetAttribute(mxfp8_tc<NT_, TC_STAGES, BNT_>, \
                           cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) == cudaSuccess, \
                  "small_m_tc: shared memory ", smem); \
      mxfp8_tc<NT_, TC_STAGES, BNT_><<<grid, BNT_ * 2, smem, stream>>>( \
          xq.data_ptr<uint8_t>(), xs.data_ptr<uint8_t>(), wq.data_ptr<uint8_t>(), \
          ws.data_ptr<uint8_t>(), partial.data_ptr<float>(), \
          (int)M, (int)N, (int)K, (int)PR, KSB, nKTiles); \
    } while (0)

  constexpr int BNT = TC_BLOCK_ROWS;
  if (NT == 1) LAUNCH2(1, BNT);
  else if (NT == 2) LAUNCH2(2, BNT);
  else if (NT == 4) LAUNCH2(4, BNT);
  else LAUNCH2(8, BNT);
  #undef LAUNCH2

  size_t tot = (size_t)M * N;
  reduce_splitk<<<(tot + 255) / 256, 256, 0, stream>>>(
      partial.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), (int)M, (int)N, (int)S, (int)PR);
}

// --- bf16 activations, MXFP8 quantization fused: each block quantizes its own K-slice into shared memory with
// FlashInfer's mxfp8_quantize semantics (amax/448 -> UE8M0 rounded up, x * 2^-e, RN satfinite e4m3), so the
// operands are bit-identical to quantize-then-GEMM and the separate quantize kernel disappears.
// Programmatic dependent launch: wait for the previous kernel's results / let the next kernel start its prologue.
// Both are no-ops when the kernel was not launched with programmatic stream serialization.
DEVI void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
DEVI void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

__global__ void reduce_splitk_pdl(const float* __restrict__ partial, __nv_bfloat16* __restrict__ out,
                                  int M, int N, int S, int PR) {
  pdl_wait();
  pdl_trigger();
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)M * N) return;
  int m = i / N, n = i % N;
  float acc = 0.f;
  for (int s = 0; s < S; s++) acc += partial[((size_t)s * PR + m) * N + n];
  out[i] = __float2bfloat16(acc);
}

DEVI uint32_t ue8m0_ceil(float v) {
  if (!(v > 0.f)) return 0u;
  uint32_t bits = __float_as_uint(v);
  uint32_t e = (bits >> 23) & 255u, m = bits & 0x7FFFFFu;
  uint32_t bump = (m != 0u && !(e == 0u && m <= 0x400000u)) ? 1u : 0u;
  uint32_t r = e + bump;
  return r > 254u ? 254u : r;
}
DEVI float ue8m0_inv(uint32_t u) {
  if (u == 0u) return 0.f;
  int ne = 254 - (int)u;
  return __uint_as_float((uint32_t)(ne < 0 ? 0 : ne) << 23);
}
DEVI uint32_t e4m3x2(float lo, float hi) {
  lo = fmaxf(fminf(lo, 448.f), -448.f);
  hi = fmaxf(fminf(hi, 448.f), -448.f);
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
  return (uint32_t)r;
}

template <int NT, int STAGES, int BNT>
__global__ void __launch_bounds__(BNT * 2) mxfp8_tc_bf16(
    const uint16_t* __restrict__ xb, int ldx,
    const uint8_t* __restrict__ wq, const uint8_t* __restrict__ ws,
    float* __restrict__ partial,
    int M, int N, int K, int PR, int KSB, int nKTiles, int XQROW) {
  constexpr int MT = NT * 8;
  constexpr int NTHR = BNT * 2;
  constexpr int WSTAGE = BNT * WROW;

  extern __shared__ uint8_t smem[];
  uint8_t* Ws  = smem;                              // STAGES * WSTAGE
  uint8_t* Xq  = Ws + STAGES * WSTAGE;              // MT * XQROW (this block's whole quantized K-slice)
  uint8_t* sWs = Xq + MT * XQROW;                   // BNT * KSB
  uint8_t* sXs = sWs + BNT * KSB;                   // MT * KSB

  int tid = threadIdx.x;
  int n0 = blockIdx.x * BNT;
  int kb0 = blockIdx.y * KSB;

  auto load_stage = [&](int stage) {
    int buf = stage % STAGES;
    int row = tid >> 1, half = tid & 1;
    int g = n0 + row;
    uint8_t* wdst = Ws + buf * WSTAGE + row * WROW + half * 16;
    if (g < N) {
      __pipeline_memcpy_async(wdst, wq + (size_t)g * K + (size_t)(kb0 + stage) * 32 + half * 16, 16);
    } else {
      *reinterpret_cast<uint4*>(wdst) = make_uint4(0, 0, 0, 0);
    }
  };
  for (int s = 0; s < STAGES - 1; s++) {
    if (s < KSB) load_stage(s);
    __pipeline_commit();
  }

  for (int i = tid; i < BNT * KSB; i += NTHR) {
    int row = i / KSB, kb = i - row * KSB;
    int g = n0 + row;
    sWs[i] = (g < N) ? ws[sf_off(g, kb0 + kb, nKTiles)] : (uint8_t)0x7F;
  }

  // Everything above reads only weights, so under PDL it overlaps the previous kernel; activations come after.
  pdl_wait();

  // 4 threads per 32-element group, 8 bf16 each; the 4 lanes of a group are adjacent in one warp.
  const int items = MT * KSB * 4;
  for (int base = 0; base < items; base += NTHR) {
    int i = base + tid;
    int grp = i >> 2, part = i & 3;
    int t = grp / KSB, kb = grp - t * KSB;
    uint4 raw = make_uint4(0, 0, 0, 0);
    if (i < items && t < M)
      raw = *reinterpret_cast<const uint4*>(xb + (size_t)t * ldx + (size_t)(kb0 + kb) * 32 + part * 8);
    uint32_t wv[4] = {raw.x, raw.y, raw.z, raw.w};
    float v[8];
    #pragma unroll
    for (int j = 0; j < 4; j++) {
      v[2 * j] = __uint_as_float(wv[j] << 16);
      v[2 * j + 1] = __uint_as_float(wv[j] & 0xFFFF0000u);
    }
    float amax = 0.f;
    #pragma unroll
    for (int j = 0; j < 8; j++) amax = fmaxf(amax, fabsf(v[j]));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    uint32_t u = ue8m0_ceil(amax * (1.0f / 448.0f));
    float inv = ue8m0_inv(u);
    if (i < items) {
      uint32_t lo = e4m3x2(v[0] * inv, v[1] * inv) | (e4m3x2(v[2] * inv, v[3] * inv) << 16);
      uint32_t hi = e4m3x2(v[4] * inv, v[5] * inv) | (e4m3x2(v[6] * inv, v[7] * inv) << 16);
      *reinterpret_cast<uint2*>(Xq + t * XQROW + kb * 32 + part * 8) = make_uint2(lo, hi);
      if (part == 0) sXs[t * KSB + kb] = (t < M) ? (uint8_t)u : (uint8_t)0x7F;
    }
  }

  float acc[NT][4];
  #pragma unroll
  for (int t = 0; t < NT; t++)
    #pragma unroll
    for (int i = 0; i < 4; i++) acc[t][i] = 0.f;

  __syncthreads();
  pdl_trigger();

  int lane = tid & 31, warp = tid >> 5;
  int g = lane >> 2, c = lane & 3;
  int r0 = warp * 16;

  for (int kb = 0; kb < KSB; kb++) {
    int cur = kb % STAGES;
    __pipeline_wait_prior(STAGES - 2);
    __syncthreads();
    int future = kb + STAGES - 1;
    if (future < KSB) load_stage(future);
    __pipeline_commit();

    const uint8_t* Wb = Ws + cur * WSTAGE;
    uint32_t a0 = *reinterpret_cast<const uint32_t*>(Wb + (r0 + g) * WROW + 4 * c);
    uint32_t a1 = *reinterpret_cast<const uint32_t*>(Wb + (r0 + g + 8) * WROW + 4 * c);
    uint32_t a2 = *reinterpret_cast<const uint32_t*>(Wb + (r0 + g) * WROW + 16 + 4 * c);
    uint32_t a3 = *reinterpret_cast<const uint32_t*>(Wb + (r0 + g + 8) * WROW + 16 + 4 * c);
    uint8_t sfa = sWs[(r0 + g + (c & 1) * 8) * KSB + kb];

    #pragma unroll
    for (int t = 0; t < NT; t++) {
      const uint8_t* xr = Xq + (t * 8 + g) * XQROW + kb * 32;
      uint32_t b0 = *reinterpret_cast<const uint32_t*>(xr + 4 * c);
      uint32_t b1 = *reinterpret_cast<const uint32_t*>(xr + 16 + 4 * c);
      uint8_t sfb = sXs[(t * 8 + g) * KSB + kb];
      mma_bs(a0, a1, a2, a3, b0, b1, acc[t][0], acc[t][1], acc[t][2], acc[t][3], sfa, sfb);
    }
  }
  __pipeline_wait_prior(0);

  float* base = partial + (size_t)blockIdx.y * PR * N;
  #pragma unroll
  for (int t = 0; t < NT; t++) {
    int tok = t * 8 + 2 * c;
    int nA = n0 + r0 + g, nB = nA + 8;
    if (nA < N) {
      if (tok < M)     base[(size_t)tok * N + nA] = acc[t][0];
      if (tok + 1 < M) base[(size_t)(tok + 1) * N + nA] = acc[t][1];
    }
    if (nB < N) {
      if (tok < M)     base[(size_t)tok * N + nB] = acc[t][2];
      if (tok + 1 < M) base[(size_t)(tok + 1) * N + nB] = acc[t][3];
    }
  }
}

void run_bf16(const torch::Tensor& xb, const torch::Tensor& wq, const torch::Tensor& ws,
              const torch::Tensor& partial, const torch::Tensor& out,
              int64_t M, int64_t N, int64_t K, int64_t S, int64_t PR, int64_t pdl) {
  int nKTiles = (int)((K / 32 + 3) / 4);
  int KSB = (int)((K / 32) / S);
  TORCH_CHECK(KSB * S * 32 == K, "K must be divisible by 32*S");
  TORCH_CHECK(xb.stride(1) == 1 && xb.stride(0) % 8 == 0 && (reinterpret_cast<uintptr_t>(xb.data_ptr()) & 15) == 0,
              "bf16 activations need unit column stride and 16-byte aligned rows");
  int XQROW = ((KSB * 32 + 127) / 128) * 128 + 16;
  int MT = (M <= 8) ? 8 : (M <= 16) ? 16 : (M <= 32) ? 32 : 64;
  int NT = MT / 8;
  auto stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  #define LAUNCHB(NT_, BNT_) \
    do { \
      size_t smem = TC_STAGES_B * (size_t)BNT_ * WROW + (size_t)NT_ * 8 * XQROW \
                    + (size_t)BNT_ * KSB + (size_t)NT_ * 8 * KSB; \
      auto kern = mxfp8_tc_bf16<NT_, TC_STAGES_B, BNT_>; \
      TORCH_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) == cudaSuccess, \
                  "small_m_tc: shared memory ", smem); \
      cudaLaunchConfig_t cfg = {}; \
      cfg.gridDim = dim3((int)((N + BNT_ - 1) / BNT_), (unsigned)S); cfg.blockDim = dim3(BNT_ * 2); \
      cfg.dynamicSmemBytes = smem; cfg.stream = stream; cfg.attrs = attr; cfg.numAttrs = 1; \
      TORCH_CHECK(cudaLaunchKernelEx(&cfg, kern, reinterpret_cast<const uint16_t*>(xb.data_ptr()), (int)xb.stride(0), \
          (const uint8_t*)wq.data_ptr<uint8_t>(), (const uint8_t*)ws.data_ptr<uint8_t>(), partial.data_ptr<float>(), \
          (int)M, (int)N, (int)K, (int)PR, KSB, nKTiles, XQROW) == cudaSuccess, "small_m_tc: launch"); \
    } while (0)
  constexpr int BNT = TC_BLOCK_ROWS;
  if (NT == 1) LAUNCHB(1, BNT);
  else if (NT == 2) LAUNCHB(2, BNT);
  else if (NT == 4) LAUNCHB(4, BNT);
  else LAUNCHB(8, BNT);
  #undef LAUNCHB
  size_t tot = (size_t)M * N;
  cudaLaunchConfig_t rcfg = {};
  rcfg.gridDim = dim3((unsigned)((tot + 255) / 256)); rcfg.blockDim = dim3(256);
  rcfg.stream = stream; rcfg.attrs = attr; rcfg.numAttrs = 1;
  TORCH_CHECK(cudaLaunchKernelEx(&rcfg, reduce_splitk_pdl, (const float*)partial.data_ptr<float>(),
                                 reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
                                 (int)M, (int)N, (int)S, (int)PR) == cudaSuccess, "small_m_tc: reduce launch");
}
"""

CPP_SRC = ("void run(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, "
           "const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, "
           "int64_t, int64_t, int64_t, int64_t, int64_t);\n"
           "void run_bf16(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, "
           "const torch::Tensor&, const torch::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);")

ENABLED = os.environ.get("VLLM_MXFP8_SMALL_M_TC", "0") == "1"
# 1 = also take bf16 activations and quantize inside the GEMM (no separate mxfp8_quantize kernel).
FUSE_QUANT = os.environ.get("VLLM_MXFP8_SMALL_M_FUSE_QUANT", "0") == "1"
MAX_M = int(os.environ.get("VLLM_MXFP8_SMALL_M_MAX", "64"))
# Comma-separated NxK shapes to use the kernel for (empty = all). At bs1 the shared expert's GEMMs run on a side
# stream next to the routed experts; a full-GPU kernel there starves the critical path, so restrict to serial GEMMs.
ONLY = {tuple(int(v) for v in t.split("x")) for t in os.environ.get("VLLM_MXFP8_SMALL_M_ONLY", "").split(",") if t}
SMEM_LIMIT = 99 * 1024
# PDL launch (weights stream in while the previous kernel finishes) and the depth of the weight prologue.
PDL = os.environ.get("VLLM_MXFP8_SMALL_M_PDL", "1") == "1"
STAGES_B = int(os.environ.get("VLLM_MXFP8_SMALL_M_STAGES", "6"))
# K splits tuned on gpu07 for the DSv4.1 TP8 rank shapes (N, K); other shapes use split_for().
_S_TUNED = {(1792, 5120): 10, (4096, 1280): 8, (5120, 1024): 8, (576, 5120): 20, (5120, 288): 3}
_mod = None


def _get():
    global _mod
    if _mod is None:
        from torch.utils.cpp_extension import load_inline

        # The block-scaled mma only assembles for the arch-specific target (.target sm_120a).
        old = os.environ.get("TORCH_CUDA_ARCH_LIST")
        os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0a"
        try:
            _mod = load_inline(
                name=f"vllm_mxfp8_small_m_tc_v4_s{STAGES_B}",
                cpp_sources=CPP_SRC,
                cuda_sources=CUDA_SRC,
                functions=["run", "run_bf16"],
                extra_cuda_cflags=["-O3", "--use_fast_math", f"-DTC_STAGES=3", f"-DTC_STAGES_B={STAGES_B}", "-DTC_BLOCK_ROWS=128"],
                verbose=False,
            )
        finally:
            if old is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = old
    return _mod


def split_for(N: int, K: int) -> int:
    s = _S_TUNED.get((N, K))
    if s is not None:
        return s
    kb, tiles = K // 32, (N + 127) // 128
    best = 1
    for s in range(1, kb + 1):
        if kb % s == 0 and kb // s >= 4 and tiles * s <= 384:
            best = s
    return best


def _pr(M: int) -> int:
    return 8 if M <= 8 else 16 if M <= 16 else 32 if M <= 32 else 64


def supports(M: int, N: int, K: int) -> bool:
    if not (ENABLED and 0 < M <= MAX_M and K % 128 == 0 and N >= 128 and (not ONLY or (N, K) in ONLY)):
        return False
    ksb = K // 32 // split_for(N, K)
    return 3 * (128 * 48 + _pr(M) * 48) + (128 + _pr(M)) * ksb <= SMEM_LIMIT


def gemm(xq: torch.Tensor, xs: torch.Tensor, wq: torch.Tensor, ws: torch.Tensor,
         out_dtype: torch.dtype) -> torch.Tensor:
    """out[M, N] = (xq * xs) @ (wq * ws).T; xq [M, K] fp8, wq [N, K] fp8, xs/ws 128x4-swizzled UE8M0."""
    assert out_dtype == torch.bfloat16
    M, K = xq.shape
    N = wq.shape[0]
    S = split_for(N, K)
    PR = 8 if M <= 8 else 16 if M <= 16 else 32 if M <= 32 else 64
    partial = torch.empty((S, PR, N), device=xq.device, dtype=torch.float32)
    out = torch.empty((M, N), device=xq.device, dtype=out_dtype)
    _get().run(xq.view(torch.uint8), xs.view(torch.uint8), wq.view(torch.uint8), ws.view(torch.uint8),
               partial, out, M, N, K, S, PR)
    return out


def supports_bf16(x: torch.Tensor, N: int, K: int) -> bool:
    if not (FUSE_QUANT and x.dtype == torch.bfloat16 and x.dim() == 2 and supports(x.shape[0], N, K)
            and x.stride(1) == 1 and x.stride(0) % 8 == 0 and x.data_ptr() % 16 == 0):
        return False
    ksb = K // 32 // split_for(N, K)
    xqrow = (ksb * 32 + 127) // 128 * 128 + 16
    return STAGES_B * 128 * 48 + _pr(x.shape[0]) * xqrow + (128 + _pr(x.shape[0])) * ksb <= SMEM_LIMIT


def gemm_bf16(x: torch.Tensor, wq: torch.Tensor, ws: torch.Tensor) -> torch.Tensor:
    """out[M, N] = mxfp8_quantize(x) @ (wq * ws).T with the quantization done per K-slice inside the GEMM."""
    M, K = x.shape
    N = wq.shape[0]
    S = split_for(N, K)
    PR = 8 if M <= 8 else 16 if M <= 16 else 32 if M <= 32 else 64
    partial = torch.empty((S, PR, N), device=x.device, dtype=torch.float32)
    out = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
    _get().run_bf16(x, wq.view(torch.uint8), ws.view(torch.uint8), partial, out, M, N, K, S, PR, int(PDL))
    return out
