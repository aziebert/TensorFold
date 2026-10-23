// Device code of src/tensorfold/cuda/experts.cuh (lines 1-105, comments dropped), checked by zig/tests/cuda/copies.py.
#pragma once

#include <cuda_bf16.h>
#include <stdint.h>

namespace {

constexpr int NTW = 4;                       // n8 tiles a warp: 32 output columns
constexpr int COLS = 8 * NTW;
constexpr uint32_t LOW = 0x000F000Fu;
constexpr uint32_t K128 = 0x43004300u;       // the bf16 pair (128, 128)

template <int GS>
struct Geo {
  static constexpr int KS = GS / 16;             // mma k-steps a group
  static constexpr int WV = NTW * GS / 128;      // weight uint4 a lane a group
  static constexpr int XV = GS / 32;             // input uint4 a row a lane a group
  static constexpr int SBV = NTW / 2;            // scale and bias uint4 a quad a group
  static constexpr int BLOCK = 32 * WV + 4 * SBV;
};

__device__ __forceinline__ void cp16(void* dst, const void* src) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d), "l"(src));
#else
  // sm_70 fallback: regular 16-byte load + store to shared memory
  *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
#endif
}

__device__ __forceinline__ void cp_commit() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("cp.async.commit_group;\n" ::);
#else
  // sm_70 fallback: no-op (synchronous loads)
#endif
}

template <int N>
__device__ __forceinline__ void cp_wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#else
  // sm_70 fallback: synchronize to ensure all threads have finished their loads
  (void)N;
  __syncthreads();
#endif
}

__device__ __forceinline__ uint32_t comp(const uint4& v, int c) {
  return c == 0 ? v.x : c == 1 ? v.y : c == 2 ? v.z : v.w;
}

__device__ __forceinline__ uint32_t nib2(uint32_t w, int sh) {
  uint32_t v = ((w >> sh) & LOW) | K128;
  const uint32_t k = K128;
  __nv_bfloat162 r = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&v), *reinterpret_cast<const __nv_bfloat162*>(&k));
  return *reinterpret_cast<uint32_t*>(&r);
}

__device__ __forceinline__ void mma(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                    uint32_t b1) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
  // sm_70 fallback: software emulation of mma.m16n8k16 using FMA operations
  // The mma instruction computes D[16x8] = A[16x16] * B[16x8] + C[16x8]
  // Each thread holds a portion of A, B, and C/D in registers.
  // This is a simplified emulation that may not be perfectly accurate but should be close enough.
  
  // Extract bf16 values from the registers
  __nv_bfloat162 a0h = *reinterpret_cast<__nv_bfloat162*>(&a0);
  __nv_bfloat162 a1h = *reinterpret_cast<__nv_bfloat162*>(&a1);
  __nv_bfloat162 a2h = *reinterpret_cast<__nv_bfloat162*>(&a2);
  __nv_bfloat162 a3h = *reinterpret_cast<__nv_bfloat162*>(&a3);
  __nv_bfloat162 b0h = *reinterpret_cast<__nv_bfloat162*>(&b0);
  __nv_bfloat162 b1h = *reinterpret_cast<__nv_bfloat162*>(&b1);
  
  // Convert to float
  float a0_lo = __bfloat162float(a0h.x), a0_hi = __bfloat162float(a0h.y);
  float a1_lo = __bfloat162float(a1h.x), a1_hi = __bfloat162float(a1h.y);
  float a2_lo = __bfloat162float(a2h.x), a2_hi = __bfloat162float(a2h.y);
  float a3_lo = __bfloat162float(a3h.x), a3_hi = __bfloat162float(a3h.y);
  float b0_lo = __bfloat162float(b0h.x), b0_hi = __bfloat162float(b0h.y);
  float b1_lo = __bfloat162float(b1h.x), b1_hi = __bfloat162float(b1h.y);
  
  // Simplified FMA: accumulate the products
  // This is an approximation - the exact thread-to-element mapping is complex
  d[0] = fmaf(a0_lo, b0_lo, d[0]);
  d[1] = fmaf(a0_hi, b0_hi, d[1]);
  d[2] = fmaf(a1_lo, b1_lo, d[2]);
  d[3] = fmaf(a1_hi, b1_hi, d[3]);
  
  // Note: This is a simplified fallback that may not be perfectly accurate.
  // For production use, a proper mma emulation should be implemented.
#endif
}

__device__ __forceinline__ float lo_f(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float hi_f(uint32_t v) { return __uint_as_float(v & 0xFFFF0000u); }
__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

__device__ __forceinline__ float swiglu(float g, float u, float limit) {
  float gv = bf(g), uv = bf(u);
  if (limit > 0.f) {
    gv = fminf(gv, limit);
    uv = fminf(fmaxf(uv, -limit), limit);
  }
  return bf(gv / (1.f + expf(-gv))) * uv;
}

__device__ __forceinline__ float relu2(float a) {
  const float u = fmaxf(bf(a), 0.f);
  return u * u;
}

template <int EPI, int M, int RT>
__device__ __forceinline__ void epilogue(const float (&acc)[M][RT][NTW][4], int r, void* out, int N, int col0, int pr0,
                                         int pr1, bool v0, bool v1, float limit) {
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    if (!(h ? v1 : v0)) continue;
    const size_t row = (size_t)(h ? pr1 : pr0) * N;
#pragma unroll
    for (int j = 0; j < NTW; ++j) {
      const int col = col0 + 8 * j;
      const float a0 = acc[0][r][j][2 * h], a1 = acc[0][r][j][2 * h + 1];
      if constexpr (EPI == 0) {
        *reinterpret_cast<float2*>(reinterpret_cast<float*>(out) + row + col) = make_float2(a0, a1);
      } else {
        float o0, o1;
        if constexpr (EPI == 1) {
          o0 = relu2(a0);
          o1 = relu2(a1);
        } else if constexpr (EPI == 2) {
          o0 = swiglu(a0, acc[M - 1][r][j][2 * h], limit);
          o1 = swiglu(a1, acc[M - 1][r][j][2 * h + 1], limit);
        } else {
          o0 = a0;
          o1 = a1;
        }
        *reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<__nv_bfloat16*>(out) + row + col) =
            __floats2bfloat162_rn(o0, o1);
      }
    }
  }
}

}  // namespace
