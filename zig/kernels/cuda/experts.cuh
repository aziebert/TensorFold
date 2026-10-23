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
  // sm_70 fallback: CORRECT software emulation of
  // mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32.
  // V100 has no bf16 tensor cores, so we gather the full A rows and B columns
  // across the warp with __shfl_sync and accumulate exact fp32 dot products.
  // Authoritative PTX layout (g = lane>>2, t = lane&3):
  //   a0=(A[g][2t],A[g][2t+1])       a1=(A[g+8][2t],A[g+8][2t+1])
  //   a2=(A[g][2t+8],A[g][2t+9])     a3=(A[g+8][2t+8],A[g+8][2t+9])
  //   b0=(B[2t][g],B[2t+1][g])       b1=(B[2t+8][g],B[2t+9][g])   (B is K x N)
  //   d0=D[g][2t] d1=D[g][2t+1] d2=D[g+8][2t] d3=D[g+8][2t+1]
  //   D[i][n] = sum_k A[i][k]*B[k][n]
  {
    const unsigned FULL = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2;
    const int t = lane & 3;

    // Gather A rows g and g+8 (all 16 k-values each).
    float ag[16], ag8[16];
    #pragma unroll
    for (int tt = 0; tt < 4; ++tt) {
      const int src = g * 4 + tt;
      uint32_t r0 = __shfl_sync(FULL, a0, src);
      uint32_t r2 = __shfl_sync(FULL, a2, src);
      uint32_t q0 = __shfl_sync(FULL, a1, src);
      uint32_t q2 = __shfl_sync(FULL, a3, src);
      const __nv_bfloat162 p0 = *reinterpret_cast<__nv_bfloat162*>(&r0);
      const __nv_bfloat162 p2 = *reinterpret_cast<__nv_bfloat162*>(&r2);
      const __nv_bfloat162 qA = *reinterpret_cast<__nv_bfloat162*>(&q0);
      const __nv_bfloat162 qB = *reinterpret_cast<__nv_bfloat162*>(&q2);
      ag[2 * tt] = __bfloat162float(p0.x); ag[2 * tt + 1] = __bfloat162float(p0.y);
      ag[2 * tt + 8] = __bfloat162float(p2.x); ag[2 * tt + 9] = __bfloat162float(p2.y);
      ag8[2 * tt] = __bfloat162float(qA.x); ag8[2 * tt + 1] = __bfloat162float(qA.y);
      ag8[2 * tt + 8] = __bfloat162float(qB.x); ag8[2 * tt + 9] = __bfloat162float(qB.y);
    }

    // Gather B columns 2t and 2t+1 (all 16 k-values each). Column n is owned by
    // group n, so its values live in lanes n*4 .. n*4+3.
    float bc0[16], bc1[16];
    #pragma unroll
    for (int tt = 0; tt < 4; ++tt) {
      const int srcA = (2 * t) * 4 + tt;
      const int srcB = (2 * t + 1) * 4 + tt;
      uint32_t uA = __shfl_sync(FULL, b0, srcA);
      uint32_t uA8 = __shfl_sync(FULL, b1, srcA);
      uint32_t uB = __shfl_sync(FULL, b0, srcB);
      uint32_t uB8 = __shfl_sync(FULL, b1, srcB);
      const __nv_bfloat162 vA = *reinterpret_cast<__nv_bfloat162*>(&uA);
      const __nv_bfloat162 vA8 = *reinterpret_cast<__nv_bfloat162*>(&uA8);
      const __nv_bfloat162 vB = *reinterpret_cast<__nv_bfloat162*>(&uB);
      const __nv_bfloat162 vB8 = *reinterpret_cast<__nv_bfloat162*>(&uB8);
      bc0[2 * tt] = __bfloat162float(vA.x); bc0[2 * tt + 1] = __bfloat162float(vA.y);
      bc0[2 * tt + 8] = __bfloat162float(vA8.x); bc0[2 * tt + 9] = __bfloat162float(vA8.y);
      bc1[2 * tt] = __bfloat162float(vB.x); bc1[2 * tt + 1] = __bfloat162float(vB.y);
      bc1[2 * tt + 8] = __bfloat162float(vB8.x); bc1[2 * tt + 9] = __bfloat162float(vB8.y);
    }

    // Exact K-reduction for the 4 D elements this thread owns.
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
      s0 = fmaf(ag[k], bc0[k], s0);
      s1 = fmaf(ag[k], bc1[k], s1);
      s2 = fmaf(ag8[k], bc0[k], s2);
      s3 = fmaf(ag8[k], bc1[k], s3);
    }
    d[0] += s0;
    d[1] += s1;
    d[2] += s2;
    d[3] += s3;
  }
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
