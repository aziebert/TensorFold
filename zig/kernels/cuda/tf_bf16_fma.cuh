#pragma once
// Portable bf16x2 FMA: native on sm_80+, software-emulated on sm_70
#include <cuda_bf16.h>
#include <stdint.h>

__device__ __forceinline__ uint32_t tf_bf16x2_fma(uint32_t a, uint32_t b, uint32_t c) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  uint32_t r;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
#else
  // Software emulation for sm_70: unpack bf16 pairs, FMA in fp32, repack
  const __nv_bfloat16* pa = reinterpret_cast<const __nv_bfloat16*>(&a);
  const __nv_bfloat16* pb = reinterpret_cast<const __nv_bfloat16*>(&b);
  const __nv_bfloat16* pc = reinterpret_cast<const __nv_bfloat16*>(&c);
  float r0 = __bfloat162float(pa[0]) * __bfloat162float(pb[0]) + __bfloat162float(pc[0]);
  float r1 = __bfloat162float(pa[1]) * __bfloat162float(pb[1]) + __bfloat162float(pc[1]);
  uint32_t r;
  __nv_bfloat16* pr = reinterpret_cast<__nv_bfloat16*>(&r);
  pr[0] = __float2bfloat16_rn(r0);
  pr[1] = __float2bfloat16_rn(r1);
  return r;
#endif
}
