// Volta (sm_70) matmuls for the 4-bit affine checkpoint: fp16 tensor cores (mma.m8n8k4, the only shape Volta has)
// with fp32 sums, for decode's few rows; one cuBLASLt GEMM over a dense fp16 copy of each weight for prompt chunks.
// Engine form: no torch, every kernel extern "C" so the Zig runtime resolves it by name from this image.
//
// Rows. A bf16 row is first scaled by a power of two so its largest value lands in [2^14, 2^15): every bf16 value
// then converts to fp16 exactly (8 significant bits fit in 11, and nothing in the row underflows), and the row's
// scale comes back in the epilogue. ``tf_vprep`` keeps rows row-major; ``tf_vprep_frag`` writes them in the decode
// kernel's fragment order [M/8][K/8][8 rows][8 inputs], so one k-step of an 8-row tile is one 128-byte line.
//
// Weights. A column's 64-input group expands to fp16(q * s + b): the nibble is placed in an fp16 mantissa, 1024 is
// subtracted, and one fma applies the group's bf16 scale and bias (exact in fp16 for their normal range). The decode
// kernel reads the ``Tiled`` layout [N/32][K/64][32 columns][8 words], a warp's 32 columns of one group as 1 KB
// contiguous; the stored MLX layout (column runs of K/8 words) works too, less coalesced.
//
// Bits. The decode kernel holds one column per thread and walks the row's K in group order with fp32 sums, the K
// split a function of the weight's shape (split_k884, in the Zig host) and its slices added in slice order, so a
// row's result never depends on how many rows share its launch. Prompt rows take one cuBLASLt algorithm per weight
// shape, pinned with split-K off, so their chain over K is the same at every row count too; it is a different chain
// from decode's.

#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace {

constexpr int COLS = 128;       // weight columns a decode block covers: 4 warps x 4 quadpairs x 8

__device__ __forceinline__ float bf2f(unsigned short v) { return __uint_as_float(static_cast<unsigned>(v) << 16); }

// ---- rows -------------------------------------------------------------------------------------------------------

// The block's largest value, to every thread (``red`` holds one float per warp).
__device__ __forceinline__ float block_max(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    v = red[0];
    for (int w = 1; w < (int)(blockDim.x >> 5); ++w) v = fmaxf(v, red[w]);
    return v;
}

// The exponent the row is scaled by: 2^-e puts its largest value in [2^14, 2^15); an empty or non-finite row keeps 1.
__device__ __forceinline__ int row_exponent(float amax) { return amax > 0.f && isfinite(amax) ? ilogbf(amax) - 14 : 0; }

// Where input k of a row goes: row-major, or the fragment order [M/8][K/8][8 rows][8 inputs].
template <bool FRAG>
__device__ __forceinline__ long row_base(int row, int K) {
    return FRAG ? (long)(row >> 3) * (K / 8) * 64 + (row & 7) * 8 : (long)row * K;
}
template <bool FRAG>
__device__ __forceinline__ long piece_at(long base, int c) { return base + (long)c * (FRAG ? 64 : 8); }   // 8 inputs

// One row a block: 512 threads, five 16-byte pieces each held in registers, so the row is read once (K <= 20480).
constexpr int PREP_THREADS = 512, PREP_PIECES = 5;
template <bool FRAG>
__device__ __forceinline__ void prep_body(const unsigned short* __restrict__ x, long ldx, int M, int K,
                                          __half* __restrict__ out, float* __restrict__ rs) {
    const int row = blockIdx.x, C = K / 8;
    const long base = row_base<FRAG>(row, K);
    if (row >= M) {                                          // FRAG's padded rows read as zeros
        for (int c = threadIdx.x; c < C; c += PREP_THREADS) *reinterpret_cast<uint4*>(out + piece_at<FRAG>(base, c)) = make_uint4(0, 0, 0, 0);
        return;
    }
    uint4 v[PREP_PIECES];
    float amax = 0.f;
#pragma unroll
    for (int j = 0; j < PREP_PIECES; ++j) {
        const int c = threadIdx.x + j * PREP_THREADS;
        v[j] = c < C ? *reinterpret_cast<const uint4*>(x + row * ldx + (long)c * 8) : make_uint4(0, 0, 0, 0);
        const unsigned w4[4] = {v[j].x, v[j].y, v[j].z, v[j].w};
#pragma unroll
        for (int h = 0; h < 4; ++h)
            amax = fmaxf(amax, fmaxf(fabsf(bf2f(w4[h] & 0xffff)), fabsf(bf2f(w4[h] >> 16))));
    }
    __shared__ float red[PREP_THREADS / 32];
    const int e = row_exponent(block_max(amax, red));
#pragma unroll
    for (int j = 0; j < PREP_PIECES; ++j) {
        const int c = threadIdx.x + j * PREP_THREADS;
        if (c >= C) continue;
        const unsigned w4[4] = {v[j].x, v[j].y, v[j].z, v[j].w};
        unsigned o4[4];
#pragma unroll
        for (int h = 0; h < 4; ++h) {
            const __half lo = __float2half_rn(ldexpf(bf2f(w4[h] & 0xffff), -e));
            const __half hi = __float2half_rn(ldexpf(bf2f(w4[h] >> 16), -e));
            o4[h] = (unsigned)__half_as_ushort(lo) | ((unsigned)__half_as_ushort(hi) << 16);
        }
        *reinterpret_cast<uint4*>(out + piece_at<FRAG>(base, c)) = make_uint4(o4[0], o4[1], o4[2], o4[3]);
    }
    if (threadIdx.x == 0) rs[row] = ldexpf(1.f, e);
}

// The same values one element at a time: rows wider than the fast kernel holds, or views it cannot load 16 bytes from.
template <bool FRAG>
__device__ __forceinline__ void prep_any_body(const unsigned short* __restrict__ x, long ldx, int M, int K,
                                              __half* __restrict__ out, float* __restrict__ rs) {
    const int row = blockIdx.x;
    const long base = row_base<FRAG>(row, K);
    auto at = [&](int k) { return FRAG ? base + (long)(k >> 3) * 64 + (k & 7) : base + k; };
    if (row >= M) {
        for (int k = threadIdx.x; k < K; k += blockDim.x) out[at(k)] = __float2half_rn(0.f);
        return;
    }
    const unsigned short* xr = x + row * ldx;
    float amax = 0.f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) amax = fmaxf(amax, fabsf(bf2f(xr[k])));
    __shared__ float red[32];
    const int e = row_exponent(block_max(amax, red));
    for (int k = threadIdx.x; k < K; k += blockDim.x) out[at(k)] = __float2half_rn(ldexpf(bf2f(xr[k]), -e));
    if (threadIdx.x == 0) rs[row] = ldexpf(1.f, e);
}

// ---- weights ----------------------------------------------------------------------------------------------------

// Eight nibbles of one word -> eight fp16 values q * s + b in input order, two per instruction.
__device__ __forceinline__ int4 dequant8(unsigned w, __half2 s2, __half2 b2) {
    const unsigned magic = 0x64006400u;                    // fp16 1024.0 in each half
    const __half2 k1024 = __halves2half2(__ushort_as_half(0x6400), __ushort_as_half(0x6400));
    __half2 v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {                           // v[j] = (q_j, q_{j+4})
        unsigned u = ((w >> (4 * j)) & 0x000F000Fu) | magic;
        v[j] = __hsub2(*reinterpret_cast<__half2*>(&u), k1024);
    }
    __half2 o[4] = {__lows2half2(v[0], v[1]), __lows2half2(v[2], v[3]),
                    __highs2half2(v[0], v[1]), __highs2half2(v[2], v[3])};
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __hfma2(o[j], s2, b2);
    return *reinterpret_cast<int4*>(o);
}

// A group's bf16 scale and bias as fp16 pairs (bf16 -> fp32 -> fp16: exact for their normal range).
__device__ __forceinline__ void group_factors(unsigned short s, unsigned short b, __half2& s2, __half2& b2) {
    const __half sh = __float2half_rn(bf2f(s)), bh = __float2half_rn(bf2f(b));
    s2 = __halves2half2(sh, sh);
    b2 = __halves2half2(bh, bh);
}

// ---- decode: mma.m8n8k4 from registers, one weight column per thread ---------------------------------------------
//
// Fragment layout, measured on a V100 (closed form, 0 mismatches): quadpair q = (lane >> 2) & 3,
// idx = lane % 4 + 4 * (lane >= 16). A .row: the thread holds A[idx][0..3]. B .col: B[0..3][idx].
// D f32: d[i] sits at row (lane & 1) + 2 * ((i >> 1) & 1) + 4 * (lane >= 16), col (i & 1) + 2 * ((lane >> 1) & 1) + 4 * (i >> 2).
// The four quadpairs share A (the same 8 rows) and each owns 8 weight columns, so a warp covers 32 columns.

__device__ __forceinline__ void mma884(float (&d)[8], unsigned a0, unsigned a1, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
                 "{%0,%1,%2,%3,%4,%5,%6,%7};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

// MT 8-row tiles a block (1, 2 or 4); TILED: the ``Tiled`` layout, else the stored one; FRAG: rows from ``tf_vprep_frag``.
template <int MT, bool TILED, bool FRAG>
__device__ __forceinline__ void qmm884_body(
        const __half* __restrict__ X, const float* __restrict__ RS, const int* __restrict__ W,
        const unsigned short* __restrict__ S, const unsigned short* __restrict__ B, void* __restrict__ OUT,
        float* __restrict__ PART, int M, int N, int K, int gper, int sk, bool f32) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int q = (lane >> 2) & 3, idx = (lane & 3) + 4 * (lane >= 16);
    const int s = blockIdx.y, m0 = blockIdx.z * 8 * MT;
    const int g0 = s * gper, g1 = g0 + gper;
    const int KG = K / 64, K8 = K / 8;
    const long wstride = TILED ? 32 * 8 : 8;                    // words between consecutive groups of one column
    const long sstride = TILED ? 32 : 1;
    const int cbase = blockIdx.x * COLS + warp * 32 + q * 8;
    const int col = cbase + idx;
    const bool nok = col < N;
    const int c32 = nok ? col : 0;
    const int4* wp = reinterpret_cast<const int4*>(TILED ? W + ((long)(c32 >> 5) * KG * 32 + (c32 & 31)) * 8 : W + (long)c32 * K8);
    const unsigned short* sp = TILED ? S + (long)(c32 >> 5) * KG * 32 + (c32 & 31) : S + (long)c32 * KG;
    const unsigned short* bp = TILED ? B + (long)(c32 >> 5) * KG * 32 + (c32 & 31) : B + (long)c32 * KG;

    // FRAG: a warp's eight row loads of one k-step are one 128-byte line (rows past M are zeros); row-major: one
    // row's eight k-steps share a line (rows past M are skipped).
    const __half* xr[MT];
    bool rok[MT];
#pragma unroll
    for (int t = 0; t < MT; ++t) {
        if (FRAG) {
            rok[t] = m0 + t * 8 < M;
            xr[t] = X + ((long)(rok[t] ? (m0 >> 3) + t : 0) * K8) * 64 + idx * 8;
        } else {
            const int r = m0 + t * 8 + idx;
            rok[t] = r < M;
            xr[t] = X + (long)(rok[t] ? r : 0) * K;
        }
    }
    float acc[MT][8];
#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[t][i] = 0.f;

    // the column's current group in registers, the next one's loads in flight
    int4 c0 = make_int4(0, 0, 0, 0), c1 = c0;
    unsigned short cs = 0, cb = 0;
    if (nok) {
        c0 = __ldcs(wp + g0 * wstride / 4); c1 = __ldcs(wp + g0 * wstride / 4 + 1);
        cs = __ldg(sp + g0 * sstride); cb = __ldg(bp + g0 * sstride);
    }
    for (int g = g0; g < g1; ++g) {
        int4 n0 = make_int4(0, 0, 0, 0), n1 = n0;
        unsigned short ns = 0, nb = 0;
        if (nok && g + 1 < g1) {
            n0 = __ldcs(wp + (g + 1) * wstride / 4); n1 = __ldcs(wp + (g + 1) * wstride / 4 + 1);
            ns = __ldg(sp + (g + 1) * sstride); nb = __ldg(bp + (g + 1) * sstride);
        }
        __half2 s2, b2;
        group_factors(cs, cb, s2, b2);
        const unsigned words[8] = {(unsigned)c0.x, (unsigned)c0.y, (unsigned)c0.z, (unsigned)c0.w,
                                   (unsigned)c1.x, (unsigned)c1.y, (unsigned)c1.z, (unsigned)c1.w};
#pragma unroll
        for (int wi = 0; wi < 8; ++wi) {
            const int4 o = dequant8(words[wi], s2, b2);         // (k0,k1) (k2,k3) (k4,k5) (k6,k7)
            const int k = g * 64 + wi * 8;
#pragma unroll
            for (int t = 0; t < MT; ++t) {
                const uint4 a = rok[t] ? __ldg(reinterpret_cast<const uint4*>(xr[t] + (FRAG ? (long)(k >> 3) * 64 : (long)k)))
                                       : make_uint4(0, 0, 0, 0);
                mma884(acc[t], a.x, a.y, (unsigned)o.x, (unsigned)o.y);
                mma884(acc[t], a.z, a.w, (unsigned)o.z, (unsigned)o.w);
            }
        }
        c0 = n0; c1 = n1; cs = ns; cb = nb;
    }

#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int m = m0 + t * 8 + (lane & 1) + 2 * ((i >> 1) & 1) + 4 * (lane >= 16);
            const int n = cbase + (i & 1) + 2 * ((lane >> 1) & 1) + 4 * (i >> 2);
            if (m >= M || n >= N) continue;
            const float v = acc[t][i] * RS[m];
            if (sk > 1) PART[((long)s * M + m) * N + n] = v;
            else if (f32) static_cast<float*>(OUT)[(long)m * N + n] = v;
            else static_cast<__nv_bfloat16*>(OUT)[(long)m * N + n] = __float2bfloat16_rn(v);
        }
}

// Split-K slices summed in slice order (the same order whatever the row count).
__device__ __forceinline__ void reduce_body(const float* __restrict__ part, void* __restrict__ out, long total, int sk, bool f32) {
    const long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= total) return;
    float a = part[i];
    for (int s = 1; s < sk; ++s) a += part[s * total + i];
    if (f32) static_cast<float*>(out)[i] = a;
    else static_cast<__nv_bfloat16*>(out)[i] = __float2bfloat16_rn(a);
}

// ---- prompts: a dense fp16 copy of the weight, then cuBLASLt --------------------------------------------------------

// A Tiled weight -> dense fp16 (N, K), each value the fp16(q * s + b) the decode kernel expands. One thread a word.
__device__ __forceinline__ void dequant_body(const unsigned* __restrict__ w, const unsigned short* __restrict__ s,
                                             const unsigned short* __restrict__ b, __half* __restrict__ out, int N, int K) {
    const int kw = K / 8;                                                // words a row
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;        // n * kw + word
    if (idx >= (long)N * kw) return;
    const long n = idx / kw;
    const int word = idx % kw, g = word >> 3, j = word & 7, kg = K / 64;
    const long cell = ((n >> 5) * kg + g) * 32 + (n & 31);
    __half2 s2, b2;
    group_factors(s[cell], b[cell], s2, b2);
    reinterpret_cast<int4*>(out)[idx] = dequant8(w[cell * 8 + j], s2, b2);
}

// y (M, N) fp32 sums of the scaled rows -> y * rs[row], as bf16 or as fp32 in place.
__device__ __forceinline__ void unscale_body(const float* __restrict__ y, const float* __restrict__ rs, void* __restrict__ out,
                                             long N, long total4, bool f32) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total4) return;
    const float r = rs[(i * 4) / N];
    float4 v = reinterpret_cast<const float4*>(y)[i];
    v.x *= r; v.y *= r; v.z *= r; v.w *= r;
    if (f32) { reinterpret_cast<float4*>(out)[i] = v; return; }
    __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y), hi = __floats2bfloat162_rn(v.z, v.w);
    uint2 u = {*reinterpret_cast<unsigned*>(&lo), *reinterpret_cast<unsigned*>(&hi)};
    reinterpret_cast<uint2*>(out)[i] = u;
}

}  // namespace

// ---- the image's symbols, one per (mt, tiled, frag) for the decode kernel ----------------------------------------

// X: the rows from tf_vprep (M, K) or tf_vprep_frag (M/8, K/8, 8, 8); W: Tiled words (tiled) or the stored (N, K/8)
// words; OUT: (M, N) fp32 (f32) or bf16; PART: (sk, M, N) fp32 when sk > 1; grid ((N+127)/128, sk, (M+8*mt-1)/(8*mt)),
// block 128.
extern "C" __global__ void tf_vqmm884_1_0_0(const __half* __restrict__ X, const float* __restrict__ RS, const int* __restrict__ W, const unsigned short* __restrict__ S, const unsigned short* __restrict__ B, void* __restrict__ OUT, float* __restrict__ PART, int M, int N, int K, int gper, int sk, int f32) { qmm884_body<1, false, false>(X, RS, W, S, B, OUT, PART, M, N, K, gper, sk, f32 != 0); }
extern "C" __global__ void tf_vqmm884_1_0_1(const __half* __restrict__ X, const float* __restrict__ RS, const int* __restrict__ W, const unsigned short* __restrict__ S, const unsigned short* __restrict__ B, void* __restrict__ OUT, float* __restrict__ PART, int M, int N, int K, int gper, int sk, int f32) { qmm884_body<1, false, true>(X, RS, W, S, B, OUT, PART, M, N, K, gper, sk, f32 != 0); }
extern "C" __global__ void tf_vqmm884_1_1_0(const __half* __restrict__ X, const float* __restrict__ RS, const int* __restrict__ W, const unsigned short* __restrict__ S, const unsigned short* __restrict__ B, void* __restrict__ OUT, float* __restrict__ PART, int M, int N, int K, int gper, int sk, int f32) { qmm884_body<1, true, false>(X, RS, W, S, B, OUT, PART, M, N, K, gper, sk, f32 != 0); }
extern "C" __global__ void tf_vqmm884_1_1_1(const __half* __restrict__ X, const float* __restrict__ RS, const int* __restrict__ W, const unsigned short* __restrict__ S, const unsigned short* __restrict__ B, void* __restrict__ OUT, float* __restrict__ PART, int M, int N, int K, int gper, int sk, int f32) { qmm884_body<1, true, true>(X, RS, W, S, B, OUT, PART, M, N, K, gper, sk, f32 != 0); }
extern "C" __global__ void tf_vqmm884_2_0_0(const __half* __restrict__ X, const float* __restrict__ RS, const int* __restrict__ W, const unsigned short* __restrict__ S, const unsigned short* __restrict__ B, void* __restrict__ OUT, float* __restrict__ PART, int M, int N, int K, int gper, int sk, int f32) { qmm884_body<2, false, false>(X, RS, W, S, B, OUT, PART, M, N, K, gper, sk, f32 != 0); }
extern "C" __global__ void tf_vqmm884_2_0_1(const __half* __restrict__ X, const float* __restrict__ RS, const int* __restrict__ W, const unsigned short* __restrict__ S, const unsigned short* __restrict__ B, void* __restrict__ OUT, float* __restrict__ PART, int M, int N, int K, int gper, int sk, int f32) { qmm884_body<2, false, true>(X, RS, W, S, B, OUT, PART, M, N, K, gper, sk, f32 != 0); }
extern "C" __global__ void tf_vqmm884_2_1_0(const __half* __restrict__ X, const float* __restrict__ RS, const int* __restrict__ W, const unsigned short* __restrict__ S, const unsigned short* __restrict__ B, void* __restrict__ OUT, float* __restrict__ PART, int M, int N, int K, int gper, int sk, int f32) { qmm884_body<2, true, false>(X, RS, W, S, B, OUT, PART, M, N, K, gper, sk, f32 != 0); }
extern "C" __global__ void tf_vqmm884_2_1_1(const __half* __restrict__ X, const float* __restrict__ RS, const int* __restrict__ W, const unsigned short* __restrict__ S, const unsigned short* __restrict__ B, void* __restrict__ OUT, float* __restrict__ PART, int M, int N, int K, int gper, int sk, int f32) { qmm884_body<2, true, true>(X, RS, W, S, B, OUT, PART, M, N, K, gper, sk, f32 != 0); }
extern "C" __global__ void tf_vqmm884_4_0_0(const __half* __restrict__ X, const float* __restrict__ RS, const int* __restrict__ W, const unsigned short* __restrict__ S, const unsigned short* __restrict__ B, void* __restrict__ OUT, float* __restrict__ PART, int M, int N, int K, int gper, int sk, int f32) { qmm884_body<4, false, false>(X, RS, W, S, B, OUT, PART, M, N, K, gper, sk, f32 != 0); }
extern "C" __global__ void tf_vqmm884_4_0_1(const __half* __restrict__ X, const float* __restrict__ RS, const int* __restrict__ W, const unsigned short* __restrict__ S, const unsigned short* __restrict__ B, void* __restrict__ OUT, float* __restrict__ PART, int M, int N, int K, int gper, int sk, int f32) { qmm884_body<4, false, true>(X, RS, W, S, B, OUT, PART, M, N, K, gper, sk, f32 != 0); }
extern "C" __global__ void tf_vqmm884_4_1_0(const __half* __restrict__ X, const float* __restrict__ RS, const int* __restrict__ W, const unsigned short* __restrict__ S, const unsigned short* __restrict__ B, void* __restrict__ OUT, float* __restrict__ PART, int M, int N, int K, int gper, int sk, int f32) { qmm884_body<4, true, false>(X, RS, W, S, B, OUT, PART, M, N, K, gper, sk, f32 != 0); }
extern "C" __global__ void tf_vqmm884_4_1_1(const __half* __restrict__ X, const float* __restrict__ RS, const int* __restrict__ W, const unsigned short* __restrict__ S, const unsigned short* __restrict__ B, void* __restrict__ OUT, float* __restrict__ PART, int M, int N, int K, int gper, int sk, int f32) { qmm884_body<4, true, true>(X, RS, W, S, B, OUT, PART, M, N, K, gper, sk, f32 != 0); }

// (M, K) bf16 rows -> exact fp16 copies scaled by a power of two, and the row scales: row-major, one 16-byte piece
// a thread (K <= 20480).
extern "C" __global__ void __launch_bounds__(PREP_THREADS) tf_vprep(const unsigned short* __restrict__ x, long ldx, int M, int K, __half* __restrict__ out, float* __restrict__ rs) { prep_body<false>(x, ldx, M, K, out, rs); }
extern "C" __global__ void __launch_bounds__(PREP_THREADS) tf_vprep_frag(const unsigned short* __restrict__ x, long ldx, int M, int K, __half* __restrict__ out, float* __restrict__ rs) { prep_body<true>(x, ldx, M, K, out, rs); }
extern "C" __global__ void tf_vprep_any(const unsigned short* __restrict__ x, long ldx, int M, int K, __half* __restrict__ out, float* __restrict__ rs) { prep_any_body<false>(x, ldx, M, K, out, rs); }
extern "C" __global__ void tf_vprep_frag_any(const unsigned short* __restrict__ x, long ldx, int M, int K, __half* __restrict__ out, float* __restrict__ rs) { prep_any_body<true>(x, ldx, M, K, out, rs); }

// Split-K partials (sk, M, N) -> the (M, N) sums, slice order; OUT fp32 (f32) or bf16. Block 256, grid (total+255)/256.
extern "C" __global__ void tf_vreduce(const float* __restrict__ part, void* __restrict__ out, long total, int sk, int f32) { reduce_body(part, out, total, sk, f32 != 0); }

// Tiled words (N/32, K/64, 32, 8) and bf16 scales/biases -> dense fp16 (N, K). Block 256.
extern "C" __global__ void tf_vdequant(const unsigned* __restrict__ w, const unsigned short* __restrict__ s, const unsigned short* __restrict__ b, __half* __restrict__ out, int N, int K) { dequant_body(w, s, b, out, N, K); }

// (M, N) fp32 sums of the scaled rows * rs[row] -> OUT fp32 (f32) or bf16 in place. Block 256, N a multiple of 4.
extern "C" __global__ void tf_vunscale(const float* __restrict__ y, const float* __restrict__ rs, void* __restrict__ out, long N, long total4, int f32) { unscale_body(y, rs, out, N, total4, f32 != 0); }
