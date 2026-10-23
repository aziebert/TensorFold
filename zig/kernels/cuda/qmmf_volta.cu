// sm_70 NVFP4 / FP8 / 16-bit matmuls on qmm_volta's mma.m8n8k4 skeleton: exact fp16 weights, one fp32 factor a column.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int THREADS = 128;    // 4 warps

constexpr int COLS = 128;       // weight columns a block: 4 warps x 4 quadpairs x 8

enum Fmt { FP4 = 0, FP8 = 1, F16 = 2 };

template <int FMT> struct Words { static constexpr int n = FMT == FP4 ? 8 : FMT == FP8 ? 16 : 32; };

__device__ __forceinline__ void mma884(float (&d)[8], unsigned a0, unsigned a1, unsigned b0, unsigned b1) {
    asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
                 "{%0,%1,%2,%3,%4,%5,%6,%7};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

// One e4m3 byte -> the exact fp16 value, in both halves.
__device__ __forceinline__ __half2 e4m3_h2(unsigned b) {
    const unsigned short hb = (unsigned short)(((b & 0x80u) << 8) | ((b & 0x7Fu) << 7));   // value * 2^-8, exact
    const __half h = __hmul(__ushort_as_half(hb), __ushort_as_half(0x5C00));               // * 256, exact
    return __halves2half2(h, h);
}

// Eight e2m1 codes of one word (code j at nibble j) -> fp16 code * 2^-14 * scale, inputs in order, two a half2.
__device__ __forceinline__ int4 fp4x8(unsigned q, __half2 s2) {
    constexpr unsigned S = 0x80008000u, EM = 0x0E000E00u;
    unsigned v0 = ((q << 12) & S) | ((q << 9) & EM);       // (code 0, code 4)
    unsigned v1 = ((q << 8) & S) | ((q << 5) & EM);        // (1, 5)
    unsigned v2 = ((q << 4) & S) | ((q << 1) & EM);        // (2, 6)
    unsigned v3 = (q & S) | ((q >> 3) & EM);               // (3, 7)
    const __half2 h0 = *reinterpret_cast<__half2*>(&v0), h1 = *reinterpret_cast<__half2*>(&v1);
    const __half2 h2 = *reinterpret_cast<__half2*>(&v2), h3 = *reinterpret_cast<__half2*>(&v3);
    __half2 o[4] = {__hmul2(__lows2half2(h0, h1), s2), __hmul2(__lows2half2(h2, h3), s2),
                    __hmul2(__highs2half2(h0, h1), s2), __hmul2(__highs2half2(h2, h3), s2)};
    return *reinterpret_cast<int4*>(o);
}

// Two words of e4m3 bytes (input order) -> eight fp16 values * 2^-8, exact.
__device__ __forceinline__ unsigned e4m3_pair(unsigned w) {            // bytes 0, 1 of w -> (lo half, hi half)
    return ((w << 8) & 0x8000u) | ((w << 7) & 0x3F80u) | ((w << 16) & 0x80000000u) | ((w << 15) & 0x3F800000u);
}
__device__ __forceinline__ int4 fp8x8(unsigned a, unsigned b) {
    return make_int4((int)e4m3_pair(a), (int)e4m3_pair(a >> 16), (int)e4m3_pair(b), (int)e4m3_pair(b >> 16));
}

// One 64-input group of one column in registers.
template <int FMT> struct Group {
    int4 w[Words<FMT>::n / 4];
    unsigned s;
};

template <int FMT>
__device__ __forceinline__ void load_group(Group<FMT>& gr, const int4* wp, const unsigned* sp, bool ok) {
#pragma unroll
    for (int i = 0; i < Words<FMT>::n / 4; ++i) gr.w[i] = ok ? __ldcs(wp + i) : make_int4(0, 0, 0, 0);
    if (FMT == FP4) gr.s = ok ? __ldg(sp) : 0u;
}

// Inputs [8 wi, 8 wi + 8) of the group as four fp16 pairs.
template <int FMT>
__device__ __forceinline__ int4 expand(const Group<FMT>& gr, int wi, const __half2 (&s2)[4]) {
    const unsigned* u = reinterpret_cast<const unsigned*>(gr.w);
    if (FMT == FP4) return fp4x8(u[wi], s2[wi >> 1]);
    if (FMT == FP8) return fp8x8(u[2 * wi], u[2 * wi + 1]);
    return gr.w[wi];
}

// Rows staged AHEAD groups ahead in shared memory; with ``cnt`` the last K slice adds every slice in slice order.
constexpr int AHEAD = 3, STAGES = AHEAD + 1;
template <int FMT, int MT, int NACC>
__device__ __forceinline__ void
    qmmf884s_body(
        const __half* __restrict__ X, const float* __restrict__ RS, const int* __restrict__ W,
        const unsigned* __restrict__ S, const float* __restrict__ ALPHA, void* __restrict__ OUT,
        float* __restrict__ PART, int* __restrict__ CNT, int M, int N, int K, int gper, int sk, bool f32) {
    constexpr int WPG = Words<FMT>::n;
    __shared__ __align__(16) __half sa[STAGES][MT][512];         // [stage][tile][8 k-steps x 8 rows x 8 inputs]
    __shared__ int last;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int q = (lane >> 2) & 3, idx = (lane & 3) + 4 * (lane >= 16);
    const int s = blockIdx.y, m0 = blockIdx.z * 8 * MT;
    const int g0 = s * gper, g1 = g0 + gper;
    const int KG = K / 64, K8 = K / 8;
    const int cbase = blockIdx.x * COLS + warp * 32 + q * 8;
    const int col = cbase + idx;
    const bool nok = col < N;
    const int c32 = nok ? col : 0;
    // nvfp4.packWords layout: 64-column tiles x KG groups, 512 words a tile-group: column n, group g,
    // slot s (= 2*w+v, w in 0..3, v in 0..1) at word ((n>>6)*KG + g)*512 + (n&63)*8 + s; slot nibble p is
    // input g*64 + 32*v + 2*w + offs[p], offs = {0,8,16,24,1,9,17,25}. Four e4m3 scale bytes at
    // ((n>>6)*KG + g)*64 + (n&63), byte q scaling inputs 16q..16q+15. The bases below start at group 0;
    // the loads add g * WSTEP / SPSTEP, never g0 twice.
    const long woff = (FMT == FP4) ? ((long)(c32 >> 6) * KG * 512 + (c32 & 63) * 8)
                                   : (long)(c32 >> 5) * KG * 32 + (c32 & 31) * WPG;   // legacy layout, untested fmts
    const int4* wp = reinterpret_cast<const int4*>(W + woff);
    const unsigned* sp = S + (FMT == FP4 ? (long)(c32 >> 6) * KG * 64 + (c32 & 63) : 0);
    const long WSTEP = (FMT == FP4) ? 128 : 32L * WPG / 4;                       // int4s a group stride
    int tiles = 0;                                                  // the block's row tiles that hold rows
#pragma unroll
    for (int t = 0; t < MT; ++t) tiles += m0 + t * 8 < M;

    // staging: tile t of group g is 1 KB at X[(m0/8 + t)][g*8 .. g*8+8][8][8]; 64 threads a tile, one int4 each
    constexpr int PER = (MT * 64 + THREADS - 1) / THREADS;          // int4s a thread stages a group
    auto fetch = [&](int g, int4 (&r)[PER]) {
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int e = threadIdx.x + j * THREADS, t = e >> 6, c = e & 63;
            r[j] = (e < MT * 64 && t < tiles && g < g1)
                ? __ldg(reinterpret_cast<const int4*>(X + ((long)((m0 >> 3) + t) * K8 + g * 8) * 64) + c)
                : make_int4(0, 0, 0, 0);
        }
    };
    auto store = [&](int g, const int4 (&r)[PER]) {
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int e = threadIdx.x + j * THREADS, t = e >> 6, c = e & 63;
            if (e < MT * 64) reinterpret_cast<int4*>(sa[g % STAGES][t])[c] = r[j];
        }
    };
    {
        int4 r[PER];
#pragma unroll
        for (int d = 0; d < AHEAD; ++d) { fetch(g0 + d, r); store(d, r); }
    }
    __syncthreads();

    float acc[MT][NACC][8];
#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
        for (int a = 0; a < NACC; ++a)
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[t][a][i] = 0.f;

    Group<FMT> cur, nxt;                                            // the weights stream one group ahead
    const long SPSTEP = (FMT == FP4) ? 64 : 32;                     // scale words a group stride
    load_group<FMT>(cur, wp + g0 * WSTEP, sp + (long)g0 * SPSTEP, nok);
    for (int g = g0; g < g1; ++g) {
        const int i = g - g0;
        load_group<FMT>(nxt, wp + (g + 1) * WSTEP, sp + (long)(g + 1) * SPSTEP, nok && g + 1 < g1);
        int4 ahead[PER];
        fetch(g + AHEAD, ahead);
        __half2 s2[4];
        if (FMT == FP4) {
#pragma unroll
            for (int j = 0; j < 4; ++j) s2[j] = e4m3_h2((cur.s >> (8 * j)) & 0xFFu);
        }
        const __half* stage = &sa[i % STAGES][0][0];
        if (FMT == FP4) {
            // packWords slots: slot s = 2*w+v holds inputs g*64 + 32*v + 2*w + {0,1,8,9,16,17,24,25} in
            // nibble order {0,8,16,24,1,9,17,25} (offs above). Two mma.m8n8k4 per slot: call A on the k
            // offsets {2w, 2w+1} of k-blocks 4v and 4v+1, call B on the same offsets of blocks 4v+2/4v+3.
            // Staged-tile int4 c = (4v+q)*8 + idx holds row idx's inputs 8*(4v+q)..+7, so a k-offset 2w
            // pair is uint w of that int4 (uint index (4v+q)*32 + idx*4 + w in the tile's 256 uints).
            const unsigned* sw = reinterpret_cast<const unsigned*>(cur.w);
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                const int v = s & 1, w = s >> 1;
                const int4 ra = fp4x8(sw[s], s2[2 * v]);              // (i0,i1)(i2,i3)(i4,i5)(i6,i7) x q(2v)
                const int4 rb = fp4x8(sw[s], s2[2 * v + 1]);
                const unsigned b0a = (ra.x & 0xFFFFu) | ((ra.z & 0xFFFFu) << 16);            // (i0,i4)
                const unsigned b1a = ((ra.x >> 16) & 0xFFFFu) | (((ra.z >> 16) & 0xFFFFu) << 16);
                const unsigned b0b = (rb.y & 0xFFFFu) | ((rb.w & 0xFFFFu) << 16);            // (i2,i6)
                const unsigned b1b = ((rb.y >> 16) & 0xFFFFu) | (((rb.w >> 16) & 0xFFFFu) << 16);
#pragma unroll
                for (int t = 0; t < MT; ++t) {
                    if (t >= tiles) continue;
                    const unsigned* a4 = reinterpret_cast<const unsigned*>(stage + t * 512);
                    const unsigned u0 = a4[(4 * v + 0) * 32 + idx * 4 + w];
                    const unsigned u1 = a4[(4 * v + 1) * 32 + idx * 4 + w];
                    const unsigned u2 = a4[(4 * v + 2) * 32 + idx * 4 + w];
                    const unsigned u3 = a4[(4 * v + 3) * 32 + idx * 4 + w];
                    mma884(acc[t][(2 * s) % NACC], u0, u1, b0a, b1a);
                    mma884(acc[t][(2 * s + 1) % NACC], u2, u3, b0b, b1b);
                }
            }
        } else {
#pragma unroll
            for (int wi = 0; wi < 8; ++wi) {
                const int4 o = expand<FMT>(cur, wi, s2);
#pragma unroll
                for (int t = 0; t < MT; ++t) {
                    if (t >= tiles) continue;
                    const uint4 a = *reinterpret_cast<const uint4*>(stage + t * 512 + wi * 64 + idx * 8);
                    mma884(acc[t][(2 * wi) % NACC], a.x, a.y, (unsigned)o.x, (unsigned)o.y);
                    mma884(acc[t][(2 * wi + 1) % NACC], a.z, a.w, (unsigned)o.z, (unsigned)o.w);
                }
            }
        }
        cur = nxt;
        store(i + AHEAD, ahead);
        __syncthreads();
    }
#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
        for (int a = 1; a < NACC; ++a)
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[t][0][i] += acc[t][a][i];

#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int m = m0 + t * 8 + (lane & 1) + 2 * ((i >> 1) & 1) + 4 * (lane >= 16);
            const int n = cbase + (i & 1) + 2 * ((lane >> 1) & 1) + 4 * (i >> 2);
            if (m >= M || n >= N) continue;
            const float v = acc[t][0][i] * RS[m] * ALPHA[n];
            if (sk > 1) PART[((long)s * M + m) * N + n] = v;
            else if (f32) static_cast<float*>(OUT)[(long)m * N + n] = v;
            else static_cast<__nv_bfloat16*>(OUT)[(long)m * N + n] = __float2bfloat16_rn(v);
        }
    if (sk == 1 || CNT == nullptr) return;
    __threadfence();
    __syncthreads();
    const int tile_id = blockIdx.z * gridDim.x + blockIdx.x;
    if (threadIdx.x == 0) last = atomicAdd(CNT + tile_id, 1) == sk - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    for (int e = threadIdx.x; e < 8 * MT * COLS; e += THREADS) {
        const int m = m0 + e / COLS, n = blockIdx.x * COLS + e % COLS;
        if (m >= M || n >= N) continue;
        const long at = (long)m * N + n, total = (long)M * N;
        float v[16];                                                // every slice's load in flight, then added in order
#pragma unroll
        for (int z = 0; z < 16; ++z) v[z] = z < sk ? __ldcg(PART + z * total + at) : 0.f;
        float a = v[0];
#pragma unroll
        for (int z = 1; z < 16; ++z) if (z < sk) a += v[z];
        if (f32) static_cast<float*>(OUT)[at] = a;
        else static_cast<__nv_bfloat16*>(OUT)[at] = __float2bfloat16_rn(a);
    }
    if (threadIdx.x == 0) CNT[tile_id] = 0;                         // ready for the next launch
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

// From an fp4x8 result, (nib[ne], nib[ne + 4]) packed as a half2 (ne in 0..3) — the even/odd output pair
// taken from one slot word.
__device__ __forceinline__ unsigned slotpair(const int4& r, int ne) {
    const unsigned pe = (ne < 2) ? r.x : r.y;   // holds nibs ne, ne + 1
    const unsigned po = (ne < 2) ? r.z : r.w;   // holds nibs ne + 4, ne + 5
    const __half2 he = *reinterpret_cast<const __half2*>(&pe);
    const __half2 ho = *reinterpret_cast<const __half2*>(&po);
    const __half a = (ne & 1) ? __high2half(he) : __low2half(he);
    const __half b = (ne & 1) ? __high2half(ho) : __low2half(ho);
    const __half2 o = __halves2half2(a, b);
    return *reinterpret_cast<const unsigned*>(&o);
}

// A Tiled weight -> dense fp16 (N, K): the values the decode kernel expands. One thread eight inputs.
template <int FMT>
__device__ __forceinline__ void dequant_body(const int* __restrict__ W, const unsigned* __restrict__ S, __half* __restrict__ out,
                               int N, int K) {
    constexpr int WPG = Words<FMT>::n;
    const int k8 = K / 8, kg = K / 64;
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;      // n * k8 + piece
    if (idx >= (long)N * k8) return;
    const long n = idx / k8;
    const int piece = idx % k8, g = piece >> 3, wi = piece & 7;
    int4 o;
    if (FMT == FP4) {
        // packWords: output p of int4 wi lives in slot (p >> 1) * 2 + ((wi >> 2) & 1) at nibble
        // (wi & 3) + 4 * (p & 1), scaled by byte (wi >> 1) of the column's e4m3x4 word.
        const unsigned* u = reinterpret_cast<const unsigned*>(W) + ((long)(n >> 6) * kg + g) * 512 + (n & 63) * 8;
        const __half2 s2 = e4m3_h2((S[((n >> 6) * kg + g) * 64 + (n & 63)] >> (8 * (wi >> 1))) & 0xFFu);
        const int b = (wi >> 2) & 1, ne = wi & 3;
        o = make_int4(slotpair(fp4x8(u[b], s2), ne), slotpair(fp4x8(u[2 + b], s2), ne),
                      slotpair(fp4x8(u[4 + b], s2), ne), slotpair(fp4x8(u[6 + b], s2), ne));
    } else {
        const long cell = ((n >> 5) * kg + g) * 32 + (n & 31);
        const unsigned* u = reinterpret_cast<const unsigned*>(W) + cell * WPG;
        if (FMT == FP8) o = fp8x8(u[2 * wi], u[2 * wi + 1]);
        else o = reinterpret_cast<const int4*>(u)[wi];
    }
    reinterpret_cast<int4*>(out)[idx] = o;
}

// fp32 sums (M, N) of scaled rows -> y * rs[row] * alpha[col], bf16 or fp32 in place (the decode epilogue's order).
__device__ __forceinline__ void unscale2_body(const float* __restrict__ y, const float* __restrict__ rs,
                                const float* __restrict__ alpha, void* __restrict__ out, long N, long total, bool f32) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const float v = y[i] * rs[i / N] * alpha[i % N];
    if (f32) static_cast<float*>(out)[i] = v;
    else static_cast<__nv_bfloat16*>(out)[i] = __float2bfloat16_rn(v);
}

// Rows of a page-locked host table read over PCIe through its mapped device address, copied exactly.
__device__ __forceinline__ void gather_rows_body(const long* __restrict__ ids, const uint4* __restrict__ table, uint4* __restrict__ out,
                               int row16) {
    const long id = ids[blockIdx.x];
    for (int c = threadIdx.x; c < row16; c += blockDim.x) out[(long)blockIdx.x * row16 + c] = table[id * row16 + c];
}

// Plain row-major X [M][K] -> the 8xK8-block layout the qmmf884s matmul reads (an 8-row x K8 int4 transpose
// per 8-row tile): int4 i = (row_tile*K8 + kcol8)*8 + idx holds X[(row_tile*8+idx)][kcol8*8 .. +7].
// One thread one int4 (8 half). The buffer is the full tiled shape, ceil(M/8)*K/8 int4: rows past M in
// the last tile are zeroed, because the matmul fetches whole 8-row tiles regardless of M.
__device__ __forceinline__ void tile_x_body(const __half* __restrict__ X, __half* __restrict__ Xt, int M, int K) {
    const long K8 = K / 8;
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)((M + 7) / 8) * K8 * 8) return;
    const int idx = (int)(i & 7);                 // row within the 8-row tile
    const long base = i >> 3;                     // row_tile*K8 + kcol8
    const long kcol8 = base % K8;
    const long row = (base / K8) * 8 + idx;
    reinterpret_cast<int4*>(Xt)[i] = (row < M) ? reinterpret_cast<const int4*>(X)[row * K8 + kcol8] : make_int4(0, 0, 0, 0);
}

}  // namespace

// Engine entry points: the sm_70 image is a fatbin, not a static lib — the Zig runtime loads it with cuModuleLoad
// and resolves these by name. The matmul suffix is (fmt, mt): fmt 0/1/2 = NVFP4/FP8/16-bit, mt the row tiles;
// grid (ceil(N/COLS), sk, ceil(M/(8*mt))), block THREADS.
#define TF_VQ(FMT_, MT_) \
extern "C" __global__ void __launch_bounds__(THREADS) tf_vqmmf884s_##FMT_##_##MT_(const __half* __restrict__ X, const float* __restrict__ RS, const int* __restrict__ W, const unsigned* __restrict__ S, const float* __restrict__ ALPHA, void* __restrict__ OUT, float* __restrict__ PART, int* __restrict__ CNT, int M, int N, int K, int gper, int sk, bool f32) { qmmf884s_body<FMT_, MT_, 2>(X, RS, W, S, ALPHA, OUT, PART, CNT, M, N, K, gper, sk, f32); }
TF_VQ(0, 1)
TF_VQ(0, 2)
TF_VQ(0, 4)
TF_VQ(1, 1)
TF_VQ(1, 2)
TF_VQ(1, 4)
TF_VQ(2, 1)
TF_VQ(2, 2)
TF_VQ(2, 4)
#undef TF_VQ

// Split-K slices summed in slice order (grid ceil(total/256)), block 256.
extern "C" __global__ void tf_vqmmf884s_reduce(const float* __restrict__ part, void* __restrict__ out, long total, int sk, bool f32) { reduce_body(part, out, total, sk, f32); }
// A Tiled weight -> dense fp16 (N, K) (grid ceil(n*K/8/256)), block 256; suffix fmt 0/1/2.
extern "C" __global__ void tf_vqmmf884s_dequant_0(const int* __restrict__ W, const unsigned* __restrict__ S, __half* __restrict__ out, int N, int K) { dequant_body<0>(W, S, out, N, K); }
extern "C" __global__ void tf_vqmmf884s_dequant_1(const int* __restrict__ W, const unsigned* __restrict__ S, __half* __restrict__ out, int N, int K) { dequant_body<1>(W, S, out, N, K); }
extern "C" __global__ void tf_vqmmf884s_dequant_2(const int* __restrict__ W, const unsigned* __restrict__ S, __half* __restrict__ out, int N, int K) { dequant_body<2>(W, S, out, N, K); }
// fp32 sums -> y * rs[row] * alpha[col], bf16 or fp32 (grid ceil(total/256)), block 256.
extern "C" __global__ void tf_vqmmf884s_unscale2(const float* __restrict__ y, const float* __restrict__ rs, const float* __restrict__ alpha, void* __restrict__ out, long N, long total, bool f32) { unscale2_body(y, rs, alpha, out, N, total, f32); }
// Rows of a page-locked host table read over PCIe (grid ids), block 128.
extern "C" __global__ void tf_vqmmf884s_gather_rows(const long* __restrict__ ids, const uint4* __restrict__ table, uint4* __restrict__ out, int row16) { gather_rows_body(ids, table, out, row16); }
// Plain X [M][K] -> tiled (grid ceil(ceil(M/8)*K/8/256)), block 256.
extern "C" __global__ void tf_vqmmf884s_tile_x(const __half* __restrict__ X, __half* __restrict__ Xt, int M, int K) { tile_x_body(X, Xt, M, K); }

// ---------------------------------------------------------------------------
// Dense BF16 weights on sm_70 (the qmmf Mode-4 path has no sm_80-free kernel,
// so it lives here). X is raw bf16 [M][ldx] (rows ldx apart, ldx % 8 == 0); W
// is the packed bf16-lane layout [npad/64][KG][8 jj][4 kt][32 lane][4 e] u16
// (qmmf.packBf16). Both convert to fp16 in-kernel and accumulate on the same
// mma.m8n8k4.f16 tensor cores as the quantized lanes; `scale` is one per-call
// weight factor, applied in the epilogue like alpha. A thread's column reads a
// kt block as 8 consecutive u32 (coalesced across its quadpair). grid
// (ceil(N/COLS), sk, ceil(M/(8*MT))), block THREADS.
// ---------------------------------------------------------------------------

// Two bf16 (lo, hi) in one u32 -> two fp16 (exact in the normal range; tiny
// subnormals and overflows flush to 0 / inf with the sign kept).
__device__ __forceinline__ unsigned bf16x2_to_h2u(unsigned u) {
    auto cvt = [](unsigned b) -> unsigned {
        const unsigned s = b & 0x8000u, e8 = (b >> 7) & 0xFFu;
        if (e8 == 0) return s;                        // zero or bf16 subnormal -> 0
        if (e8 == 0xFF) return s | 0x7C00u;           // inf / nan -> fp16 inf
        const int e5 = (int)e8 - 112;                 // bias 127 -> 15
        if (e5 <= 0) return s;                        // below fp16 normal -> 0
        if (e5 >= 31) return s | 0x7C00u;             // overflow -> inf
        return s | ((unsigned)e5 << 10) | ((b & 0x7Fu) << 3);
    };
    return cvt(u & 0xFFFFu) | (cvt(u >> 16) << 16);
}

template <int MT>
__device__ __forceinline__ void
bf16_body(const __nv_bfloat16* __restrict__ X, const unsigned* __restrict__ W, float scale,
          void* __restrict__ OUT, float* __restrict__ PART, int* __restrict__ CNT,
          int M, int N, int K, int gper, int sk, int ldx, bool f32) {
    constexpr int NACC = 2;
    constexpr int AHEAD = 3, STAGES = AHEAD + 1;
    __shared__ __align__(16) __half sa[STAGES][MT][512];   // [stage][tile][8 wi x 8 rows x 8 inputs] fp16
    __shared__ int last;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int q = (lane >> 2) & 3, idx = (lane & 3) + 4 * (lane >= 16);
    const int s = blockIdx.y, m0 = blockIdx.z * 8 * MT;
    const int g0 = s * gper, g1 = g0 + gper;
    const int KG = K / 64;
    const int cbase = blockIdx.x * COLS + warp * 32 + q * 8;
    const int col = cbase + idx;
    const bool nok = col < N;
    const int c32 = nok ? col : 0;
    const int jj = (c32 & 63) >> 3, r7 = c32 & 7;          // packBf16 column factors
    int tiles = 0;
#pragma unroll
    for (int t = 0; t < MT; ++t) tiles += m0 + t * 8 < M;

    // Staging: tile t of group g is [wi=8][row=8] int4s; c = wi*8+row, the 8
    // bf16 of row m0+t*8+row at k g*64+wi*8..+7, converted to fp16 on the way in.
    constexpr int PER = (MT * 64 + THREADS - 1) / THREADS;
    auto fetch = [&](int g, int4 (&r)[PER]) {
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int e = threadIdx.x + j * THREADS, t = e >> 6, c = e & 63;
            int4 h;
            if (e < MT * 64 && t < tiles && g < g1) {
                const int m = m0 + t * 8 + (c & 7);
                if (m < M) {
                    const int4 b = __ldg(reinterpret_cast<const int4*>(X + (long)m * ldx + g * 64 + (c >> 3) * 8));
                    const unsigned* u = reinterpret_cast<const unsigned*>(&b);
                    unsigned* hw = reinterpret_cast<unsigned*>(&h);
#pragma unroll
                    for (int w = 0; w < 4; ++w) hw[w] = bf16x2_to_h2u(u[w]);
                }
            }
            r[j] = h;
        }
    };
    auto store = [&](int g, const int4 (&r)[PER]) {
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int e = threadIdx.x + j * THREADS, t = e >> 6, c = e & 63;
            if (e < MT * 64) reinterpret_cast<int4*>(sa[g % STAGES][t])[c] = r[j];
        }
    };
    {
        int4 r[PER];
#pragma unroll
        for (int d = 0; d < AHEAD; ++d) {
            fetch(g0 + d, r);
            store(d, r);
        }
    }
    __syncthreads();

    float acc[MT][NACC][8];
#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
        for (int a = 0; a < NACC; ++a)
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[t][a][i] = 0.f;

    // The packed-lane base for this column at group g (u32 words).
    const unsigned* wbase = W + (long)(c32 >> 6) * KG * 2048;
#pragma unroll
    for (int g = g0; g < g1; ++g) {
        const int i = g - g0;
        int4 ahead[PER];
        fetch(g + AHEAD, ahead);
        const __half* stage = &sa[i % STAGES][0][0];
#pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            unsigned chunk[8];
            const unsigned* cp = wbase + (long)g * 2048 + (jj * 4 + kt) * 64 + r7 * 8;
#pragma unroll
            for (int o = 0; o < 8; ++o) chunk[o] = nok ? __ldcs(cp + o) : 0u;
#pragma unroll
            for (int sub = 0; sub < 2; ++sub) {
                const int wi = 2 * kt + sub;
                int4 o;
                unsigned* ou = reinterpret_cast<unsigned*>(&o);
#pragma unroll
                for (int h = 0; h < 4; ++h) ou[h] = bf16x2_to_h2u(chunk[2 * h + sub]);
#pragma unroll
                for (int t = 0; t < MT; ++t) {
                    if (t >= tiles) continue;
                    const uint4 a = *reinterpret_cast<const uint4*>(stage + t * 512 + wi * 64 + idx * 8);
                    mma884(acc[t][(2 * wi) % NACC], a.x, a.y, (unsigned)o.x, (unsigned)o.y);
                    mma884(acc[t][(2 * wi + 1) % NACC], a.z, a.w, (unsigned)o.z, (unsigned)o.w);
                }
            }
        }
        store(i + AHEAD, ahead);
        __syncthreads();
    }

#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
        for (int a = 1; a < NACC; ++a)
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[t][0][i] += acc[t][a][i];

#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int m = m0 + t * 8 + (lane & 1) + 2 * ((i >> 1) & 1) + 4 * (lane >= 16);
            const int n = cbase + (i & 1) + 2 * ((lane >> 1) & 1) + 4 * (i >> 2);
            if (m >= M || n >= N) continue;
            const float v = acc[t][0][i] * scale;
            if (sk > 1) PART[((long)s * M + m) * N + n] = v;
            else if (f32) static_cast<float*>(OUT)[(long)m * N + n] = v;
            else static_cast<__nv_bfloat16*>(OUT)[(long)m * N + n] = __float2bfloat16_rn(v);
        }
    if (sk == 1 || CNT == nullptr) return;
    __threadfence();
    __syncthreads();
    const int tile_id = blockIdx.z * gridDim.x + blockIdx.x;
    if (threadIdx.x == 0) last = atomicAdd(CNT + tile_id, 1) == sk - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    for (int e = threadIdx.x; e < 8 * MT * COLS; e += THREADS) {
        const int m = m0 + e / COLS, n = blockIdx.x * COLS + e % COLS;
        if (m >= M || n >= N) continue;
        const long at = (long)m * N + n, total = (long)M * N;
        float v[16];
#pragma unroll
        for (int z = 0; z < 16; ++z) v[z] = z < sk ? __ldcg(PART + z * total + at) : 0.f;
        float a = v[0];
#pragma unroll
        for (int z = 1; z < 16; ++z) if (z < sk) a += v[z];
        if (f32) static_cast<float*>(OUT)[at] = a;
        else static_cast<__nv_bfloat16*>(OUT)[at] = __float2bfloat16_rn(a);
    }
    if (threadIdx.x == 0) CNT[tile_id] = 0;
}

// grid (ceil(N/COLS), sk, ceil(M/(8*MT))), block THREADS; suffix MT in {1,2,4}.
extern "C" __global__ void __launch_bounds__(THREADS) tf_vbf16_1(const __nv_bfloat16* X, const unsigned* W, float scale, void* OUT, float* PART, int* CNT, int M, int N, int K, int gper, int sk, int ldx, int f32) { bf16_body<1>(X, W, scale, OUT, PART, CNT, M, N, K, gper, sk, ldx, f32 != 0); }
extern "C" __global__ void __launch_bounds__(THREADS) tf_vbf16_2(const __nv_bfloat16* X, const unsigned* W, float scale, void* OUT, float* PART, int* CNT, int M, int N, int K, int gper, int sk, int ldx, int f32) { bf16_body<2>(X, W, scale, OUT, PART, CNT, M, N, K, gper, sk, ldx, f32 != 0); }
extern "C" __global__ void __launch_bounds__(THREADS) tf_vbf16_4(const __nv_bfloat16* X, const unsigned* W, float scale, void* OUT, float* PART, int* CNT, int M, int N, int K, int gper, int sk, int ldx, int f32) { bf16_body<4>(X, W, scale, OUT, PART, CNT, M, N, K, gper, sk, ldx, f32 != 0); }
