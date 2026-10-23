// Volta (sm_70) DFlash2 block attention: draft_attention.py's _block_attention without tl.dot.
//
// Drafts only propose (the target verifies every token), so the context is split across blocks for parallelism:
// block (stream j, kv head g, split c) takes KC context keys (the last split takes the block's own keys) for the
// G x L query rows of that kv head and writes a partial (max, sum, unnormalized output); a second kernel folds
// the splits in order and normalizes. Masks follow the Triton kernel: a context key is seen while
// s + r - key < window + 1, a block key always (causal: key <= r).

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int KC = 32, THREADS = 128;           // keys a split (a block's own keys fit one): 74 KB of smem at 64 rows x D 128

__device__ __forceinline__ float bf(unsigned short v) { return __uint_as_float(static_cast<unsigned>(v) << 16); }

template <int D>
__device__ __forceinline__ void split_body(
        const unsigned short* __restrict__ Q, const unsigned short* __restrict__ KB, const unsigned short* __restrict__ VB,
        const long* __restrict__ TABLE, const long* __restrict__ LENS, float* __restrict__ PM, float* __restrict__ PL,
        float* __restrict__ PO, int R, int L, int G, int window, int nsplit, bool causal, float scale) {
    const int j = blockIdx.x, g = blockIdx.y, c = blockIdx.z;
    const int rows = G * L;                                  // query rows of this kv head: (head, block row)
    const int s = (int)LENS[j];
    const bool block_keys = c == nsplit - 1;
    const int k0 = c * KC;
    const int nk = block_keys ? L : max(0, min(KC, s - k0));

    extern __shared__ float smem[];
    float* sq = smem;                                        // rows x D
    float* sk = sq + rows * D;                               // KC x (D + 1)
    float* sv = sk + KC * (D + 1);                           // KC x D
    float* sp = sv + KC * D;                                 // rows x KC
    const long pbase = (((long)j * gridDim.y + g) * nsplit + c) * rows;

    if (nk == 0) {
        for (int i = threadIdx.x; i < rows; i += THREADS) { PM[pbase + i] = -INFINITY; PL[pbase + i] = 0.f; }
        for (int i = threadIdx.x; i < rows * D; i += THREADS) PO[pbase * D + i] = 0.f;
        return;
    }
    for (int i = threadIdx.x; i < rows * D; i += THREADS) {
        const int row = i / D, d = i % D, head = g * G + row / L, r = row % L;
        sq[i] = bf(Q[((long)head * R + j * L + r) * D + d]);
    }
    const unsigned short* kc = reinterpret_cast<const unsigned short*>(TABLE[2 * j]) + (long)g * s * D;
    const unsigned short* vc = reinterpret_cast<const unsigned short*>(TABLE[2 * j + 1]) + (long)g * s * D;
    for (int i = threadIdx.x; i < nk * D; i += THREADS) {
        const int t = i / D, d = i % D;
        const long src = block_keys ? ((long)g * R + j * L + t) * D + d : (long)(k0 + t) * D + d;
        sk[t * (D + 1) + d] = bf(block_keys ? KB[src] : kc[src]);
        sv[t * D + d] = bf(block_keys ? VB[src] : vc[src]);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < rows * KC; i += THREADS) {
        const int row = i / KC, t = i % KC, r = row % L;
        float a = -INFINITY;
        bool seen = t < nk;
        if (seen) seen = block_keys ? (!causal || t <= r) : (s + r - (k0 + t) < window + 1);
        if (seen) {
            a = 0.f;
            const float* qr = sq + row * D;
            const float* kr = sk + t * (D + 1);
#pragma unroll 8
            for (int d = 0; d < D; ++d) a = fmaf(qr[d], kr[d], a);
            a *= scale;
        }
        sp[i] = a;
    }
    __syncthreads();
    // per row: max and sum over this split's keys (one thread a row)
    for (int row = threadIdx.x; row < rows; row += THREADS) {
        float m = -INFINITY;
        for (int t = 0; t < KC; ++t) m = fmaxf(m, sp[row * KC + t]);
        float l = 0.f;
        for (int t = 0; t < KC; ++t) {
            const float e = m == -INFINITY ? 0.f : expf(sp[row * KC + t] - m);
            sp[row * KC + t] = e;
            l += e;
        }
        PM[pbase + row] = m;
        PL[pbase + row] = l;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < rows * D; i += THREADS) {
        const int row = i / D, d = i % D;
        float o = 0.f;
        for (int t = 0; t < nk; ++t) o = fmaf(sp[row * KC + t], sv[t * D + d], o);
        PO[pbase * D + i] = o;
    }
}

__device__ __forceinline__ void fold_body(const float* __restrict__ PM, const float* __restrict__ PL, const float* __restrict__ PO,
                            unsigned short* __restrict__ OUT, int L, int G, int HKV, int D, int nsplit) {
    const int j = blockIdx.x, g = blockIdx.y, row = blockIdx.z;   // row: (head in group, block row)
    const int rows = G * L, head = g * G + row / L, r = row % L;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float M = -INFINITY, Ls = 0.f, O = 0.f;
        for (int c = 0; c < nsplit; ++c) {
            const long b = (((long)j * HKV + g) * nsplit + c) * rows + row;
            const float cm = PM[b], cl = PL[b];
            if (!(cl > 0.f)) continue;
            const float nm = fmaxf(M, cm);
            const float a = M == -INFINITY ? 0.f : expf(M - nm), e = expf(cm - nm);
            O = O * a + PO[b * D + d] * e;
            Ls = Ls * a + cl * e;
            M = nm;
        }
        const float v = O / Ls;
        const __nv_bfloat16 h = __float2bfloat16_rn(v);
        OUT[((long)j * L + r) * (G * HKV * D) + (long)head * D + d] = *reinterpret_cast<const unsigned short*>(&h);
    }
}

}  // namespace

// Engine entry points: the sm_70 image is a fatbin, not a static lib — the Zig runtime loads it with cuModuleLoad
// and resolves these by name. D (head dim) is the suffix; grid (S, HKV, nsplit), block THREADS, with dynamic
// shared memory the caller sizes (rows*D + KC*(D+1) + KC*D + rows*KC floats).
extern "C" __global__ void __launch_bounds__(THREADS) tf_vdraft_split_128(const unsigned short* __restrict__ Q, const unsigned short* __restrict__ KB, const unsigned short* __restrict__ VB, const long* __restrict__ TABLE, const long* __restrict__ LENS, float* __restrict__ PM, float* __restrict__ PL, float* __restrict__ PO, int R, int L, int G, int window, int nsplit, int causal, float scale) { split_body<128>(Q, KB, VB, TABLE, LENS, PM, PL, PO, R, L, G, window, nsplit, causal != 0, scale); }
extern "C" __global__ void __launch_bounds__(THREADS) tf_vdraft_split_64(const unsigned short* __restrict__ Q, const unsigned short* __restrict__ KB, const unsigned short* __restrict__ VB, const long* __restrict__ TABLE, const long* __restrict__ LENS, float* __restrict__ PM, float* __restrict__ PL, float* __restrict__ PO, int R, int L, int G, int window, int nsplit, int causal, float scale) { split_body<64>(Q, KB, VB, TABLE, LENS, PM, PL, PO, R, L, G, window, nsplit, causal != 0, scale); }

// Fold the splits in order and normalize (grid (S, HKV, rows)), block 128; OUT bf16 (R, H*D).
extern "C" __global__ void tf_vdraft_fold(const float* __restrict__ PM, const float* __restrict__ PL, const float* __restrict__ PO, unsigned short* __restrict__ OUT, int L, int G, int HKV, int D, int nsplit) { fold_body(PM, PL, PO, OUT, L, G, HKV, D, nsplit); }
