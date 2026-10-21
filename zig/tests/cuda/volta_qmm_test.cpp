// Standalone verification of volta_qmm.cu on the Volta GPU (Quadro GV100): the engine's fatbin kernels are linked
// straight in (extern "C" symbols) and driven through cudaLaunchKernel, with CPU references.
//
// Bit-exact where the semantics are closed form: prep (power-of-two row scaling + fp16 RNE), dequant (fp16 FMA of
// exact values), reduce (slice-order fp32 sums), unscale (fp32 multiplies). For the tensor-core decode and the
// cuBLASLt prompt GEMM, the accumulation order is hardware-chosen, so those are checked two ways: structural
// invariants that must hold bit for bit (the row-tile count, the Tiled vs stored layout, and the fragment vs
// row-major rows never change a row's bits; the prompt's pinned algorithm keeps its chain across row counts) and a
// tolerance against a double reference built from the *kernel's own* expanded fp16 operands. The K split is the one
// axis that changes the fp32 addition tree (slice partials summed in slice order), so it is covered by the
// tolerance, not the bit check.
//
// Build: nvcc -arch=sm_70 -O3 -std=c++20 -o volta_qmm_test volta_qmm_test.cpp volta_qmm.cu -lcublasLt

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublasLt.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

// ---- the image's symbols, as the engine's Module.function resolves them by name -------------------------------------

extern "C" {
__global__ void tf_vprep(const unsigned short* x, long ldx, int M, int K, __half* out, float* rs);
__global__ void tf_vprep_frag(const unsigned short* x, long ldx, int M, int K, __half* out, float* rs);
__global__ void tf_vprep_any(const unsigned short* x, long ldx, int M, int K, __half* out, float* rs);
__global__ void tf_vprep_frag_any(const unsigned short* x, long ldx, int M, int K, __half* out, float* rs);
__global__ void tf_vqmm884_1_0_0(const __half* X, const float* RS, const int* W, const unsigned short* S, const unsigned short* B, void* OUT, float* PART, int M, int N, int K, int gper, int sk, int f32);
__global__ void tf_vqmm884_1_0_1(const __half* X, const float* RS, const int* W, const unsigned short* S, const unsigned short* B, void* OUT, float* PART, int M, int N, int K, int gper, int sk, int f32);
__global__ void tf_vqmm884_1_1_0(const __half* X, const float* RS, const int* W, const unsigned short* S, const unsigned short* B, void* OUT, float* PART, int M, int N, int K, int gper, int sk, int f32);
__global__ void tf_vqmm884_1_1_1(const __half* X, const float* RS, const int* W, const unsigned short* S, const unsigned short* B, void* OUT, float* PART, int M, int N, int K, int gper, int sk, int f32);
__global__ void tf_vqmm884_2_0_0(const __half* X, const float* RS, const int* W, const unsigned short* S, const unsigned short* B, void* OUT, float* PART, int M, int N, int K, int gper, int sk, int f32);
__global__ void tf_vqmm884_2_0_1(const __half* X, const float* RS, const int* W, const unsigned short* S, const unsigned short* B, void* OUT, float* PART, int M, int N, int K, int gper, int sk, int f32);
__global__ void tf_vqmm884_2_1_0(const __half* X, const float* RS, const int* W, const unsigned short* S, const unsigned short* B, void* OUT, float* PART, int M, int N, int K, int gper, int sk, int f32);
__global__ void tf_vqmm884_2_1_1(const __half* X, const float* RS, const int* W, const unsigned short* S, const unsigned short* B, void* OUT, float* PART, int M, int N, int K, int gper, int sk, int f32);
__global__ void tf_vqmm884_4_0_0(const __half* X, const float* RS, const int* W, const unsigned short* S, const unsigned short* B, void* OUT, float* PART, int M, int N, int K, int gper, int sk, int f32);
__global__ void tf_vqmm884_4_0_1(const __half* X, const float* RS, const int* W, const unsigned short* S, const unsigned short* B, void* OUT, float* PART, int M, int N, int K, int gper, int sk, int f32);
__global__ void tf_vqmm884_4_1_0(const __half* X, const float* RS, const int* W, const unsigned short* S, const unsigned short* B, void* OUT, float* PART, int M, int N, int K, int gper, int sk, int f32);
__global__ void tf_vqmm884_4_1_1(const __half* X, const float* RS, const int* W, const unsigned short* S, const unsigned short* B, void* OUT, float* PART, int M, int N, int K, int gper, int sk, int f32);
__global__ void tf_vreduce(const float* part, void* out, long total, int sk, int f32);
__global__ void tf_vdequant(const unsigned* w, const unsigned short* s, const unsigned short* b, __half* out, int N, int K);
__global__ void tf_vunscale(const float* y, const float* rs, void* out, long N, long total4, int f32);
}

// ---- deterministic data ---------------------------------------------------------------------------------------------

static uint32_t seed = 0x1234567u;
static uint32_t rng() { seed = seed * 1664525u + 1013904223u; return seed; }
static double unif() { return (double)(rng() >> 8) / 16777216.0 * 2.0 - 1.0; }

static uint16_t to_bf16(double v) {
    uint64_t b; memcpy(&b, &v, 8);
    uint32_t s = (uint32_t)(b >> 63);
    int32_t e = (int32_t)((b >> 52) & 0x7FF);
    uint64_t m = b & 0xFFFFFFFFFFFFFULL;
    if (e == 0x7FF) return (uint16_t)((s << 15) | 0x7F80);   // inf/NaN -> inf
    if (e == 0) return (uint16_t)(s << 15);                   // zero (subnormals flushed)
    uint64_t keep = m >> 45;                                  // top 7 mantissa bits
    uint64_t rem  = m & ((1ULL << 45) - 1);
    uint64_t half = 1ULL << 44;
    uint32_t mant7 = (uint32_t)keep;
    if (rem > half || (rem == half && (keep & 1))) mant7 += 1;  // RNE
    if (mant7 == 128) { mant7 = 0; e += 1; }                    // carry into exponent
    int32_t bexp = e - 896;                                     // re-bias 1023 -> 127
    if (bexp >= 255) return (uint16_t)((s << 15) | 0x7F80);     // overflow -> inf
    if (bexp <= 0) return (bexp == 0) ? (uint16_t)((s << 15) | mant7) : (uint16_t)(s << 15);
    return (uint16_t)((s << 15) | ((uint32_t)bexp << 7) | mant7);
}
static float bf2f(uint16_t v) { uint32_t u = (uint32_t)v << 16; float f; memcpy(&f, &u, 4); return f; }
static double bf2d(uint16_t v) { return (double)bf2f(v); }
static float h2f(uint16_t bits) { __half_raw r; r.x = bits; return __half2float(r); }

// fp16 round-to-nearest-even of an exact double, independent of the CUDA headers.
static uint16_t fp16_rne(double v) {
    const bool neg = std::signbit(v);
    v = std::fabs(v);
    if (std::isnan(v)) return (neg ? 0x8000u : 0u) | 0x7E00u;
    if (v == 0.0) return neg ? 0x8000u : 0u;
    if (v >= 65520.0) return (neg ? 0x8000u : 0u) | 0x7C00u;
    if (v < std::ldexp(1.0, -14)) {                    // fp16 subnormal: v = m * 2^-24
        const double m = v / std::ldexp(1.0, -24);
        uint32_t i = (uint32_t)std::floor(m);
        const double f = m - i;
        if (f > 0.5 || (f == 0.5 && (i & 1))) i += 1;
        if (i == 1024) return (neg ? 0x8000u : 0u) | 0x0400u;   // rounds up to smallest normal
        return (neg ? 0x8000u : 0u) | (uint16_t)i;
    }
    int e;
    const double m = std::frexp(v, &e);                // v = m * 2^e, m in [0.5, 1): exact
    const double mant = 2.0 * m - 1.0;                 // in [0, 1): exact
    const double mm = mant * 1024.0;                   // 10-bit fraction: exact
    uint32_t i = (uint32_t)std::floor(mm);
    const double f = mm - i;
    if (f > 0.5 || (f == 0.5 && (i & 1))) i += 1;
    int exp = e - 1;
    if (i == 1024) { i = 0; exp += 1; }
    if (exp > 15) return (neg ? 0x8000u : 0u) | 0x7C00u;
    return (neg ? 0x8000u : 0u) | (uint16_t)(((uint32_t)(exp + 15) << 10) | i);
}

// bf16 round-to-nearest-even of an exact double (no subnormal territory in these tests).
static uint16_t bf16_rne(double v) {
    const bool neg = std::signbit(v);
    v = std::fabs(v);
    if (std::isnan(v)) return (neg ? 0x8000u : 0u) | 0x7FC0u;
    if (v == 0.0) return neg ? 0x8000u : 0u;
    if (v >= 3.3895313892515355e38) return (neg ? 0x8000u : 0u) | 0x7F80u;
    int e;
    const double m = std::frexp(v, &e);
    const double mant = 2.0 * m - 1.0;
    const double mm = mant * 128.0;                   // 7-bit fraction, RNE
    uint32_t i = (uint32_t)std::floor(mm);
    const double f = mm - i;
    if (f > 0.5 || (f == 0.5 && (i & 1))) i += 1;
    int exp = e - 1;
    if (i == 128) { i = 0; exp += 1; }
    if (exp > 127) return (neg ? 0x8000u : 0u) | 0x7F80u;
    return (neg ? 0x8000u : 0u) | (uint16_t)(((uint32_t)(exp + 127) << 7) | i);
}

// ---- check bookkeeping ------------------------------------------------------------------------------------------------

static int failures = 0;
static void check(bool ok, const char* what, long extra) {
    if (ok) { printf("  ok   %s\n", what); return; }
    failures++;
    printf("  FAIL %s (extra=%ld)\n", what, extra);
}

static int same_fp16(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b, long& first) {
    first = -1;
    if (a.size() != b.size()) return 0;
    for (size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) { first = (long)i; return 0; }
    return 1;
}
static int same_f32(const std::vector<float>& a, const std::vector<float>& b, long& first) {
    first = -1;
    if (a.size() != b.size()) return 0;
    for (size_t i = 0; i < a.size(); ++i) {
        uint32_t x, y; memcpy(&x, &a[i], 4); memcpy(&y, &b[i], 4);
        if (x != y) { first = (long)i; return 0; }
    }
    return 1;
}
// check(same_f32(a, b, first), msg, first) is argument-order UB for `first`; compare-then-report instead.
static void check_fp16(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b, const char* what) {
    long first = -1;
    const int ok = same_fp16(a, b, first);
    check(ok, what, first);
}
static void check_f32(const std::vector<float>& a, const std::vector<float>& b, const char* what) {
    long first = -1;
    const int ok = same_f32(a, b, first);
    check(ok, what, first);
}

// The host plan, as the Python reference has it.
static int split_k884(int n, int k) {
    int tiles = (n + 127) / 128, groups = k / 64, sk = 1;
    while (sk < 16 && tiles * sk < 640 && groups % (sk * 2) == 0 && groups / (sk * 2) >= 4) sk *= 2;
    return sk;
}
static int row_tiles(int m) { return m <= 8 ? 1 : m <= 16 ? 2 : 4; }

// ---- the weight set ---------------------------------------------------------------------------------------------------

// A random 4-bit weight (N, K) in the stored layout: words (N, K/8), bf16 scales/biases (N, K/64); the Tiled
// repack [N/32][K/64][32][8] (scales and biases [N/32][K/64][32]), padded to 32 columns.
struct Weight {
    int n, k, kg;
    std::vector<uint32_t> words;          // stored (N, K/8)
    std::vector<uint16_t> scales, biases; // stored (N, K/64)
    std::vector<uint32_t> tw;             // tiled words (N32, KG, 32, 8)
    std::vector<uint16_t> ts, tb;         // tiled scales/biases (N32, KG, 32)
    std::vector<uint16_t> w16;            // dense fp16 (N, K): the expanded values, the dequant's reference
    void make(int n_, int k_) {
        n = n_; k = k_; kg = k / 64;
        const int k8 = k / 8, np = (n + 31) / 32 * 32;
        words.assign((size_t)n * k8, 0);
        scales.assign((size_t)n * kg, 0);
        biases.assign((size_t)n * kg, 0);
        for (int i = 0; i < n * k8; ++i) words[i] = rng() & 0x0F0F0F0Fu;
        for (int i = 0; i < n * kg; ++i) {
            scales[i] = to_bf16(0.002 + 0.01 * ((double)(rng() >> 8) / 16777216.0));   // positive, ~2^-9..2^-6
            biases[i] = to_bf16(unif());
        }
        tw.assign((size_t)np * kg * 32 * 8, 0);
        ts.assign((size_t)np * kg * 32, 0);
        tb.assign((size_t)np * kg * 32, 0);
        w16.assign((size_t)n * k, 0);
        for (int r = 0; r < n; ++r)
            for (int g = 0; g < kg; ++g) {
                const int cell = ((r >> 5) * kg + g) * 32 + (r & 31);
                for (int j = 0; j < 8; ++j) tw[cell * 8 + j] = words[r * k8 + g * 8 + j];
                ts[cell] = scales[r * kg + g];
                tb[cell] = biases[r * kg + g];
            }
        for (int r = 0; r < n; ++r)
            for (int g = 0; g < kg; ++g)
                for (int j = 0; j < 8; ++j)
                    for (int i = 0; i < 8; ++i) {
                        const uint32_t q = (words[r * k8 + g * 8 + j] >> (4 * i)) & 0xF;
                        const double v = (double)q * bf2d(scales[r * kg + g]) + bf2d(biases[r * kg + g]);
                        w16[(size_t)r * k + g * 64 + j * 8 + i] = fp16_rne(v);
                    }
    }
};

// ---- device helpers ---------------------------------------------------------------------------------------------------

template <class T>
static T* dev(const T* p, size_t n) {
    T* d;
    cudaError_t e1 = cudaMalloc((void**)&d, n * sizeof(T));
    cudaError_t e2 = cudaMemcpy(d, p, n * sizeof(T), cudaMemcpyHostToDevice);
    if (e1 != cudaSuccess || e2 != cudaSuccess)
        fprintf(stderr, "dev: n=%zu malloc=%d copy=%d d=%p\n", n, (int)e1, (int)e2, (void*)d);
    return d;
}
template <class T>
static T* dev(const std::vector<T>& v) { return dev(v.data(), v.size()); }
template <class T>
static std::vector<T> host(T* p, size_t n) {
    std::vector<T> v(n);
    cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    return v;
}

// ---- the checks -------------------------------------------------------------------------------------------------------

// prep: bf16 rows -> scaled fp16 + the row scales, row-major and fragment layouts, fast and any kernels.
static void test_prep() {
    printf("prep\n");
    for (int k = 0; k < 2; ++k) {
        const int M = 5, K = k == 0 ? 4096 : 20608;         // 20608/8 > 2560: the any kernel's territory
        std::vector<uint16_t> x((size_t)M * (K + (k == 1 ? 2 : 0)));
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < K; ++j) {
                if (j % 997 == 0) x[i * (K + (k == 1 ? 2 : 0)) + j] = to_bf16(0.0);
                else if (j % 991 == 0) x[i * (K + (k == 1 ? 2 : 0)) + j] = to_bf16(1e-10);   // fp16-subnormal after scaling
                else x[i * (K + (k == 1 ? 2 : 0)) + j] = to_bf16(unif() * (j % 893 == 0 ? 4000.0 : 1.0));
            }
        const long ldx = k == 0 ? K : K + 2;                // the any path also takes a non-multiple-of-8 stride
        std::vector<uint16_t> ref_rm((size_t)M * K), ref_fr((size_t)((M + 7) / 8) * 8 * K);
        std::vector<float> rs(M, 1.f);
        for (int r = 0; r < M; ++r) {
            float amax = 0.f;
            for (int c = 0; c < K; ++c) amax = std::fmaxf(amax, fabsf(bf2f(x[r * ldx + c])));
            const int e = (amax > 0.f && std::isfinite(amax)) ? std::ilogb(amax) - 14 : 0;
            rs[r] = ldexpf(1.f, e);
            for (int c2 = 0; c2 < K; ++c2) {
                const uint16_t h = fp16_rne(ldexp(bf2d(x[r * ldx + c2]), -e));
                ref_rm[r * K + c2] = h;
                ref_fr[((r >> 3) * (K / 8) + (c2 >> 3)) * 64 + (r & 7) * 8 + (c2 & 7)] = h;
            }
        }
        const bool fast = (K / 8 <= 2560) && (ldx % 8 == 0);
        const unsigned short* xd = dev(x);
        for (int frag = 0; frag < 2; ++frag) {
            void (*kern)(const unsigned short*, long, int, int, __half*, float*);
            if (frag == 0) kern = fast ? tf_vprep : tf_vprep_any;
            else kern = fast ? tf_vprep_frag : tf_vprep_frag_any;
            const int grid = frag ? (M + 7) / 8 * 8 : M;
            const int block = fast ? 512 : 256;
            const size_t bytes = frag ? (size_t)((M + 7) / 8) * 8 * K * 2 : (size_t)M * K * 2;
            __half* xo; float* ro;
            cudaMalloc((void**)&xo, bytes); cudaMalloc((void**)&ro, M * 4);
            cudaMemset(xo, 0xEE, bytes);
            void* args[] = {(void*)&xd, (void*)&ldx, (void*)&M, (void*)&K, (void*)&xo, (void*)&ro};
            cudaError_t err = cudaLaunchKernel((void*)kern, dim3(grid), dim3(block), args, 0, 0);
            check(err == cudaSuccess, frag ? "prep_frag launches" : "prep launches", (long)err);
            std::vector<uint16_t> got(bytes / 2);
            cudaMemcpy(got.data(), xo, bytes, cudaMemcpyDeviceToHost);
            std::vector<float> rgot = host(ro, M);
            cudaFree(xo); cudaFree(ro);
            const std::vector<uint16_t>& ref = frag ? ref_fr : ref_rm;
            check_fp16(got, ref, frag ? "prep_frag bit-exact" : "prep bit-exact");
            check_f32(rgot, rs, frag ? "prep_frag row scales" : "prep row scales");
        }
        cudaFree((void*)xd);
    }
}

// dequant: the Tiled words -> dense fp16, the exact fp16(q * s + b) values.
static void test_dequant(Weight& w) {
    printf("dequant (N=%d, K=%d)\n", w.n, w.k);
    const int kw = w.k / 8;
    __half* out;
    cudaMalloc((void**)&out, (size_t)w.n * w.k * 2);
    cudaMemset(out, 0xEE, (size_t)w.n * w.k * 2);
    uint32_t* tw = dev(w.tw); uint16_t* ts = dev(w.ts); uint16_t* tb = dev(w.tb);
    void* args[] = {(void*)&tw, (void*)&ts, (void*)&tb, (void*)&out, (void*)&w.n, (void*)&w.k};
    check(cudaLaunchKernel((void*)tf_vdequant, dim3((w.n * kw + 255) / 256), dim3(256), args, 0, 0) == cudaSuccess,
          "dequant launches", 0);
    std::vector<uint16_t> got = host((uint16_t*)out, (size_t)w.n * w.k);
    cudaFree(out); cudaFree(tw); cudaFree(ts); cudaFree(tb);
    check_fp16(got, w.w16, "dequant bit-exact vs the expanded values");
}

// qmm884: the row-tile count and the layouts (Tiled vs stored words, fragment vs row-major rows) never change a
// row's bits (the same mma chain per row); the K split changes the fp32 addition tree, so it is checked against
// the double reference of the expanded fp16 operands within the fp32 accumulation bound.
static void test_qmm(Weight& w, int M) {
    const int N = w.n, K = w.k, kg = w.kg, k8 = K / 8;
    printf("qmm884 (M=%d, N=%d, K=%d, sk=%d, mt=%d)\n", M, N, K, split_k884(N, K), row_tiles(M));
    std::vector<uint16_t> xb((size_t)M * K);
    for (int i = 0; i < M * K; ++i) xb[i] = (i % K % 733 == 0) ? to_bf16(1e-10) : to_bf16(unif());
    const unsigned short* xbg = dev(xb);

    // Device copies of both weight layouts, once.
    uint32_t* dw = dev(w.words); uint16_t* ds = dev(w.scales); uint16_t* db = dev(w.biases);
    uint32_t* dtw = dev(w.tw); uint16_t* dts = dev(w.ts); uint16_t* dtb = dev(w.tb);
    const long ldx = K;                       // tf_vprep takes a 64-bit ldx

    // The rows in both layouts, from the tested prep kernels (their output is the matmul's operand).
    auto prep_rows = [&](bool frag) {
        __half* xo; float* ro;
        const size_t bytes = frag ? (size_t)((M + 7) / 8) * 8 * K * 2 : (size_t)M * K * 2;
        cudaMalloc((void**)&xo, bytes); cudaMalloc((void**)&ro, M * 4);
        void* args[] = {(void*)&xbg, (void*)&ldx, (void*)&M, (void*)&K, (void*)&xo, (void*)&ro};
        cudaLaunchKernel((void*)(frag ? tf_vprep_frag : tf_vprep), dim3(frag ? (M + 7) / 8 * 8 : M), dim3(512), args, 0, 0);
        cudaDeviceSynchronize();
        std::vector<uint16_t> x16(bytes / 2);
        cudaMemcpy(x16.data(), xo, bytes, cudaMemcpyDeviceToHost);
        std::vector<float> rs = host(ro, M);
        cudaFree(xo); cudaFree(ro);
        return std::make_pair(std::move(x16), std::move(rs));
    };
    std::pair<std::vector<uint16_t>, std::vector<float>> rm = prep_rows(false), fr = prep_rows(true);
    check_f32(rm.second, fr.second, "row scales: both layouts");

    std::vector<uint16_t> w16dev;
    {
        __half* wd;
        cudaMalloc((void**)&wd, (size_t)N * K * 2);
        uint32_t* tw = dev(w.tw); uint16_t* ts = dev(w.ts); uint16_t* tb = dev(w.tb);
        void* args[] = {(void*)&tw, (void*)&ts, (void*)&tb, (void*)&wd, (void*)&N, (void*)&K};
        cudaLaunchKernel((void*)tf_vdequant, dim3((N * k8 + 255) / 256), dim3(256), args, 0, 0);
        cudaDeviceSynchronize();
        w16dev = host((uint16_t*)wd, (size_t)N * K);
        cudaFree(wd); cudaFree(tw); cudaFree(ts); cudaFree(tb);
    }

    // One variant: launch the named kernel with a split (and the reduce when split > 1); the (M, N) fp32 sums.
    auto run_variant = [&](bool tiled, bool frag, int sk, int mt) -> std::vector<float> {
        const int gper = kg / sk;
        std::vector<float> out((size_t)M * N, -1e30f);
        __half* xg = (__half*)dev(frag ? fr.first : rm.first);
        float* rg = dev(frag ? fr.second : rm.second);
        float* pg = nullptr;
        if (sk > 1) cudaMalloc((void**)&pg, (size_t)sk * M * N * 4);
        void* ob;
        cudaMalloc((void**)&ob, (size_t)M * N * 4);
        const int f32 = 1;
        const int Ml = M, Nl = N, Kl = K, gperl = gper, skl = sk;
        const uint32_t* wp = tiled ? dtw : dw;
        const uint16_t* sp = tiled ? dts : ds;
        const uint16_t* bp = tiled ? dtb : db;
        void* args[] = {(void*)&xg, (void*)&rg, (void*)&wp, (void*)&sp, (void*)&bp,
                        (void*)&ob, (void*)&pg, (void*)&Ml, (void*)&Nl, (void*)&Kl, (void*)&gperl, (void*)&skl, (void*)&f32};
        void* kern = (mt == 1 ? (frag ? (tiled ? (void*)tf_vqmm884_1_1_1 : (void*)tf_vqmm884_1_0_1)
                                      : (tiled ? (void*)tf_vqmm884_1_1_0 : (void*)tf_vqmm884_1_0_0))
                    : (mt == 2 ? (frag ? (tiled ? (void*)tf_vqmm884_2_1_1 : (void*)tf_vqmm884_2_0_1)
                                       : (tiled ? (void*)tf_vqmm884_2_1_0 : (void*)tf_vqmm884_2_0_0))
                               : (frag ? (tiled ? (void*)tf_vqmm884_4_1_1 : (void*)tf_vqmm884_4_0_1)
                                       : (tiled ? (void*)tf_vqmm884_4_1_0 : (void*)tf_vqmm884_4_0_0))));
        dim3 grid((N + 127) / 128, sk, (M + 8 * mt - 1) / (8 * mt));
        cudaError_t err = cudaLaunchKernel(kern, grid, dim3(128), args, 0, 0);
        cudaDeviceSynchronize();
        check(err == cudaSuccess, "variant launches", (long)err);
        if (sk > 1) {
            const long total = (long)M * N;
            const int skl2 = sk, f32r = 1;
            void* rargs[] = {(void*)&pg, (void*)&ob, (void*)&total, (void*)&skl2, (void*)&f32r};
            cudaLaunchKernel((void*)tf_vreduce, dim3((total + 255) / 256), dim3(256), rargs, 0, 0);
        }
        cudaMemcpy(out.data(), ob, (size_t)M * N * 4, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        cudaFree(ob); cudaFree(xg); cudaFree(rg);
        if (sk > 1) cudaFree(pg);
        return out;
    };

    // 12 variants x 2 splits: [0..11] split 1, [12..23] the host plan's split; cell = mt*4 + tiled*2 + frag.
    std::vector<std::vector<float>> results(24);
    for (int pass = 0; pass < 2; ++pass)
        for (int mt = 1; mt <= 4; mt *= 2)
            for (int tiled = 0; tiled < 2; ++tiled)
                for (int frag = 0; frag < 2; ++frag) {
                    const int mti = (mt == 1 ? 0 : mt == 2 ? 1 : 2);   // mt 1,2,4 -> cells 0,4,8
                    const int idx = pass * 12 + mti * 4 + tiled * 2 + frag;
                    results[idx] = run_variant(tiled, frag, pass ? split_k884(N, K) : 1, mt);
                }

    // Row tiles: one row's mma chain is independent of how many rows share its block, so the bits are stable
    // across mt (within the same split pass).
    for (int tiled = 0; tiled < 2; ++tiled)
        for (int frag = 0; frag < 2; ++frag) {
            const int i = tiled * 2 + frag;
            check_f32(results[i], results[4 + i], "mt=2: bits stable across row tiles");
            check_f32(results[i], results[8 + i], "mt=4: bits stable across row tiles");
        }
    // Layouts: fragment vs row-major rows (frag) and Tiled vs stored words (tiled) feed the same mma operands in
    // the same order, so every cell equals its partner's, for every row-tile count.
    for (int mti = 0; mti < 3; ++mti)
        for (int tiled = 0; tiled < 2; ++tiled)
            for (int frag = 0; frag < 2; ++frag) {
                const int i = mti * 4 + tiled * 2 + frag;
                check_f32(results[i], results[i + (frag ? -1 : 1)], "frag: bits equal to the row-major rows");
                check_f32(results[i], results[i + (tiled ? -2 : 2)], "tiled: bits equal to the stored layout");
            }
    // The split changes the fp32 addition tree (slice partials summed in slice order), so the bits across splits
    // are not required to be stable; every variant must land on the double reference of the expanded fp16
    // operands within the fp32 accumulation bound (a sampled sweep of columns).
    {
        std::vector<double> refs;
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; n += 3) {
                double ref = 0.0;
                for (int kk = 0; kk < K; ++kk)
                    ref += (double)h2f(rm.first[(size_t)m * K + kk]) * (double)h2f(w16dev[(size_t)n * K + kk]);
                refs.push_back(ref * (double)rm.second[m]);
            }
        double maxerr = 0.0;
        for (int v = 0; v < 24; ++v) {
            size_t si = 0;
            for (int m = 0; m < M; ++m)
                for (int n = 0; n < N; n += 3)
                    maxerr = std::fmax(maxerr, std::fabs((double)results[v][(size_t)m * N + n] - refs[si++]));
        }
        check(maxerr < 5e-3, "within the fp32 accumulation bound (all 24 variants)", (long)(maxerr * 1e9));
    }
    cudaFree((void*)xbg); cudaFree(dw); cudaFree(ds); cudaFree(db); cudaFree(dtw); cudaFree(dts); cudaFree(dtb);
}

// reduce: slice-order fp32 sums, bit for bit.
static void test_reduce(int sk, int total) {
    printf("reduce (sk=%d)\n", sk);
    std::vector<float> part((size_t)sk * total), out((size_t)total, -1e30f), ref((size_t)total);
    for (int i = 0; i < sk * total; ++i) part[i] = unif() * 10.f;
    for (int i = 0; i < total; ++i) { float a = part[i]; for (int s = 1; s < sk; ++s) a += part[s * total + i]; ref[i] = a; }
    float* pg = dev(part);
    void* ob;
    cudaMalloc((void**)&ob, total * 4);
    const long totall = total;                 // the kernel takes a 64-bit total
    const int skl = sk, f32 = 1;
    void* args[] = {(void*)&pg, (void*)&ob, (void*)&totall, (void*)&skl, (void*)&f32};
    cudaLaunchKernel((void*)tf_vreduce, dim3((total + 255) / 256), dim3(256), args, 0, 0);
    cudaMemcpy(out.data(), ob, total * 4, cudaMemcpyDeviceToHost);
    cudaFree(ob); cudaFree(pg);
    check_f32(out, ref, "reduce bit-exact (slice order)");
}

// unscale: the rows' power-of-two scales, bit for bit, fp32 and bf16 outs.
static void test_unscale(Weight& w, int M) {
    printf("unscale (M=%d)\n", M);
    const int N = w.n;
    std::vector<float> y((size_t)M * N), rs(M), out((size_t)M * N, -1e30f), ref((size_t)M * N);
    for (int i = 0; i < M * N; ++i) y[i] = unif() * 8.f;
    for (int r = 0; r < M; ++r) rs[r] = ldexpf(1.f, (int)(rng() % 11) - 5);
    for (int r = 0; r < M; ++r)
        for (int n = 0; n < N; ++n) ref[r * N + n] = y[r * N + n] * rs[r];
    float* yg = dev(y); float* rsg = dev(rs);
    void* ob;
    cudaMalloc((void**)&ob, (size_t)M * N * 4);
    const long total4 = (long)M * N / 4;
    const long Nl = N; const int f32 = 1;
    void* args[] = {(void*)&yg, (void*)&rsg, (void*)&ob, (void*)&Nl, (void*)&total4, (void*)&f32};
    cudaLaunchKernel((void*)tf_vunscale, dim3((total4 + 255) / 256), dim3(256), args, 0, 0);
    cudaMemcpy(out.data(), ob, (size_t)M * N * 4, cudaMemcpyDeviceToHost);
    cudaFree(ob);
    check_f32(out, ref, "unscale fp32 bit-exact");
    cudaMalloc((void**)&ob, (size_t)M * N * 2);
    const int f32b = 0;
    args[5] = (void*)&f32b;
    cudaLaunchKernel((void*)tf_vunscale, dim3((total4 + 255) / 256), dim3(256), args, 0, 0);
    std::vector<uint16_t> gout((size_t)M * N);
    cudaMemcpy(gout.data(), ob, (size_t)M * N * 2, cudaMemcpyDeviceToHost);
    cudaFree(ob);
    std::vector<uint16_t> rebb((size_t)M * N);
    for (int i = 0; i < M * N; ++i) rebb[i] = bf16_rne((double)y[i] * (double)rs[i / N]);
    check_fp16(gout, rebb, "unscale bf16 bit-exact");
    cudaFree(yg); cudaFree(rsg);
}

// The prompt path: the pinned cuBLASLt fp16 GEMM. Row-count independence is the invariant (one algorithm a shape,
// split K off): y(M) and y(2M)'s shared rows are bit for bit, and the sums land on the double reference.
static void test_prompt(Weight& w, int M) {
    printf("prompt cuBLASLt (M=%d, N=%d, K=%d)\n", M, w.n, w.k);
    const int N = w.n, K = w.k;
    std::vector<uint16_t> x16((size_t)M * K);
    for (int i = 0; i < M * K; ++i) x16[i] = fp16_rne(unif());
    cublasLtHandle_t handle;
    cublasLtMatmulDesc_t op;
    cublasLtMatrixLayout_t wl;
    check(cublasLtCreate(&handle) == CUBLAS_STATUS_SUCCESS, "cublasLt handle", 0);
    check(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F) == CUBLAS_STATUS_SUCCESS, "desc", 0);
    const cublasOperation_t t = CUBLAS_OP_T;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &t, sizeof(t));
    check(cublasLtMatrixLayoutCreate(&wl, CUDA_R_16F, K, N, K) == CUBLAS_STATUS_SUCCESS, "weight layout", 0);
    // The algorithm pinned per shape: the first heuristic that already runs without split K, then split K off.
    cublasLtMatmulAlgo_t algo;
    {
        cublasLtMatrixLayout_t xr, yr;
        cublasLtMatmulPreference_t pref;
        cublasLtMatmulPreferenceCreate(&pref);
        size_t ws = 0;
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
        check(cublasLtMatrixLayoutCreate(&xr, CUDA_R_16F, K, 4096, K) == CUBLAS_STATUS_SUCCESS, "plan rows", 0);
        check(cublasLtMatrixLayoutCreate(&yr, CUDA_R_32F, N, 4096, N) == CUBLAS_STATUS_SUCCESS, "plan sums", 0);
        cublasLtMatmulHeuristicResult_t found[8];
        int count = 0;
        check(cublasLtMatmulAlgoGetHeuristic(handle, op, wl, xr, yr, yr, pref, 8, found, &count) == CUBLAS_STATUS_SUCCESS && count > 0,
              "heuristic offers an algorithm", count);
        int pick = 0;
        for (int i = 0; i < count; ++i) {
            uint32_t splits = 0; size_t written = 0;
            cublasLtMatmulAlgoConfigGetAttribute(&found[i].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splits, sizeof(splits), &written);
            if (splits <= 1) { pick = i; break; }
        }
        algo = found[pick].algo;
        const uint32_t one = 1;
        const uint32_t none = CUBLASLT_REDUCTION_SCHEME_NONE;
        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &one, sizeof(one));
        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &none, sizeof(none));
        cublasLtMatrixLayoutDestroy(xr);
        cublasLtMatrixLayoutDestroy(yr);
        cublasLtMatmulPreferenceDestroy(pref);
    }
    auto run = [&](int rows) -> std::vector<float> {
        std::vector<float> y((size_t)rows * N);
        cublasLtMatrixLayout_t xl, yl;
        check(cublasLtMatrixLayoutCreate(&xl, CUDA_R_16F, K, rows, K) == CUBLAS_STATUS_SUCCESS, "rows layout", 0);
        check(cublasLtMatrixLayoutCreate(&yl, CUDA_R_32F, N, rows, N) == CUBLAS_STATUS_SUCCESS, "sums layout", 0);
        std::vector<uint16_t> xsub(x16.begin(), x16.begin() + (long)rows * K);
        uint16_t* xg = dev(xsub);
        uint16_t* wg = dev(w.w16);
        float* yg;
        cudaMalloc((void**)&yg, (size_t)rows * N * 4);
        const float alpha = 1.f, beta = 0.f;
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        cublasStatus_t st = cublasLtMatmul(handle, op, &alpha, wg, wl, xg, xl, &beta, yg, yl, yg, yl, &algo, nullptr, 0, stream);
        cudaStreamDestroy(stream);
        check(st == CUBLAS_STATUS_SUCCESS, "matmul", (long)st);
        cudaMemcpy(y.data(), yg, (size_t)rows * N * 4, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        cudaFree(yg); cudaFree(xg); cudaFree(wg);
        cublasLtMatrixLayoutDestroy(xl);
        cublasLtMatrixLayoutDestroy(yl);
        return y;
    };
    std::vector<float> y1 = run(M), y2 = run(2 * M);
    check_f32(y1, std::vector<float>(y2.begin(), y2.begin() + M * N), "row-count independence (bits)");
    double maxerr = 0.0;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; n += 5) {
            double ref = 0.0;
            for (int kk = 0; kk < K; ++kk) ref += (double)h2f(x16[(size_t)m * K + kk]) * (double)h2f(w.w16[(size_t)n * K + kk]);
            maxerr = std::fmax(maxerr, std::fabs((double)y1[m * N + n] - ref));
        }
    check(maxerr < 5e-3, "within the fp32 accumulation bound", (long)(maxerr * 1e9));
    cublasLtMatmulDescDestroy(op);
    cublasLtMatrixLayoutDestroy(wl);
    cublasLtDestroy(handle);
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    int devCount = 0;
    cudaGetDeviceCount(&devCount);
    if (devCount < 1) { printf("no CUDA device\n"); return 1; }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    fprintf(stderr, "== prep ==\n");
    test_prep();
    Weight w;
    w.make(48, 4096);      // N not a multiple of 32: the Tiled padding; K/64 = 64 groups -> sk 16
    fprintf(stderr, "== dequant ==\n");
    test_dequant(w);
    fprintf(stderr, "== qmm M=3 ==\n");
    test_qmm(w, 3);        // mt = 1: the padded-row and wide-split paths
    fprintf(stderr, "== qmm M=9 ==\n");
    test_qmm(w, 9);        // mt = 2
    fprintf(stderr, "== qmm M=17 ==\n");
    test_qmm(w, 17);       // mt = 4
    Weight w2;
    w2.make(128, 1024);    // sk = 4
    fprintf(stderr, "== qmm w2 M=5 ==\n");
    test_qmm(w2, 5);
    fprintf(stderr, "== reduce ==\n");
    test_reduce(4, 3000);
    fprintf(stderr, "== unscale ==\n");
    test_unscale(w, 6);
    fprintf(stderr, "== prompt ==\n");
    test_prompt(w, 100);
    printf(failures ? "FAIL: %d check(s)\n" : "all checks passed\n", failures);
    return failures ? 1 : 0;
}
