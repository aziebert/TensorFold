// Standalone verification of attention_volta.cu on the Volta GPU (Quadro GV100): the engine's fatbin kernels are
// linked straight in (extern "C" symbols) and driven through cudaLaunchKernel, with CPU references.
//
// Each chunk partial (unnormalized o, running max m, sum l) is checked against a double reference of the exact bf16
// operands with a tolerance: the scores come from fp16 mma chains and the P.V product from fp16 mma chains, so the
// accumulation is hardware-chosen. The one closed invariant is structural: a row's shared-chunk partial depends only
// on its own query row and the chunk's keys, so grouping the window's rows into warp groups (warps) and blocks
// (tiles) never changes the bits. That is checked exactly, for the shared kernel, across two groupings; the tail
// kernel is one row a block and has no grouping axis.
//
// Data: one stream, three window rows with speculative depths 400/300/200, committed keys p=700. Chunk 0 is fully
// below p (shared kernel, all cache); chunk 1 mixes cache keys [512,700) with path keys; chunk 2 is path-only, and
// is empty for the two shorter rows (the -inf init store). G=4 gives four query heads per key head.
//
// Build: nvcc -arch=sm_70 -O3 -std=c++20 -o volta_attn_test volta_attn_test.cpp attention_volta.cu

#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
__global__ void tf_vattn_shared_128(const unsigned short*, const unsigned char*, const long*, const int*, float*, float*, float*, int, int, int, int, int, int, float);
__global__ void tf_vattn_shared_256(const unsigned short*, const unsigned char*, const long*, const int*, float*, float*, float*, int, int, int, int, int, int, float);
__global__ void tf_vattn_tail_128(const unsigned short*, const unsigned char*, const unsigned char*, const unsigned char*, const long*, const int*, const int*, const int*, const int*, float*, float*, float*, int, int, int, int, int, float);
__global__ void tf_vattn_tail_256(const unsigned short*, const unsigned char*, const unsigned char*, const unsigned char*, const long*, const int*, const int*, const int*, const int*, float*, float*, float*, int, int, int, int, int, float);
long tf_vattn_smem_shared(int D, int warps);
long tf_vattn_smem_tail(int D);
}

// ---- deterministic data (same helpers as volta_qmm_test.cpp) ---------------------------------------------------------

static uint32_t seed = 0x9E3779B9u;
static uint32_t rng() { seed = seed * 1664525u + 1013904223u; return seed; }
static double unif() { return (double)(rng() >> 8) / 16777216.0 * 2.0 - 1.0; }

static uint16_t bf16_rne(double v) {
    const bool neg = std::signbit(v);
    v = std::fabs(v);
    if (std::isnan(v)) return (neg ? 0x8000u : 0u) | 0x7FC0u;
    if (v == 0.0) return neg ? 0x8000u : 0u;
    if (v >= 3.3895313892515355e38) return (neg ? 0x8000u : 0u) | 0x7F80u;
    int e;
    const double m = std::frexp(v, &e);
    const double mm = (2.0 * m - 1.0) * 128.0;             // 7-bit fraction, RNE
    uint32_t i = (uint32_t)std::floor(mm);
    const double f = mm - i;
    if (f > 0.5 || (f == 0.5 && (i & 1))) i += 1;
    int exp = e - 1;
    if (i == 128) { i = 0; exp += 1; }
    if (exp > 127) return (neg ? 0x8000u : 0u) | 0x7F80u;
    return (neg ? 0x8000u : 0u) | (uint16_t)(((uint32_t)(exp + 127) << 7) | i);
}
static float bf2f(uint16_t v) { uint32_t u = (uint32_t)v << 16; float f; memcpy(&f, &u, 4); return f; }

// ---- check bookkeeping ----------------------------------------------------------------------------------------------

static int failures = 0;
static void check(bool ok, const char* what, long extra) {
    if (ok) { printf("  ok   %s\n", what); return; }
    failures++;
    printf("  FAIL %s (extra=%ld)\n", what, extra);
}
static int same_f32_bits(const std::vector<float>& a, const std::vector<float>& b, long& first) {
    first = -1;
    if (a.size() != b.size()) return 0;
    for (size_t i = 0; i < a.size(); ++i) {
        uint32_t x, y; memcpy(&x, &a[i], 4); memcpy(&y, &b[i], 4);
        if (x != y) { first = (long)i; return 0; }
    }
    return 1;
}
// Tolerance against a double reference; reports the worst absolute error.
static int near_ref(const std::vector<float>& got, const std::vector<double>& ref, double tol, long& first, double& worst) {
    first = -1; worst = 0.0;
    if (got.size() != ref.size()) return 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double d = std::fabs((double)got[i] - ref[i]);
        if (d > worst) worst = d;
        if (d > tol) { if (first < 0) first = (long)i; return 0; }
    }
    return 1;
}

// ---- the test ---------------------------------------------------------------------------------------------------------

static void run_attn(int D) {
    printf("attn D=%d\n", D);
    const int CH = 512;
    const int W = 3, H = 8, HK = 2, G = 4, S = 1;
    const int p = 700, nch = 3, maxd = 400;
    const int depth[3] = {400, 300, 200};
    const float scale = 1.0f / std::sqrt((double)D);
    const int pairs = W * G;
    const int warpsA = 12, tilesA = 1;                          // one block for all 12 (row, head) pairs
    const int warpsB = 1,  tilesB = (pairs + warpsB * 8 - 1) / (warpsB * 8);   // one warp group a block

    // Host data: bf16 bits for the device, exact double values for the reference.
    std::vector<uint16_t> Qb((size_t)W * H * D);
    std::vector<double> qd((size_t)W * H * D);
    for (size_t i = 0; i < Qb.size(); ++i) { Qb[i] = bf16_rne(unif()); qd[i] = bf2f(Qb[i]); }
    const size_t cacheE = (size_t)p * HK * D;
    std::vector<uint16_t> Kb(cacheE), Vb(cacheE);
    std::vector<double> kd(cacheE), vd(cacheE);
    for (size_t i = 0; i < Kb.size(); ++i) { Kb[i] = bf16_rne(unif()); kd[i] = bf2f(Kb[i]); }
    for (size_t i = 0; i < Vb.size(); ++i) { Vb[i] = bf16_rne(unif()); vd[i] = bf2f(Vb[i]); }
    const size_t winE = (size_t)maxd * HK * D;
    std::vector<uint16_t> KNb(winE), VNb(winE);
    std::vector<double> knd(winE), vnd(winE);
    for (size_t i = 0; i < winE; ++i) { KNb[i] = bf16_rne(unif()); knd[i] = bf2f(KNb[i]); }
    for (size_t i = 0; i < winE; ++i) { VNb[i] = bf16_rne(unif()); vnd[i] = bf2f(VNb[i]); }
    std::vector<int> ROWS(W, 0), PATHS((size_t)W * maxd), DEPTHS(W);
    for (int n = 0; n < W; ++n) { DEPTHS[n] = depth[n]; for (int j = 0; j < maxd; ++j) PATHS[n * maxd + j] = j; }
    std::vector<long> OFF = {0, (long)cacheE};
    std::vector<int> STREAM = {0, W, p, nch};

    // Double reference: for every (chunk, row, head), the keys the kernel sees in that chunk, scores scale * q.k,
    // online-softmax partials (m the running max, l and o re-based onto m).
    std::vector<double> ro((size_t)nch * W * H * D, 0.0), rm((size_t)nch * W * H, -INFINITY), rl((size_t)nch * W * H, 0.0);
    for (int c = 0; c < nch; ++c)
        for (int n = 0; n < W; ++n)
            for (int h = 0; h < H; ++h) {
                const int hk = h / G;
                const int lo = c * CH, hi = c < p / CH ? (c + 1) * CH : std::min((c + 1) * CH, p + depth[n]);
                double m = -INFINITY, l = 0.0;
                std::vector<double> o(D, 0.0);
                for (int i = lo; i < hi; ++i) {
                    const size_t ki = i < p ? (size_t)i * HK * D : (size_t)(i - p) * HK * D;
                    const double* krow = (i < p ? kd.data() : knd.data()) + ki + hk * D;
                    double s = 0.0;
                    for (int d = 0; d < D; ++d) s += qd[(n * H + h) * D + d] * krow[d];
                    s *= scale;
                    const double mn = std::max(m, s);
                    const double alpha = (m == -INFINITY) ? 1.0 : std::exp(m - mn);
                    const double pnew = std::exp(s - mn);
                    m = mn;
                    l = l * alpha + pnew;
                    const double* vrow = (i < p ? vd.data() : vnd.data()) + ki + hk * D;
                    for (int d = 0; d < D; ++d) o[d] = o[d] * alpha + pnew * vrow[d];
                }
                const size_t bi = ((size_t)c * W + n) * H + h;
                rm[bi] = m;
                rl[bi] = l;
                for (int d = 0; d < D; ++d) ro[bi * D + d] = o[d];
            }

    // Device buffers (void*: this CUDA runtime's cudaMalloc wants void**; the kernel calls convert back).
    void *dQ, *dKN, *dVN, *dBase, *dOFF, *dSTREAM, *dROWS, *dPATHS, *dDEPTHS, *dPO, *dPM, *dPL;
    cudaMalloc(&dQ, Qb.size() * 2);  cudaMemcpy(dQ, Qb.data(), Qb.size() * 2, cudaMemcpyHostToDevice);
    cudaMalloc(&dKN, winE * 2);      cudaMemcpy(dKN, KNb.data(), winE * 2, cudaMemcpyHostToDevice);
    cudaMalloc(&dVN, winE * 2);      cudaMemcpy(dVN, VNb.data(), winE * 2, cudaMemcpyHostToDevice);
    cudaMalloc(&dBase, 2 * cacheE * 2);
    cudaMemcpy(dBase, Kb.data(), cacheE * 2, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)((unsigned char*)dBase + cacheE * 2), Vb.data(), cacheE * 2, cudaMemcpyHostToDevice);
    cudaMalloc(&dOFF, OFF.size() * 8);      cudaMemcpy(dOFF, OFF.data(), OFF.size() * 8, cudaMemcpyHostToDevice);
    cudaMalloc(&dSTREAM, STREAM.size() * 4);cudaMemcpy(dSTREAM, STREAM.data(), STREAM.size() * 4, cudaMemcpyHostToDevice);
    cudaMalloc(&dROWS, ROWS.size() * 4);    cudaMemcpy(dROWS, ROWS.data(), ROWS.size() * 4, cudaMemcpyHostToDevice);
    cudaMalloc(&dPATHS, PATHS.size() * 4);  cudaMemcpy(dPATHS, PATHS.data(), PATHS.size() * 4, cudaMemcpyHostToDevice);
    cudaMalloc(&dDEPTHS, DEPTHS.size() * 4);cudaMemcpy(dDEPTHS, DEPTHS.data(), DEPTHS.size() * 4, cudaMemcpyHostToDevice);
    cudaMalloc(&dPO, (size_t)nch * W * H * D * 4);
    cudaMalloc(&dPM, (size_t)nch * W * H * 4);
    cudaMalloc(&dPL, (size_t)nch * W * H * 4);

    cudaFuncSetAttribute((const void*)tf_vattn_shared_128, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)tf_vattn_smem_shared(128, 12));
    cudaFuncSetAttribute((const void*)tf_vattn_shared_256, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)tf_vattn_smem_shared(256, 12));
    cudaFuncSetAttribute((const void*)tf_vattn_tail_128, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)tf_vattn_smem_tail(128));
    cudaFuncSetAttribute((const void*)tf_vattn_tail_256, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)tf_vattn_smem_tail(256));

    auto run = [&](void* sh, void* ta, int warps, int tiles) {
        std::vector<float> po((size_t)nch * W * H * D), pm((size_t)nch * W * H), pl((size_t)nch * W * H);
        cudaMemset(dPO, 0, po.size() * 4);
        cudaMemset(dPM, 0, pm.size() * 4);
        cudaMemset(dPL, 0, pl.size() * 4);
        {
            void* args[] = { (void*)&dQ, (void*)&dBase, (void*)&dOFF, (void*)&dSTREAM, (void*)&dPO, (void*)&dPM, (void*)&dPL, (void*)&W, (void*)&H, (void*)&HK, (void*)&G, (void*)&tiles, (void*)&warps, (void*)&scale };
            cudaLaunchKernel(sh, dim3(p / CH, HK, S * tiles), dim3(std::max(256, warps * 32)), args, (size_t)tf_vattn_smem_shared(D, warps), 0);
        }
        {
            void* args[] = { (void*)&dQ, (void*)&dKN, (void*)&dVN, (void*)&dBase, (void*)&dOFF, (void*)&dSTREAM, (void*)&dROWS, (void*)&dPATHS, (void*)&dDEPTHS, (void*)&dPO, (void*)&dPM, (void*)&dPL, (void*)&W, (void*)&H, (void*)&HK, (void*)&G, (void*)&maxd, (void*)&scale };
            cudaLaunchKernel(ta, dim3(W, HK, 2), dim3(256), args, (size_t)tf_vattn_smem_tail(D), 0);
        }
        if (cudaDeviceSynchronize() != cudaSuccess) { printf("  FAIL launch D=%d warps=%d: %s\n", D, warps, cudaGetErrorString(cudaGetLastError())); failures++; }
        cudaMemcpy(po.data(), dPO, po.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(pm.data(), dPM, pm.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(pl.data(), dPL, pl.size() * 4, cudaMemcpyDeviceToHost);
        return std::make_tuple(po, pm, pl);
    };

    auto A = run(D == 256 ? (void*)tf_vattn_shared_256 : (void*)tf_vattn_shared_128,
                 D == 256 ? (void*)tf_vattn_tail_256 : (void*)tf_vattn_tail_128, warpsA, tilesA);
    auto B = run(D == 256 ? (void*)tf_vattn_shared_256 : (void*)tf_vattn_shared_128,
                 D == 256 ? (void*)tf_vattn_tail_256 : (void*)tf_vattn_tail_128, warpsB, tilesB);

    // 1. Against the double reference (both groupings must agree with it).
    for (int which = 0; which < 2; ++which) {
        const auto& r = which == 0 ? A : B;
        const char* tag = which == 0 ? "grouping A (12 warps, 1 tile)" : "grouping B (1 warp, 2 tiles)";
        long first = -1;
        double worst = 0.0;
        {
            long f = -1; double w = 0.0;
            bool ok = near_ref(std::get<0>(r), ro, 5e-3, f, w);
            check(ok, (std::string(tag) + " o within 5e-3 of double ref").c_str(), f);
            if (!ok) printf("    o[%ld] got=%.9g ref=%.9g worst=%.3g\n", f, (double)std::get<0>(r)[f], ro[f], w);
        }
        {
            long f = -1; double w = 0.0;
            bool ok = near_ref(std::get<1>(r), rm, 1e-3, f, w);
            check(ok, (std::string(tag) + " m within 1e-3 of double ref").c_str(), f);
            if (!ok) printf("    m[%ld] got=%.9g ref=%.9g worst=%.3g\n", f, (double)std::get<1>(r)[f], rm[f], w);
        }
        {
            long f = -1; double w = 0.0;
            bool ok = near_ref(std::get<2>(r), rl, 5e-3, f, w);
            check(ok, (std::string(tag) + " l within 5e-3 of double ref").c_str(), f);
            if (!ok) printf("    l[%ld] got=%.9g ref=%.9g worst=%.3g\n", f, (double)std::get<2>(r)[f], rl[f], w);
        }
    }

    // 2. The grouping invariant: the same partials bit for bit.
    long first = -1;
    {
        std::vector<float> po = std::get<0>(A), pm = std::get<1>(A), pl = std::get<2>(A);
        std::vector<float> po2 = std::get<0>(B), pm2 = std::get<1>(B), pl2 = std::get<2>(B);
        check(same_f32_bits(po, po2, first), "grouping A vs B: o bit-stable", first);
        check(same_f32_bits(pm, pm2, first), "grouping A vs B: m bit-stable", first);
        check(same_f32_bits(pl, pl2, first), "grouping A vs B: l bit-stable", first);
    }

    cudaFree(dQ); cudaFree(dKN); cudaFree(dVN); cudaFree(dBase);
    cudaFree(dOFF); cudaFree(dSTREAM); cudaFree(dROWS); cudaFree(dPATHS); cudaFree(dDEPTHS);
    cudaFree(dPO); cudaFree(dPM); cudaFree(dPL);
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    if (prop.major != 7 || prop.minor != 0)
        printf("  note: expected a Volta (sm_70) device\n");
    run_attn(128);
    run_attn(256);
    if (failures == 0) { printf("all checks passed\n"); return 0; }
    printf("%d FAILURES\n", failures);
    return 1;
}
