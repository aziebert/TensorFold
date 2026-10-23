// Validation of the 3 NEW sm_70 engines on the GV100: qmmf_volta, experts_volta, draft_attention_volta.
// The engines' fatbin symbols are linked straight in (extern "C") and driven through cudaLaunchKernel, with CPU
// references. Build:
//   nvcc -arch=sm_70 -O3 -std=c++20 -I<repo>/zig -o volta_new_test volta_new_test.cpp \
//        kernels/cuda/qmmf_volta.cu kernels/cuda/experts_volta.cu kernels/cuda/draft_attention_volta.cu -lcudart
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

static int failures = 0;
static void check(bool ok, const char* what, long extra) {
    if (ok) { printf("  ok   %s\n", what); return; }
    failures++;
    printf("  FAIL %s (extra=%ld)\n", what, extra);
}

template <class T>
static T* dev(const std::vector<T>& v) {
    T* d;
    cudaError_t e1 = cudaMalloc((void**)&d, v.size() * sizeof(T));
    cudaError_t e2 = cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    if (e1 != cudaSuccess || e2 != cudaSuccess)
        fprintf(stderr, "dev: n=%zu malloc=%d copy=%d\n", v.size(), (int)e1, (int)e2);
    return d;
}
template <class T>
static std::vector<T> host(T* p, size_t n) {
    std::vector<T> v(n);
    cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    return v;
}

// ---- the image's symbols, resolved by name as the runtime's Module.function does ----------------------------------
extern "C" {
__global__ void tf_vqmmf884s_2_1(const __half* X, const float* RS, const int* W, const unsigned* S, const float* ALPHA, void* OUT, float* PART, int* CNT, int M, int N, int K, int gper, int sk, bool f32);
__global__ void tf_vqmmf884s_reduce(const float* part, void* out, long total, int sk, bool f32);
__global__ void tf_vqmmf884s_dequant_0(const int* W, const unsigned* S, __half* out, int N, int K);
__global__ void tf_vqmmf884s_dequant_1(const int* W, const unsigned* S, __half* out, int N, int K);
__global__ void tf_vqmmf884s_dequant_2(const int* W, const unsigned* S, __half* out, int N, int K);
__global__ void tf_vqmmf884s_unscale2(const float* y, const float* rs, const float* alpha, void* out, long N, long total, bool f32);
__global__ void tf_vqmmf884s_gather_rows(const long* ids, const uint4* table, uint4* out, int row16);
__global__ void tf_vexp_32_1_0(const __half* X, const float* RS, int slots, const unsigned* W, int KG, int NB, const int* items, const int* counts, const int* members, void* out, int N, float limit);
__global__ void tf_vdraft_split_128(const unsigned short* q, const unsigned short* kb, const unsigned short* vb, const long* table, const long* lens, float* pm, float* pl, float* po, int R, int L, int G, int window, int nsplit, int causal, float scale);
__global__ void tf_vdraft_fold(const float* pm, const float* pl, const float* po, unsigned short* out, int L, int G, int HKV, int D, int nsplit);
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    int devCount = 0;
    cudaGetDeviceCount(&devCount);
    if (devCount < 1) { printf("no CUDA device\n"); return 1; }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("device: %s (sm_%d%d, %d SMs)\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    // ======================================================================
    // QMMF F16 matmul (tf_vqmmf884s_2_1): all-ones A and B -> every cell = K*1*1*rs*alpha = 64.
    // The fp16 mma is exact (fp32 accum), and the all-ones input is robust to the fragment/tile permutation.
    // K=64, M=8, N=128, sk=1, mt=1, f32. grid=(ceil(N/128)=1, sk=1, ceil(M/8)=1), block=128.
    // ======================================================================
    {
        printf("qmmf F16 matmul\n");
        const int K = 64, M = 8, N = 128, KG = K / 64, WPG = 32;
        std::vector<__half> Xh(M * K, __ushort_as_half(0x3C00));        // 1.0 in fp16
        std::vector<int> Wh(N * KG * WPG, 0x3C003C00);                 // every 16-bit half = 1.0
        std::vector<unsigned> Sh(1, 0);                                // unused for F16
        std::vector<float> RSh(M, 1.f), ALPha(N, 1.f);
        const __half* dX = dev(Xh);
        const int* dW = dev(Wh);
        const unsigned* dS = dev(Sh);
        const float* dRS = dev(RSh);
        const float* dA = dev(ALPha);
        float* dO;
        cudaMalloc((void**)&dO, (size_t)M * N * 4);
        cudaMemset(dO, 0xEE, (size_t)M * N * 4);
        float* PART = nullptr; int* CNT = nullptr;
        const int gper = K / 64, sk = 1;
        bool f32 = true;
        void* args[] = {(void*)&dX, (void*)&dRS, (void*)&dW, (void*)&dS, (void*)&dA, (void*)&dO,
                        (void*)&PART, (void*)&CNT, (void*)&M, (void*)&N, (void*)&K, (void*)&gper, (void*)&sk, (void*)&f32};
        cudaError_t err = cudaLaunchKernel((void*)tf_vqmmf884s_2_1, dim3((N + 127) / 128, sk, (M + 7) / 8), dim3(128), args, 0, 0);
        cudaDeviceSynchronize();
        check(err == cudaSuccess, "qmmf F16 matmul launches", (long)err);
        std::vector<float> OOut = host(dO, (size_t)M * N);
        float mn = 1e30f, mx = -1e30f; bool all64 = true;
        for (float v : OOut) { mn = fminf(mn, v); mx = fmaxf(mx, v); if (fabsf(v - 64.f) > 1e-2f) all64 = false; }
        check(all64, "every cell == 64 (min/max)", (long)(mx * 10000));
        printf("        min=%.5f max=%.5f (expect 64)\n", mn, mx);
        cudaFree(dO); cudaFree((void*)dX); cudaFree((void*)dW); cudaFree((void*)dS); cudaFree((void*)dRS); cudaFree((void*)dA);
    }

    // ======================================================================
    // QMMF FP4 (NVFP4, fmt=0) dequant: the production 4-bit path.
    // ======================================================================
    {
        printf("qmmf FP4 dequant\n");
        // e2m1 values (code 0..15): used for zero/sign/ratio checks (scale-invariant).
        static const float e2m1[16] = {0, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
        // Exact CPU replication of the GPU fp4x8 bit manipulation (same ops, same order).
        auto fp4x8_cpu = [](unsigned q, float scale, float out[8]) {
            const unsigned S = 0x80008000u, EM = 0x0E000E00u;
            unsigned v0 = ((q << 12) & S) | ((q << 9) & EM);
            unsigned v1 = ((q << 8) & S) | ((q << 5) & EM);
            unsigned v2 = ((q << 4) & S) | ((q << 1) & EM);
            unsigned v3 = (q & S) | ((q >> 3) & EM);
            auto hf = [](unsigned short u) { return __half2float(__ushort_as_half(u)); };
            out[0] = hf(v0 & 0xFFFF) * scale; out[1] = hf((v0 >> 16) & 0xFFFF) * scale;
            out[2] = hf(v1 & 0xFFFF) * scale; out[3] = hf((v1 >> 16) & 0xFFFF) * scale;
            out[4] = hf(v2 & 0xFFFF) * scale; out[5] = hf((v2 >> 16) & 0xFFFF) * scale;
            out[6] = hf(v3 & 0xFFFF) * scale; out[7] = hf((v3 >> 16) & 0xFFFF) * scale;
        };
        auto e4m3_val = [](unsigned b) {
            unsigned short hb = (unsigned short)(((b & 0x80u) << 8) | ((b & 0x7Fu) << 7));
            return __half2float(__ushort_as_half(hb)) * 256.f;
        };
        auto rep8 = [](int c) { return (unsigned)(c & 0xF) * 0x11111111u; };
        bool ok = true;
        // (a) e2m1 correctness (CPU side, scale=1): all-8-equal, zero code, sign, ratios vs code 2 (==1.0).
        float vref[8]; fp4x8_cpu(rep8(2), 1.f, vref); float v2 = vref[0];
        for (int c = 0; c < 16; ++c) {
            float vv[8]; fp4x8_cpu(rep8(c), 1.f, vv); float val = vv[0];
            for (int k = 1; k < 8; ++k) if (vv[k] != val) ok = false;   // all nibbles equal -> all outputs equal
            if (c == 0 || c == 8) { if (std::fabs(val) != 0.f) ok = false; }   // zero codes -> +/-0
            else if (c != 2) {
                if (val == 0.f) ok = false;                              // non-zero code -> non-zero
                if ((val < 0.f) != (e2m1[c] < 0.f)) ok = false;         // sign matches e2m1
                if (std::fabs(val / v2 - e2m1[c] / e2m1[2]) > 1e-6) ok = false;   // ratio vs code 2
            }
        }
        check(ok, "fp4 e2m1 dequant (16 codes: zero, sign, ratios)", 0);
        // (b) GPU dequant_0 matches the CPU replication, with a real e4m3 scale (0x40 == 0.125).
        int N = 8, K = 64; int k8 = K / 8, kg = K / 64;
        std::vector<int> W((size_t)N * kg * 8);
        std::vector<unsigned> S((size_t)N * kg, 0x40404040u);
        for (int n = 0; n < N; ++n) for (int wi = 0; wi < 8; ++wi) W[n * 8 + wi] = (int)rep8((n + wi) & 15);
        const int *dW = dev(W); const unsigned *dS = dev(S);
        std::vector<__half> out((size_t)N * K); __half *dOut = dev(out);
        dim3 db(256); dim3 dg((N * k8 + 255) / 256);
        void* dargs[] = { (void*)&dW, (void*)&dS, (void*)&dOut, (void*)&N, (void*)&K };
        cudaError_t e = cudaLaunchKernel((const void*)tf_vqmmf884s_dequant_0, db, dg, dargs, 0, 0);
        if (e != cudaSuccess) check(false, "dequant_0 launch", (long)e);
        else {
            std::vector<__half> o = host(dOut, (size_t)N * K);
            bool good = true;
            float scale = e4m3_val(0x40); float want[8];
            for (int n = 0; n < N && good; ++n) for (int wi = 0; wi < 8 && good; ++wi) {
                fp4x8_cpu((unsigned)W[n * 8 + wi], scale, want);
                for (int k = 0; k < 8; ++k) {
                    float got = __half2float(o[(size_t)n * K + wi * 8 + k]);
                    if (got != want[k] && std::fabs(got - want[k]) > 1e-7) { good = false; break; }
                }
            }
            check(good, "dequant_0 GPU==CPU (N=8 K=64, e4m3 scale 0.125)", 0);
        }
    }

    // ======================================================================
    // QMMF reduce: PART[0]=1.0, PART[1]=2.0 -> out = 3.0 (slice-order fp32 sums).
    // ======================================================================
    {
        printf("qmmf reduce\n");
        const int sk = 2, total = 1024;
        std::vector<float> part((size_t)sk * total, 1.f);
        for (int i = 0; i < total; ++i) part[total + i] = 2.f;
        std::vector<float> ref((size_t)total, 3.f);
        float* pg = dev(part);
        void* ob; cudaMalloc((void**)&ob, (size_t)total * 4);
        cudaMemset(ob, 0xEE, (size_t)total * 4);
        const long totall = total; bool f32 = true;
        void* args[] = {(void*)&pg, (void*)&ob, (void*)&totall, (void*)&sk, (void*)&f32};
        cudaError_t err = cudaLaunchKernel((void*)tf_vqmmf884s_reduce, dim3((total + 255) / 256), dim3(256), args, 0, 0);
        cudaDeviceSynchronize();
        check(err == cudaSuccess, "reduce launches", (long)err);
        std::vector<float> out = host((float*)ob, total);
        bool ok = true; for (float v : out) if (fabsf(v - 3.f) > 1e-6f) ok = false;
        check(ok, "out == 3 (slice order)", (long)(out[0] * 1000000));
        cudaFree(ob); cudaFree(pg);
    }

    // ======================================================================
    // QMMF dequant_2 (F16): raw fp16 weight copy. W all 1.0 -> out all 1.0.
    // ======================================================================
    {
        printf("qmmf dequant F16\n");
        const int N = 128, K = 64;
        std::vector<int> Wh(N * (K / 64) * 32, 0x3C003C00);
        std::vector<unsigned> Sh(1, 0);
        std::vector<__half> ref(N * K, __ushort_as_half(0x3C00));
        const int* dW = dev(Wh); const unsigned* dS = dev(Sh);
        __half* do_; cudaMalloc((void**)&do_, (size_t)N * K * 2);
        cudaMemset(do_, 0xEE, (size_t)N * K * 2);
        void* args[] = {(void*)&dW, (void*)&dS, (void*)&do_, (void*)&N, (void*)&K};
        cudaError_t err = cudaLaunchKernel((void*)tf_vqmmf884s_dequant_2, dim3((N * K / 8 + 255) / 256), dim3(256), args, 0, 0);
        cudaDeviceSynchronize();
        check(err == cudaSuccess, "dequant launches", (long)err);
        std::vector<__half> out = host(do_, (size_t)N * K);
        bool ok = true; for (int i = 0; i < N * K; ++i) if (__half2float(out[i]) != 1.f) ok = false;
        check(ok, "out all 1.0 (raw copy)", (long)(__half2float(out[0]) * 10000));
        cudaFree(do_); cudaFree((void*)dW); cudaFree((void*)dS);
    }

    // ======================================================================
    // QMMF unscale2: out[m*N+n] = y[m*N+n] * rs[m] * alpha[n].
    // ======================================================================
    {
        printf("qmmf unscale2\n");
        const int M = 4, N = 3, total = M * N;
        std::vector<float> y(total, 1.f), rs(M), alpha(N), ref(total);
        for (int m = 0; m < M; ++m) rs[m] = m + 1.f;
        for (int n = 0; n < N; ++n) alpha[n] = (n + 1) * 10.f;
        for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) ref[m * N + n] = rs[m] * alpha[n];
        float* dy = dev(y); float* drs = dev(rs); float* da = dev(alpha);
        void* ob; cudaMalloc((void**)&ob, (size_t)total * 4);
        cudaMemset(ob, 0xEE, (size_t)total * 4);
        const long Nl = N, totall = total; bool f32 = true;
        void* args[] = {(void*)&dy, (void*)&drs, (void*)&da, (void*)&ob, (void*)&Nl, (void*)&totall, (void*)&f32};
        cudaError_t err = cudaLaunchKernel((void*)tf_vqmmf884s_unscale2, dim3((total + 255) / 256), dim3(256), args, 0, 0);
        cudaDeviceSynchronize();
        check(err == cudaSuccess, "unscale2 launches", (long)err);
        std::vector<float> out = host((float*)ob, total);
        bool ok = true; for (int i = 0; i < total; ++i) if (fabsf(out[i] - ref[i]) > 1e-4f) ok = false;
        check(ok, "out == y*rs*alpha", (long)(out[0] * 10000));
        cudaFree(ob); cudaFree(dy); cudaFree(drs); cudaFree(da);
    }

    // ======================================================================
    // QMMF gather_rows: out[row] = table[ids[row]] (identity gather).
    // ======================================================================
    {
        printf("qmmf gather_rows\n");
        const int rows = 4, row16 = 8;
        std::vector<long> ids(rows); for (int r = 0; r < rows; ++r) ids[r] = r;
        std::vector<uint4> table((size_t)rows * row16), ref((size_t)rows * row16);
        for (int r = 0; r < rows; ++r) for (int c = 0; c < row16; ++c) table[(size_t)r * row16 + c] = make_uint4(r * 100 + c, 1, 2, 3);
        for (int r = 0; r < rows; ++r) for (int c = 0; c < row16; ++c) ref[(size_t)r * row16 + c] = table[(size_t)ids[r] * row16 + c];
        long* did = dev(ids); uint4* dt = dev(table);
        uint4* do_; cudaMalloc((void**)&do_, (size_t)rows * row16 * 16);
        cudaMemset(do_, 0, (size_t)rows * row16 * 16);
        void* args[] = {(void*)&did, (void*)&dt, (void*)&do_, (void*)&row16};
        cudaError_t err = cudaLaunchKernel((void*)tf_vqmmf884s_gather_rows, dim3(rows), dim3(128), args, 0, 0);
        cudaDeviceSynchronize();
        check(err == cudaSuccess, "gather_rows launches", (long)err);
        std::vector<uint4> out = host(do_, (size_t)rows * row16);
        bool ok = true; for (int i = 0; i < rows * row16; ++i) if (memcmp(&out[i], &ref[i], 16) != 0) ok = false;
        check(ok, "out[row] == table[ids[row]]", 0);
        cudaFree(do_); cudaFree(did); cudaFree((void*)dt);
    }

    // ======================================================================
    // EXPERTS (tf_vexp_32_1_0): one expert, one block, 8 pairs, all-zero 4-bit weights -> out all 0.0.
    // GS=32, M=1, EPI=0. grid=1, block=128. Exercises the full routing + dequant + mma path.
    // ======================================================================
    {
        printf("experts\n");
        const int GS = 32, KG = 2, NB = 1, M = 1, N = 32, pairs = 8;
        const int K = KG * GS, H = GS / 32, WORDS = 128 * H, BLOCK = WORDS + 32;
        std::vector<__half> Xh((size_t)pairs * K, __ushort_as_half(0x3C00));     // 1.0
        std::vector<float> RSh(pairs, 1.f);
        std::vector<unsigned> Wh((size_t)NB * KG * M * BLOCK, 0u);              // all-zero weights
        std::vector<int> items(3, 0); items[2] = pairs;                          // e=0, first=0, cnt=8
        std::vector<int> counts(1, 1);
        std::vector<int> members(pairs); for (int i = 0; i < pairs; ++i) members[i] = i;
        std::vector<float> ref((size_t)pairs * N, 0.f);
        const __half* dX = dev(Xh); const float* dRS = dev(RSh); const unsigned* dW = dev(Wh);
        const int* dIt = dev(items); const int* dCt = dev(counts); const int* dMm = dev(members);
        void* dOut; cudaMalloc((void**)&dOut, (size_t)pairs * N * 4);
        cudaMemset(dOut, 0xEE, (size_t)pairs * N * 4);
        const int slots = 0, limit0 = 0;
        float limit = 0.f;
        void* args[] = {(void*)&dX, (void*)&dRS, (void*)&slots, (void*)&dW, (void*)&KG, (void*)&NB,
                        (void*)&dIt, (void*)&dCt, (void*)&dMm, (void*)&dOut, (void*)&N, (void*)&limit};
        cudaError_t err = cudaLaunchKernel((void*)tf_vexp_32_1_0, dim3(1), dim3(128), args, 0, 0);
        cudaDeviceSynchronize();
        check(err == cudaSuccess, "experts launches", (long)err);
        std::vector<float> out = host((float*)dOut, (size_t)pairs * N);
        bool ok = true; for (float v : out) if (fabsf(v) > 1e-6f) ok = false;
        check(ok, "out all 0 (zero weights)", (long)(out[0] * 1000000));
        cudaFree(dOut); cudaFree((void*)dX); cudaFree((void*)dRS); cudaFree((void*)dW);
        cudaFree((void*)dIt); cudaFree((void*)dCt); cudaFree((void*)dMm);
    }

    // ======================================================================
    // DRAFT (tf_vdraft_split_128 + fold): D=128, H=2, HKV=1, G=2, L=16, context=32, nsplit=2.
    // Launch smoke: correct grid/block/smem, no CUDA error, finite output.
    // ======================================================================
    {
        printf("draft\n");
        const int D = 128, H = 2, HKV = 1, G = 2, L = 16, S = 1, R = S * L, rows = G * L;
        const int KC = 32, maxc = 32, nsplit = (maxc + KC - 1) / KC + 1;
        const size_t smem = (size_t)(rows * D + KC * (D + 1) + KC * D + rows * KC) * sizeof(float);
        check(smem <= 96 * 1024, "draft smem under 96KB", (long)smem);
        unsigned short half5 = __bfloat16_as_ushort(__float2bfloat16_rn(0.5f));
        std::vector<unsigned short> qh((size_t)H * R * D, half5), kbh((size_t)HKV * maxc * D, half5), vbh((size_t)HKV * maxc * D, half5);
        std::vector<long> lens(1, maxc);
        const unsigned short* dQ = dev(qh); const unsigned short* dK = dev(kbh); const unsigned short* dV = dev(vbh);
        // TABLE[2j] / TABLE[2j+1] are RAW device addresses (cast to pointers in-kernel): base of stream j's context keys / values.
        std::vector<long> table(2 * S);
        for (int j = 0; j < S; ++j) { table[2 * j] = (long)(uintptr_t)dK; table[2 * j + 1] = (long)(uintptr_t)dV; }
        const long* dTab = dev(table); const long* dLens = dev(lens);
        float* dpm; cudaMalloc((void**)&dpm, (size_t)S * HKV * nsplit * rows * 4);
        float* dpl; cudaMalloc((void**)&dpl, (size_t)S * HKV * nsplit * rows * 4);
        float* dpo; cudaMalloc((void**)&dpo, (size_t)S * HKV * nsplit * rows * D * 4);
        cudaMemset(dpm, 0, (size_t)S * HKV * nsplit * rows * 4);
        cudaMemset(dpl, 0, (size_t)S * HKV * nsplit * rows * 4);
        cudaMemset(dpo, 0, (size_t)S * HKV * nsplit * rows * D * 4);
        unsigned short* dOut; cudaMalloc((void**)&dOut, (size_t)R * H * D * 2);
        cudaMemset(dOut, 0, (size_t)R * H * D * 2);
        float scale = 1.f / sqrtf((float)D);
        int causal_int = 1;
        void* args[] = {(void*)&dQ, (void*)&dK, (void*)&dV, (void*)&dTab, (void*)&dLens, (void*)&dpm, (void*)&dpl, (void*)&dpo,
                        (void*)&R, (void*)&L, (void*)&G, (void*)&maxc, (void*)&nsplit, (void*)&causal_int, (void*)&scale};
        cudaError_t e0 = cudaFuncSetAttribute((const void*)tf_vdraft_split_128, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        check(e0 == cudaSuccess, "split smem attr set", (long)e0);
        cudaError_t err = cudaLaunchKernel((void*)tf_vdraft_split_128, dim3(S, HKV, nsplit), dim3(128), args, smem, 0);
        cudaDeviceSynchronize();
        check(err == cudaSuccess && cudaGetLastError() == cudaSuccess, "split launches", (long)err);
        void* fargs[] = {(void*)&dpm, (void*)&dpl, (void*)&dpo, (void*)&dOut, (void*)&L, (void*)&G, (void*)&HKV, (void*)&D, (void*)&nsplit};
        cudaError_t err2 = cudaLaunchKernel((void*)tf_vdraft_fold, dim3(S, HKV, rows), dim3(128), fargs, 0, 0);
        cudaDeviceSynchronize();
        check(err2 == cudaSuccess && cudaGetLastError() == cudaSuccess, "fold launches", (long)err2);
        std::vector<unsigned short> out = host(dOut, (size_t)R * H * D);
        int nonfin = 0; float sample = 0.f;
        for (unsigned short b : out) { float v = __bfloat162float(__ushort_as_bfloat16(b)); if (!std::isfinite(v)) nonfin++; }
        sample = out.empty() ? 0.f : __bfloat162float(__ushort_as_bfloat16(out[0]));
        check(nonfin == 0, "out finite (smoke)", nonfin);
        printf("        out %d/%d finite, sample=%.4f\n", (int)out.size() - nonfin, (int)out.size(), sample);
        cudaFree(dOut); cudaFree(dpm); cudaFree(dpl); cudaFree(dpo);
        cudaFree((void*)dQ); cudaFree((void*)dK); cudaFree((void*)dV); cudaFree((void*)dTab); cudaFree((void*)dLens);
    }

    printf(failures ? "FAIL: %d check(s)\n" : "all checks passed\n", failures);
    return failures ? 1 : 0;
}
