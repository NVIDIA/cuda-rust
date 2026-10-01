// Standalone cuBLASLt comparison; does not exercise cuda-oxide's WGMMA lowering.
// Follows the TN descriptor setup in gemm_sol_final/bench/cublaslt_bench.c.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cublasLt.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA(call) do { auto e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "line %d: %s: %s\n", __LINE__, #call, cudaGetErrorString(e)); exit(1); } } while (0)
#define BLAS(call) do { auto e = (call); if (e != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "line %d: %s: cuBLAS status %d\n", __LINE__, #call, int(e)); exit(1); } } while (0)

// Exactly representable in both formats. Periodic inputs make a full-output
// CPU reference inexpensive; this is not a model-accuracy/quantization study.
float value(int row, int k, bool b) {
    return float(((row % 32) * (b ? 7 : 3) + (k % 32) * (b ? 3 : 5)) % 5 - 2) / 8;
}

float reference_value(int r, int c, int k) {
    float sum = 0;
    for (int j = 0; j < 32; ++j) sum += value(r, j, false) * value(c, j, true);
    return float(__nv_bfloat16(sum * (k / 32)));
}

bool matches(float got, float want) {
    return std::isfinite(got) && std::abs(got - want) <= 0.03125f + 0.01f * std::abs(want);
}

template<class T> void *input(int rows, int k, bool b) {
    std::vector<T> host(size_t(rows) * k);
    for (int r = 0; r < rows; ++r)
        for (int j = 0; j < k; ++j) host[size_t(r) * k + j] = T(value(r, j, b));
    void *device;
    CUDA(cudaMalloc(&device, host.size() * sizeof(T)));
    CUDA(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    return device;
}

struct Gemm {
    cublasLtHandle_t handle;
    cublasLtMatmulDesc_t desc;
    cublasLtMatrixLayout_t a_layout, b_layout, out_layout;
    cublasLtMatmulHeuristicResult_t algo;
    void *a, *b, *workspace;
    __nv_bfloat16 *out;
    float *scale;
    size_t workspace_bytes = 32 * 1024 * 1024;
    int m, n, k;
    const char *label;

    Gemm(cublasLtHandle_t h, int M, int N, int K, bool fp8, int8_t fast)
        : handle(h), m(M), n(N), k(K), label(fp8 ? "FP8" : "BF16") {
        a = fp8 ? input<__nv_fp8_e4m3>(m, k, false) : input<__nv_bfloat16>(m, k, false);
        b = fp8 ? input<__nv_fp8_e4m3>(n, k, true) : input<__nv_bfloat16>(n, k, true);
        CUDA(cudaMalloc(&out, size_t(m) * n * sizeof(*out)));
        CUDA(cudaMemset(out, 0, size_t(m) * n * sizeof(*out)));
        CUDA(cudaMalloc(&workspace, workspace_bytes));
        CUDA(cudaMalloc(&scale, sizeof(float)));
        float one = 1;
        CUDA(cudaMemcpy(scale, &one, sizeof(one), cudaMemcpyHostToDevice));
        BLAS(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        cublasOperation_t trans = CUBLAS_OP_T;
        BLAS(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)));
        if (fp8) {
            BLAS(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scale, sizeof(scale)));
            BLAS(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scale, sizeof(scale)));
            BLAS(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast, sizeof(fast)));
        }
        auto type = fp8 ? CUDA_R_8F_E4M3 : CUDA_R_16BF;
        BLAS(cublasLtMatrixLayoutCreate(&a_layout, type, k, m, k));
        BLAS(cublasLtMatrixLayoutCreate(&b_layout, type, k, n, k));
        BLAS(cublasLtMatrixLayoutCreate(&out_layout, CUDA_R_16BF, m, n, m));
        cublasLtMatmulPreference_t pref;
        BLAS(cublasLtMatmulPreferenceCreate(&pref));
        BLAS(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                               &workspace_bytes, sizeof(workspace_bytes)));
        int count = 0;
        // ponytail: first heuristic only; tune multiple candidates if library selection limits performance.
        BLAS(cublasLtMatmulAlgoGetHeuristic(handle, desc, a_layout, b_layout, out_layout, out_layout,
                                           pref, 1, &algo, &count));
        BLAS(cublasLtMatmulPreferenceDestroy(pref));
        if (count != 1 || algo.state != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "%s: no supported algorithm for %dx%dx%d\n", label, m, n, k);
            exit(1);
        }
    }

    void run() {
        float alpha = 1, beta = 0;
        BLAS(cublasLtMatmul(handle, desc, &alpha, a, a_layout, b, b_layout, &beta,
                           out, out_layout, out, out_layout, &algo.algo, workspace, workspace_bytes, nullptr));
    }

    float time(int repeats) {
        cudaEvent_t start, stop;
        CUDA(cudaEventCreate(&start)); CUDA(cudaEventCreate(&stop));
        CUDA(cudaEventRecord(start));
        for (int i = 0; i < repeats; ++i) run();
        CUDA(cudaEventRecord(stop)); CUDA(cudaEventSynchronize(stop));
        float ms;
        CUDA(cudaEventElapsedTime(&ms, start, stop));
        CUDA(cudaEventDestroy(start)); CUDA(cudaEventDestroy(stop));
        if (!std::isfinite(ms) || ms <= 0) { fprintf(stderr, "invalid timing\n"); exit(1); }
        return ms / repeats;
    }

    void check() {
        float reference[32][32];
        for (int r = 0; r < 32; ++r) for (int c = 0; c < 32; ++c)
            reference[r][c] = reference_value(r, c, k);
        std::vector<__nv_bfloat16> host(size_t(m) * n);
        CUDA(cudaMemcpy(host.data(), out, host.size() * sizeof(*out), cudaMemcpyDeviceToHost));
        float max_error = 0;
        for (int c = 0; c < n; ++c) for (int r = 0; r < m; ++r) {
            float got = float(host[size_t(c) * m + r]), want = reference[r % 32][c % 32];
            if (!matches(got, want)) {
                fprintf(stderr, "%s mismatch (%d,%d): %g != %g\n", label, r, c, got, want);
                exit(1);
            }
            max_error = std::max(max_error, std::abs(got - want));
        }
        printf("%s correctness: %zu outputs, max_abs_error=%g (atol=0.03125, rtol=0.01)\n",
               label, host.size(), max_error);
    }

    ~Gemm() {
        BLAS(cublasLtMatrixLayoutDestroy(a_layout)); BLAS(cublasLtMatrixLayoutDestroy(b_layout));
        BLAS(cublasLtMatrixLayoutDestroy(out_layout)); BLAS(cublasLtMatmulDescDestroy(desc));
        CUDA(cudaFree(a)); CUDA(cudaFree(b)); CUDA(cudaFree(out));
        CUDA(cudaFree(scale)); CUDA(cudaFree(workspace));
    }
};

int dimension(const char *s) {
    char *end;
    long v = strtol(s, &end, 10);
    if (!*s || *end || v < 32 || v > 8192 || v % 32) {
        fprintf(stderr, "dimension must be a multiple of 32 in [32,8192]: %s\n", s); exit(2);
    }
    return int(v);
}

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--self-test")) {
        for (int r = 0; r < 64; ++r) for (int c = 0; c < 96; ++c) {
            float sum = 0;
            for (int j = 0; j < 128; ++j) {
                float a = value(r, j, false), b = value(c, j, true);
                assert(float(__nv_fp8_e4m3(a)) == a && float(__nv_bfloat16(a)) == a);
                assert(float(__nv_fp8_e4m3(b)) == b && float(__nv_bfloat16(b)) == b);
                sum += a * b;
            }
            assert(float(__nv_bfloat16(sum)) == reference_value(r, c, 128));
        }
        assert(matches(1, 1) && !matches(2, 1) && !matches(NAN, 1) && !matches(INFINITY, 1));
        puts("PASS: input encodings, periodic CPU reference, and error checks (no GPU required)");
        return 0;
    }
    if (argc == 2 && !strcmp(argv[1], "--help")) {
        puts("Usage: fp8_vs_bf16 M N K [0|1 fast_accum, default 0]\nRequires H100/H200. Preconverted inputs; BF16 outputs; FP32 compute.");
        return 0;
    }
    if (argc < 4 || argc > 5 || (argc == 5 && strcmp(argv[4], "0") && strcmp(argv[4], "1"))) {
        fprintf(stderr, "Usage: fp8_vs_bf16 M N K [0|1 fast_accum]\n"); return 2;
    }
    int m = dimension(argv[1]), n = dimension(argv[2]), k = dimension(argv[3]);
    int8_t fast = argc == 5 ? argv[4][0] - '0' : 0;
    cudaDeviceProp prop;
    CUDA(cudaGetDeviceProperties(&prop, 0));
    if (prop.major != 9) { fprintf(stderr, "Requires Hopper H100/H200; found %s\n", prop.name); return 2; }
    int runtime, driver;
    CUDA(cudaRuntimeGetVersion(&runtime)); CUDA(cudaDriverGetVersion(&driver));
    printf("GPU=%s runtime=%d driver=%d cuBLASLt=%zu M=%d N=%d K=%d fast_accum=%d\n",
           prop.name, runtime, driver, cublasLtGetVersion(), m, n, k, int(fast));
    puts("Dense GEMM only; conversion/transfers/allocations excluded; scales=1; BF16 output; FP32 compute.");
    cublasLtHandle_t handle;
    BLAS(cublasLtCreate(&handle));
    {
        Gemm bf16(handle, m, n, k, false, 0), fp8(handle, m, n, k, true, fast);
        bf16.run(); fp8.run(); bf16.check(); fp8.check();
        for (int i = 0; i < 10; ++i) {
            if (i % 2) { fp8.run(); bf16.run(); } else { bf16.run(); fp8.run(); }
        }
        CUDA(cudaDeviceSynchronize());
        std::vector<float> bf_ms, fp_ms;
        for (int i = 0; i < 7; ++i) {
            if (i % 2) { fp_ms.push_back(fp8.time(50)); bf_ms.push_back(bf16.time(50)); }
            else { bf_ms.push_back(bf16.time(50)); fp_ms.push_back(fp8.time(50)); }
            printf("sample=%d BF16_ms=%.6f FP8_ms=%.6f\n", i, bf_ms.back(), fp_ms.back());
        }
        bf16.check(); fp8.check();
        std::sort(bf_ms.begin(), bf_ms.end()); std::sort(fp_ms.begin(), fp_ms.end());
        double ops = 2.0 * m * n * k;
        printf("RESULT M=%d N=%d K=%d fast_accum=%d BF16_ms=%.6f FP8_ms=%.6f BF16_TFLOPS=%.2f FP8_TFLOPS=%.2f speedup=%.3fx\n",
               m, n, k, int(fast), bf_ms[3], fp_ms[3], ops / (bf_ms[3] * 1e9), ops / (fp_ms[3] * 1e9), bf_ms[3] / fp_ms[3]);
    }
    BLAS(cublasLtDestroy(handle));
    puts("SUCCESS: both formats passed before and after timing (speedup is measured, not required)");
}
