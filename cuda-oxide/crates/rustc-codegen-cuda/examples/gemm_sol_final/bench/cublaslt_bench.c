/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Top-8 tuning and the NVIDIA LtMatmulCustomFind search grid, without the
 * sample's 16-algorithm / 100-success limits. Split-K is a finite grid, not
 * every integer: 0,1,2,3,4,5,6,8,12,16,32. Search time is excluded from results.
 * https://github.com/NVIDIA/CUDALibrarySamples/tree/main/cuBLASLt
 */
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK_CUDA(call)                                                                           \
    do {                                                                                           \
        cudaError_t e = (call);                                                                    \
        if (e != cudaSuccess) {                                                                    \
            fprintf(stderr, "CUDA: %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));       \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)
#define CHECK_BLAS(call)                                                                           \
    do {                                                                                           \
        cublasStatus_t s = (call);                                                                 \
        if (s != CUBLAS_STATUS_SUCCESS) {                                                          \
            fprintf(stderr, "cuBLAS: %s:%d: %s\n", __FILE__, __LINE__,                             \
                    cublasLtGetStatusString(s));                                                   \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)
#define FINALISTS 16
static const size_t workspace_size = 32 * 1024 * 1024;
static int warmup = 2000, iters = 301, repeats = 5;
static const char *mode = "top8";
static cudaDataType_t output_type = CUDA_R_16F;
static const char *output_name = "fp16";
typedef struct {
    cublasLtHandle_t h;
    cudaStream_t stream;
    cudaEvent_t start, stop;
    cublasLtMatmulDesc_t op;
    cublasLtMatrixLayout_t a, b, c;
    void *A, *B, *C, *workspace;
    int n;
} Problem;
typedef struct {
    cublasLtMatmulAlgo_t algo;
    double us;
    size_t workspace;
} Candidate;
static cublasStatus_t launch(Problem *p, const cublasLtMatmulAlgo_t *a) {
    float alpha = 1, beta = 0;
    return cublasLtMatmul(p->h, p->op, &alpha, p->A, p->a, p->B, p->b, &beta, p->C, p->c, p->C,
                          p->c, a, p->workspace, workspace_size, p->stream);
}
static double measure(Problem *p, const cublasLtMatmulAlgo_t *a, int nw, int ni) {
    for (int i = 0; i < nw; i++)
        if (launch(p, a) != CUBLAS_STATUS_SUCCESS)
            return DBL_MAX;
    CHECK_CUDA(cudaStreamSynchronize(p->stream));
    CHECK_CUDA(cudaEventRecord(p->start, p->stream));
    for (int i = 0; i < ni; i++)
        if (launch(p, a) != CUBLAS_STATUS_SUCCESS)
            return DBL_MAX;
    CHECK_CUDA(cudaEventRecord(p->stop, p->stream));
    CHECK_CUDA(cudaEventSynchronize(p->stop));
    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, p->start, p->stop));
    return ms * 1000.0 / ni;
}
static int compare(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
static double median(double *values, int count) {
    qsort(values, count, sizeof(double), compare);
    return values[count / 2];
}
static int cap(const cublasLtMatmulAlgo_t *a, cublasLtMatmulAlgoCapAttributes_t attr) {
    int v = 0;
    CHECK_BLAS(cublasLtMatmulAlgoCapGetAttribute(a, attr, &v, sizeof(v), NULL));
    return v;
}
static int *cap_list(const cublasLtMatmulAlgo_t *a, cublasLtMatmulAlgoCapAttributes_t attr,
                     int *count) {
    size_t bytes = 0;
    CHECK_BLAS(cublasLtMatmulAlgoCapGetAttribute(a, attr, NULL, 0, &bytes));
    *count = bytes / sizeof(int);
    int *v = calloc(*count ? *count : 1, sizeof(int));
    if (!v)
        exit(1);
    if (*count)
        CHECK_BLAS(cublasLtMatmulAlgoCapGetAttribute(a, attr, v, bytes, NULL));
    else
        *count = 1;
    return v;
}
static int set(cublasLtMatmulAlgo_t *a, cublasLtMatmulAlgoConfigAttributes_t attr, int v) {
    return cublasLtMatmulAlgoConfigSetAttribute(a, attr, &v, sizeof(v)) == CUBLAS_STATUS_SUCCESS;
}
static int get(const cublasLtMatmulAlgo_t *a, cublasLtMatmulAlgoConfigAttributes_t attr) {
    int v = 0;
    CHECK_BLAS(cublasLtMatmulAlgoConfigGetAttribute(a, attr, &v, sizeof(v), NULL));
    return v;
}
static void describe(const cublasLtMatmulAlgo_t *a) {
    uint16_t cluster = 0;
    CHECK_BLAS(cublasLtMatmulAlgoConfigGetAttribute(a, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID,
                                                    &cluster, sizeof(cluster), NULL));
    printf("\"algo_id\":%d,\"tile\":%d,\"stages\":%d,\"cluster\":%u,\"split_k\":%d,\"reduction\":%"
           "d,\"swizzle\":%d,\"custom\":%d",
           get(a, CUBLASLT_ALGO_CONFIG_ID), get(a, CUBLASLT_ALGO_CONFIG_TILE_ID),
           get(a, CUBLASLT_ALGO_CONFIG_STAGES_ID), cluster, get(a, CUBLASLT_ALGO_CONFIG_SPLITK_NUM),
           get(a, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME),
           get(a, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING), get(a, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION));
}
static void keep(Candidate *best, int *count, Candidate c) {
    int pos = *count;
    if (pos == FINALISTS) {
        if (c.us >= best[pos - 1].us)
            return;
        pos--;
    } else
        (*count)++;
    while (pos > 0 && c.us < best[pos - 1].us) {
        best[pos] = best[pos - 1];
        pos--;
    }
    best[pos] = c;
}
static int trial(Problem *p, const cublasLtMatmulAlgo_t *a, Candidate *best, int *count) {
    cublasLtMatmulHeuristicResult_t r;
    if (cublasLtMatmulAlgoCheck(p->h, p->op, p->a, p->b, p->c, p->c, a, &r) !=
            CUBLAS_STATUS_SUCCESS ||
        r.state != CUBLAS_STATUS_SUCCESS || r.workspaceSize > workspace_size)
        return 0;
    double us = measure(p, a, 1, 5);
    if (us == DBL_MAX)
        return 0;
    Candidate c = {*a, us, r.workspaceSize};
    keep(best, count, c);
    return 1;
}
/* Nonzero, exactly representable signed 1/8 inputs; independent CPU dot-product
 * checks at 64 distributed output coordinates. This is outside timed regions. */
static uint16_t small_half(int v) {
    if (!v)
        return 0;
    int sign = v < 0 ? 0x8000 : 0;
    v = abs(v);
    return sign | (v == 1 ? 0x3000 : v == 2 ? 0x3400 : 0x3600);
}
static float from16(uint16_t x) {
    if (output_type == CUDA_R_16BF) {
        uint32_t b = (uint32_t)x << 16;
        float f;
        memcpy(&f, &b, 4);
        return f;
    }
    int exp = (x >> 10) & 31, mant = x & 1023;
    float f = exp == 31 ? (mant ? NAN : INFINITY)
              : exp     ? ldexpf(1 + mant / 1024.0f, exp - 15)
                        : ldexpf(mant / 1024.0f, -14);
    return x & 0x8000 ? -f : f;
}
static int av(int row, int k) {
    return (int)(((uint64_t)row * 13 + (uint64_t)k * 7 + (uint64_t)(k / 11) * 3) % 7) - 3;
}
static int bv(int col, int k) {
    return (int)(((uint64_t)col * 11 + (uint64_t)k * 5 + (uint64_t)(k / 13) * 2) % 7) - 3;
}
static void validate(Problem *p, const cublasLtMatmulAlgo_t *a) {
    size_t count = (size_t)p->n * p->n, bytes = count * 2;
    uint16_t *host = malloc(bytes);
    if (!host)
        exit(1);
    for (int row = 0; row < p->n; row++)
        for (int k = 0; k < p->n; k++)
            host[(size_t)row * p->n + k] = small_half(av(row, k));
    CHECK_CUDA(cudaMemcpy(p->A, host, bytes, cudaMemcpyHostToDevice));
    for (int col = 0; col < p->n; col++)
        for (int k = 0; k < p->n; k++)
            host[(size_t)col * p->n + k] = small_half(bv(col, k));
    CHECK_CUDA(cudaMemcpy(p->B, host, bytes, cudaMemcpyHostToDevice));
    CHECK_BLAS(launch(p, a));
    CHECK_CUDA(cudaStreamSynchronize(p->stream));
    CHECK_CUDA(cudaMemcpy(host, p->C, bytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < 64; i++) {
        int row = (i * 7919 + 17) % p->n, col = (i * 6701 + 31) % p->n;
        double expected = 0;
        for (int k = 0; k < p->n; k++)
            expected += av(row, k) * bv(col, k) / 64.0;
        double actual = from16(host[(size_t)col * p->n + row]);
        double tolerance =
            fmax(0.03125, fabs(expected) * (output_type == CUDA_R_16BF ? 0.004 : 0.0005));
        if (!isfinite(actual) || fabs(actual - expected) > tolerance) {
            fprintf(stderr, "VALIDATION FAILED n=%d row=%d col=%d actual=%g expected=%g\n", p->n,
                    row, col, actual, expected);
            exit(1);
        }
    }
    free(host);
    CHECK_CUDA(cudaMemset(p->A, 0, bytes));
    CHECK_CUDA(cudaMemset(p->B, 0, bytes));
    CHECK_CUDA(cudaMemset(p->C, 0, bytes));
    CHECK_CUDA(cudaStreamSynchronize(p->stream));
}
static Candidate refine(Problem *p, Candidate *best, int count) {
    Candidate winner = best[0];
    winner.us = DBL_MAX;
    for (int j = 0; j < count; j++) {
        double ts[5];
        for (int r = 0; r < 5; r++)
            ts[r] = measure(p, &best[j].algo, 10, 31);
        double us = median(ts, 5);
        if (us < winner.us) {
            winner = best[j];
            winner.us = us;
        }
    }
    return winner;
}
static void report(Problem *p, Candidate chosen, const char *label, long checked, long valid,
                   int algo_count) {
    validate(p, &chosen.algo);
    double *times = calloc(repeats, sizeof(double));
    /* Long warmup once, then repeated independent event batches. */
    for (int i = 0; i < warmup; i++)
        CHECK_BLAS(launch(p, &chosen.algo));
    CHECK_CUDA(cudaStreamSynchronize(p->stream));
    for (int r = 0; r < repeats; r++) {
        times[r] = measure(p, &chosen.algo, 0, iters);
        if (!isfinite(times[r]) || times[r] <= 0 || times[r] == DBL_MAX) {
            fprintf(stderr, "Final timing failed; no result will be reported.\n");
            exit(1);
        }
    }
    printf("RESULT "
           "{\"implementation\":\"cublaslt\",\"mode\":\"%s\",\"n\":%d,\"input\":\"fp16\","
           "\"output\":\"%s\",\"compute\":\"fp32\",\"workspace_limit\":%zu,\"workspace_used\":%zu,"
           "\"warmup\":%d,\"iters\":%d,\"repeats\":%d,\"samples_us\":[",
           label, p->n, output_name, workspace_size, chosen.workspace, warmup, iters, repeats);
    for (int r = 0; r < repeats; r++)
        printf("%s%.9f", r ? "," : "", times[r]);
    double us = median(times, repeats);
    double tf = 2.0 * p->n * p->n * p->n / us / 1e6;
    printf("],\"median_us\":%.9f,\"tflops\":%.9f,\"configs_checked\":%ld,\"configs_timed\":%ld,"
           "\"algo_ids\":%d,\"validation\":\"64-nonzero-cpu-dot-products-pass\",",
           us, tf, checked, valid, algo_count);
    describe(&chosen.algo);
    printf("}\n");
    if (!strcmp(label, "top8") || !strcmp(mode, "exhaustive"))
        printf("  FP16 FP32 compute %dx%dx%d %.9f ms %.9f TFLOPS\n", p->n, p->n, p->n, us / 1000,
               tf);
    free(times);
    fflush(stdout);
}
static void bench(cublasLtHandle_t h, int n) {
    Problem p = {0};
    p.h = h;
    p.n = n;
    CHECK_CUDA(cudaStreamCreate(&p.stream));
    CHECK_CUDA(cudaEventCreate(&p.start));
    CHECK_CUDA(cudaEventCreate(&p.stop));
    size_t bytes = (size_t)n * n * 2;
    CHECK_CUDA(cudaMalloc(&p.A, bytes));
    CHECK_CUDA(cudaMalloc(&p.B, bytes));
    CHECK_CUDA(cudaMalloc(&p.C, bytes));
    CHECK_CUDA(cudaMalloc(&p.workspace, workspace_size));
    CHECK_CUDA(cudaMemset(p.A, 0, bytes));
    CHECK_CUDA(cudaMemset(p.B, 0, bytes));
    CHECK_CUDA(cudaMemset(p.C, 0, bytes));
    CHECK_BLAS(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t t = CUBLAS_OP_T, normal = CUBLAS_OP_N;
    CHECK_BLAS(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &t, sizeof(t)));
    CHECK_BLAS(
        cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &normal, sizeof(normal)));
    CHECK_BLAS(cublasLtMatrixLayoutCreate(&p.a, CUDA_R_16F, n, n, n));
    CHECK_BLAS(cublasLtMatrixLayoutCreate(&p.b, CUDA_R_16F, n, n, n));
    CHECK_BLAS(cublasLtMatrixLayoutCreate(&p.c, output_type, n, n, n));
    cublasLtMatmulPreference_t pref;
    CHECK_BLAS(cublasLtMatmulPreferenceCreate(&pref));
    CHECK_BLAS(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                    &workspace_size, sizeof(workspace_size)));
    cublasLtMatmulHeuristicResult_t heur[8] = {0};
    int returned = 0;
    cublasStatus_t status =
        cublasLtMatmulAlgoGetHeuristic(h, p.op, p.a, p.b, p.c, p.c, pref, 8, heur, &returned);
    if (status != CUBLAS_STATUS_SUCCESS || !returned) {
        fprintf(stderr, "No supported FP16->%s heuristic for %d (status=%d count=%d)\n",
                output_name, n, status, returned);
        exit(2);
    }
    Candidate top[FINALISTS];
    int ntop = 0;
    long topvalid = 0;
    for (int i = 0; i < returned; i++)
        topvalid += trial(&p, &heur[i].algo, top, &ntop);
    if (!ntop) {
        fprintf(stderr, "No valid top8 candidates\n");
        exit(2);
    }
    Candidate topwinner = refine(&p, top, ntop);
    if (strcmp(mode, "exhaustive"))
        report(&p, topwinner, "top8", returned, topvalid, 0);
    if (strcmp(mode, "top8")) {
        int capacity = 64, count = 0, *ids = NULL;
        do {
            capacity *= 2;
            ids = realloc(ids, capacity * sizeof(int));
            if (!ids)
                exit(1);
            CHECK_BLAS(cublasLtMatmulAlgoGetIds(h, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16F,
                                                CUDA_R_16F, output_type, output_type, capacity, ids,
                                                &count));
        } while (count == capacity);
        Candidate finalists[FINALISTS];
        int nf = 0;
        for (int i = 0; i < ntop; i++)
            keep(finalists, &nf, top[i]);
        const int splits[] = {0, 1, 2, 3, 4, 5, 6, 8, 12, 16, 32};
        long checked = 0, valid = 0;
        for (int ai = 0; ai < count; ai++) {
            cublasLtMatmulAlgo_t a;
            if (cublasLtMatmulAlgoInit(h, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16F, CUDA_R_16F,
                                       output_type, output_type, ids[ai],
                                       &a) != CUBLAS_STATUS_SUCCESS)
                continue;
            int tiles_count, stages_count;
            int *tiles = cap_list(&a, CUBLASLT_ALGO_CAP_TILE_IDS, &tiles_count),
                *stages = cap_list(&a, CUBLASLT_ALGO_CAP_STAGES_IDS, &stages_count);
            int split = cap(&a, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT),
                redmask = cap(&a, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK),
                swmax = cap(&a, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT),
                custommax = cap(&a, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX);
            for (int ti = 0; ti < tiles_count; ti++)
                for (int si = 0; si < stages_count; si++)
                    for (uint16_t cl = 0; cl < CUBLASLT_CLUSTER_SHAPE_END; cl++)
                        for (int cu = 0; cu <= custommax; cu++)
                            for (int sw = 0; sw <= swmax; sw++)
                                for (int ki = 0; ki < (split ? 11 : 2); ki++)
                                    for (int red = ki > 1 ? 1 : 0;
                                         red < (int)CUBLASLT_REDUCTION_SCHEME_MASK;
                                         red = red ? red * 2 : 8) {
                                        if (ki > 1 && !(red & redmask))
                                            continue;
                                        if (!set(&a, CUBLASLT_ALGO_CONFIG_TILE_ID, tiles[ti]) ||
                                            !set(&a, CUBLASLT_ALGO_CONFIG_STAGES_ID, stages[si]) ||
                                            !set(&a, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, cu) ||
                                            !set(&a, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, sw) ||
                                            !set(&a, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, splits[ki]) ||
                                            !set(&a, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, red))
                                            continue;
                                        if (cublasLtMatmulAlgoConfigSetAttribute(
                                                &a, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &cl,
                                                sizeof(cl)) != CUBLAS_STATUS_SUCCESS)
                                            continue;
                                        checked++;
                                        valid += trial(&p, &a, finalists, &nf);
                                    }
            fprintf(stderr, "SEARCH n=%d algo=%d/%d id=%d checked=%ld timed=%ld best_us=%.3f\n", n,
                    ai + 1, count, ids[ai], checked, valid, finalists[0].us);
            fflush(stderr);
            free(tiles);
            free(stages);
        }
        /* Remeasure every heuristic candidate as well as the screened grid
         * finalists: noisy screening must never evict the heuristic fallback. */
        Candidate combined[FINALISTS + 8];
        memcpy(combined, finalists, nf * sizeof(Candidate));
        memcpy(combined + nf, top, ntop * sizeof(Candidate));
        report(&p, refine(&p, combined, nf + ntop), "exhaustive-grid", checked, valid, count);
        free(ids);
    }
    CHECK_BLAS(cublasLtMatmulPreferenceDestroy(pref));
    CHECK_BLAS(cublasLtMatrixLayoutDestroy(p.a));
    CHECK_BLAS(cublasLtMatrixLayoutDestroy(p.b));
    CHECK_BLAS(cublasLtMatrixLayoutDestroy(p.c));
    CHECK_BLAS(cublasLtMatmulDescDestroy(p.op));
    CHECK_CUDA(cudaFree(p.A));
    CHECK_CUDA(cudaFree(p.B));
    CHECK_CUDA(cudaFree(p.C));
    CHECK_CUDA(cudaFree(p.workspace));
    CHECK_CUDA(cudaEventDestroy(p.start));
    CHECK_CUDA(cudaEventDestroy(p.stop));
    CHECK_CUDA(cudaStreamDestroy(p.stream));
}
static int positive(const char *s) {
    char *end;
    long v = strtol(s, &end, 10);
    if (*end || v <= 0 || v > 10000000) {
        fprintf(stderr, "Invalid positive integer: %s\n", s);
        exit(2);
    }
    return v;
}
int main(int argc, char **argv) {
    int only = 0;
    const char *env = getenv("GEMM_CUBLAS_MODE");
    if (env)
        mode = env;
    for (int i = 1; i < argc; i++) {
        if (i + 1 == argc) {
            fprintf(stderr,
                    "Usage: %s [--mode top8|exhaustive|both] [--n N] [--warmup N] [--iters N] "
                    "[--repeats odd-N] [--output fp16|bf16]\n",
                    argv[0]);
            return 2;
        }
        const char *key = argv[i++], *value = argv[i];
        if (!strcmp(key, "--mode"))
            mode = value;
        else if (!strcmp(key, "--n"))
            only = positive(value);
        else if (!strcmp(key, "--warmup"))
            warmup = positive(value);
        else if (!strcmp(key, "--iters"))
            iters = positive(value);
        else if (!strcmp(key, "--repeats"))
            repeats = positive(value);
        else if (!strcmp(key, "--output") && !strcmp(value, "fp16")) {
            output_type = CUDA_R_16F;
            output_name = "fp16";
        } else if (!strcmp(key, "--output") && !strcmp(value, "bf16")) {
            output_type = CUDA_R_16BF;
            output_name = "bf16";
        } else {
            fprintf(stderr, "Invalid option %s %s\n", key, value);
            return 2;
        }
    }
    if ((strcmp(mode, "top8") && strcmp(mode, "exhaustive") && strcmp(mode, "both")) ||
        !(repeats & 1)) {
        fprintf(stderr, "Invalid mode or even repeat count\n");
        return 2;
    }
    int dev, driver, runtime;
    struct cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDevice(&dev));
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
    CHECK_CUDA(cudaDriverGetVersion(&driver));
    CHECK_CUDA(cudaRuntimeGetVersion(&runtime));
    printf("GPU: %s, sm_%d%d, %d SMs\ncuBLASLt: %zu, CUDA runtime: %d, driver API: %d\n", prop.name,
           prop.major, prop.minor, prop.multiProcessorCount, cublasLtGetVersion(), runtime, driver);
    printf("Protocol: zero-data timing; CUDA events around launch batches; median of %d batches; "
           "%d warmup, %d launches/batch; tuning excluded; 32 MiB workspace\n",
           repeats, warmup, iters);
    printf("--- FP16 ---\n");
    fflush(stdout);
    cublasLtHandle_t h;
    CHECK_BLAS(cublasLtCreate(&h));
    if (only)
        bench(h, only);
    else {
        bench(h, 4096);
        bench(h, 8192);
        bench(h, 16384);
    }
    CHECK_BLAS(cublasLtDestroy(h));
    return 0;
}
