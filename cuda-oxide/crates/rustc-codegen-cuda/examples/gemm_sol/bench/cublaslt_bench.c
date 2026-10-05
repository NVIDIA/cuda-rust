/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublasLt.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = (call); \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: status=%d (%s)\n", __FILE__, __LINE__, \
                (int)status, cublasLtGetStatusString(status)); \
        exit(1); \
    } \
} while(0)

static double bench_cublaslt(cublasLtHandle_t handle, cudaStream_t stream,
                              int M, int N, int K,
                              cudaDataType_t dataType, const char *label) {
    size_t elem_bytes = 2;
    size_t workspace_size = 32 * 1024 * 1024;

    void *d_A, *d_B, *d_C, *d_workspace;
    CHECK_CUDA(cudaMalloc(&d_A, (size_t)M * K * elem_bytes));
    CHECK_CUDA(cudaMalloc(&d_B, (size_t)K * N * elem_bytes));
    CHECK_CUDA(cudaMalloc(&d_C, (size_t)M * N * elem_bytes));
    CHECK_CUDA(cudaMalloc(&d_workspace, workspace_size));
    CHECK_CUDA(cudaMemset(d_A, 0, (size_t)M * K * elem_bytes));
    CHECK_CUDA(cudaMemset(d_B, 0, (size_t)K * N * elem_bytes));
    CHECK_CUDA(cudaMemset(d_C, 0, (size_t)M * N * elem_bytes));

    cublasLtMatmulDesc_t matmulDesc;
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    cublasLtMatrixLayout_t layoutA, layoutB, layoutC, layoutD;
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutA, dataType, K, M, K));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutB, dataType, K, N, K));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutC, dataType, M, N, M));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutD, dataType, M, N, M));

    cublasLtMatmulPreference_t preference;
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_size, sizeof(workspace_size)));

    cublasLtMatmulHeuristicResult_t heuristicResult;
    int returnedAlgoCount = 0;
    cublasStatus_t heurStatus = cublasLtMatmulAlgoGetHeuristic(
        handle, matmulDesc, layoutA, layoutB, layoutC, layoutD,
        preference, 1, &heuristicResult, &returnedAlgoCount);

    double tflops = 0.0;
    if (heurStatus == CUBLAS_STATUS_SUCCESS && returnedAlgoCount > 0) {
        float alpha = 1.0f, beta = 0.0f;
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        for (int i = 0; i < 10; i++) {
            cublasLtMatmul(handle, matmulDesc,
                &alpha, d_A, layoutA, d_B, layoutB, &beta,
                d_C, layoutC, d_C, layoutD,
                &heuristicResult.algo, d_workspace, workspace_size, stream);
        }
        CHECK_CUDA(cudaStreamSynchronize(stream));

        int iters = 100;
        CHECK_CUDA(cudaEventRecord(start, stream));
        for (int i = 0; i < iters; i++) {
            cublasLtMatmul(handle, matmulDesc,
                &alpha, d_A, layoutA, d_B, layoutB, &beta,
                d_C, layoutC, d_C, layoutD,
                &heuristicResult.algo, d_workspace, workspace_size, stream);
        }
        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));

        float elapsed_ms;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
        double avg_ms = elapsed_ms / iters;
        double flops = 2.0 * (double)M * N * K;
        tflops = (flops / (avg_ms / 1000.0)) / 1e12;
        printf("  %-24s %5dx%5dx%5d  %8.4f ms  %8.1f TFLOPS\n",
               label, M, N, K, avg_ms, tflops);

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    } else {
        printf("  %-24s %5dx%5dx%5d  NO ALGORITHM (status=%d)\n",
               label, M, N, K, (int)heurStatus);
    }

    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtMatrixLayoutDestroy(layoutB);
    cublasLtMatrixLayoutDestroy(layoutC);
    cublasLtMatrixLayoutDestroy(layoutD);
    cublasLtMatmulDescDestroy(matmulDesc);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_workspace));

    return tflops;
}

int main() {
    int dev;
    CHECK_CUDA(cudaGetDevice(&dev));
    struct cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
    printf("GPU: %s, sm_%d%d, %d SMs\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    printf("cublasLt version: %zu\n\n", cublasLtGetVersion());

    cublasLtHandle_t handle;
    CHECK_CUBLAS(cublasLtCreate(&handle));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    int sizes[][3] = {
        {4096, 4096, 4096},
        {8192, 8192, 8192},
        {16384, 16384, 16384},
    };
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("==========================================================\n");
    printf("  cublasLtMatmul Benchmark (TN format, FP32 compute)\n");
    printf("==========================================================\n");

    printf("\n--- BF16 ---\n");
    for (int i = 0; i < num_sizes; i++)
        bench_cublaslt(handle, stream, sizes[i][0], sizes[i][1], sizes[i][2], CUDA_R_16BF, "BF16 FP32 compute");

    printf("\n--- FP16 ---\n");
    for (int i = 0; i < num_sizes; i++)
        bench_cublaslt(handle, stream, sizes[i][0], sizes[i][1], sizes[i][2], CUDA_R_16F, "FP16 FP32 compute");

    printf("\n==========================================================\n");

    cublasLtDestroy(handle);
    CHECK_CUDA(cudaStreamDestroy(stream));

    return 0;
}
