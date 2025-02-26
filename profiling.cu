#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <runner.cuh>
#include <cublas_v2.h>

// Kernel12 template parameters:
//   @tparam BM The threadblock size for M dimension SMEM caching.
//   @tparam BN The threadblock size for N dimension SMEM caching.
//   @tparam BK The threadblock size for K dimension SMEM caching.
//   @tparam WM M dim of continuous tile computed by each warp
//   @tparam WN N dim of continuous tile computed by each warp
//   @tparam WMITER The number of subwarp tiling steps in M dimension.
//   @tparam WNITER The number of subwarp tiling steps in N dimension.
//   @tparam TM The per-thread tile size for M dimension.
//   @tparam TN The per-thread tile size for N dimension.

#define cudaCheck(error) \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

int main() {
    // Use fixed large dimensions and GEMM parameters
    int m = 4096, n = 4096, k = 4096;
    float alpha = 0.5f, beta = 3.0f;

    cudaCheck(cudaSetDevice(0));

    // Create cuBLAS handle
    cublasHandle_t handle;
    if(cublasCreate(&handle)) {
        std::cerr << "Failed to create cuBLAS handle." << std::endl;
        return EXIT_FAILURE;
    }

    // Allocate host matrices
    size_t sizeA = m * k, sizeB = k * n, sizeC = m * n;
    float *hA = (float*)malloc(sizeof(float) * sizeA);
    float *hB = (float*)malloc(sizeof(float) * sizeB);
    float *hC = (float*)malloc(sizeof(float) * sizeC);
    // Initialize with random data
    for (size_t i = 0; i < sizeA; i++) hA[i] = static_cast<float>(rand()) / RAND_MAX;
    for (size_t i = 0; i < sizeB; i++) hB[i] = static_cast<float>(rand()) / RAND_MAX;
    for (size_t i = 0; i < sizeC; i++) hC[i] = static_cast<float>(rand()) / RAND_MAX;

    // Allocate device matrices
    float *dA, *dB, *dC;
    cudaCheck(cudaMalloc((void**)&dA, sizeof(float) * sizeA));
    cudaCheck(cudaMalloc((void**)&dB, sizeof(float) * sizeB));
    cudaCheck(cudaMalloc((void**)&dC, sizeof(float) * sizeC));

    cudaCheck(cudaMemcpy(dA, hA, sizeof(float) * sizeA, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dB, hB, sizeof(float) * sizeB, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dC, hC, sizeof(float) * sizeC, cudaMemcpyHostToDevice));

    // Timing kernel12 (run_kernel with kernel_num 12)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    run_kernel(13, m, n, k, alpha, dA, dB, beta, dC, handle);
    cudaCheck(cudaDeviceSynchronize());

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    double flops = 2.0 * m * n * k;
    double gflops = (flops / (milliseconds/1000.0)) * 1e-9;

    std::cout << "Kernel12 execution time: " << (milliseconds/1000.0) << " s" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;

    free(hA); free(hB); free(hC);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}


