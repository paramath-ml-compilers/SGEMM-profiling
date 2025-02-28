
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

// #include "kernels.cuh"
#include "runner.cuh"

template <int BM, int BN, int BK, int WM, int WN, int WNITER,
          int TM, int TN, int THREADS>
__global__ void runSgemmDoubleBuffering2(
    int M, int N, int K, float alpha,
    float* A, float* B, float beta,
    float* C);

// -----------------------------------------------------
// Macro that instantiates and times runSgemmDoubleBuffering2<...>().
// Writes results (params, time, GFLOPS) to an output CSV file.
//
// Usage example:
//   PROFILE_KERNEL(outFile, BM, BN, BK, WM, WN, WNITER, TM, TN, THREADS);
// -----------------------------------------------------
#define PROFILE_KERNEL(                                                       \
	outFile, BM, BN, BK, WM, WN, WNITER, TM, TN, THREADS, M, N, K,            \
	alpha, dA, dB, beta, dC, totalFlops)                                      \
	do {                                                                      \
		/* Create CUDA events for timing */                                   \
		cudaEvent_t start, stop;                                              \
		cudaEventCreate(&start);                                              \
		cudaEventCreate(&stop);                                               \
                                                                              \
		/* Launch configuration */                                            \
		dim3 blockDim(THREADS);                                               \
		dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);                   \
                                                                              \
		/* Record start time */                                               \
		cudaEventRecord(start);                                               \
                                                                              \
		/* Launch the kernel (template instantiation) */                      \
		runSgemmDoubleBuffering2<BM, BN, BK, WM, WN, WNITER, TM, TN, THREADS> \
			<<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);        \
                                                                              \
		/* Record stop time and synchronize */                                \
		cudaEventRecord(stop);                                                \
		cudaEventSynchronize(stop);                                           \
                                                                              \
		float milliseconds = 0.0f;                                            \
		cudaEventElapsedTime(&milliseconds, start, stop);                     \
		double seconds = static_cast<double>(milliseconds) / 1000.0;          \
                                                                              \
		/* Compute GFLOPS = 2*M*N*K / (time * 1e9) */                         \
		double gflops = (totalFlops * 1e-9) / seconds;                        \
                                                                              \
		/* Write CSV row: BM,BN,BK,WM,WN,WNITER,TM,TN,THREADS,Time,GFLOPS */  \
		outFile << BM << "," << BN << "," << BK << ","                        \
				<< WM << "," << WN << "," << WNITER << ","                    \
				<< TM << "," << TN << "," << THREADS << ","                   \
				<< std::fixed << std::setprecision(6) << seconds << ","       \
				<< std::setprecision(2) << gflops << std::endl;               \
                                                                              \
		cudaEventDestroy(start);                                              \
		cudaEventDestroy(stop);                                               \
	} while (0)

int main() {
	// -----------------------------------------------------
	// 1. Setup problem size and allocate host/device memory
	// -----------------------------------------------------
	// For a single large test, pick M=N=K=8192 (adjust if needed)
	const int M = 8192;
	const int N = 8192;
	const int K = 8192;

	// For computing GFLOPS: total floating-point ops = 2 * M * N * K
	double totalFlops = 2.0 * static_cast<double>(M) *
						static_cast<double>(N) * static_cast<double>(K);

	// Host arrays
	std::vector<float> hA(M * K);
	std::vector<float> hB(K * N);
	std::vector<float> hC(M * N, 0.f);

	// Initialize input matrices (randomize, range_init, etc.)
	randomize_matrix(hA.data(), M * K);
	randomize_matrix(hB.data(), K * N);

	// Device arrays
	float *dA = nullptr, *dB = nullptr, *dC = nullptr;
	cudaCheck(cudaMalloc((void **)&dA, sizeof(float) * M * K), __FILE__, __LINE__);
	cudaCheck(cudaMalloc((void **)&dB, sizeof(float) * K * N), __FILE__, __LINE__);
	cudaCheck(cudaMalloc((void **)&dC, sizeof(float) * M * N), __FILE__, __LINE__);

	// Copy host -> device
	cudaCheck(cudaMemcpy(dA, hA.data(), sizeof(float) * M * K,
						 cudaMemcpyHostToDevice),
			  __FILE__, __LINE__);
	cudaCheck(cudaMemcpy(dB, hB.data(), sizeof(float) * K * N,
						 cudaMemcpyHostToDevice),
			  __FILE__, __LINE__);
	cudaCheck(cudaMemset(dC, 0, sizeof(float) * M * N), __FILE__, __LINE__);

	// We'll do alpha=1.0f, beta=0.0f for standard GEMM
	const float alpha = 1.0f;
	const float beta = 0.0f;

	// -----------------------------------------------------
	// 2. Profile Different Cases
	// -----------------------------------------------------
	std::ofstream outFileBM("BM_profiling.csv");
	outFileBM << "BM,BN,BK,WM,WN,WNITER,TM,TN,THREADS,TimeSec,GFLOPS\n";
	PROFILE_KERNEL(outFileBM, 64, 128, 32, 64, 64, 4, 8, 4, 64,
				   M, N, K, alpha, dA, dB, beta, dC, totalFlops);
	cudaCheck(cudaMemset(dC, 0, sizeof(float) * M * N), __FILE__, __LINE__);

	PROFILE_KERNEL(outFileBM, 128, 128, 32, 64, 64, 4, 8, 4, 64,
				   M, N, K, alpha, dA, dB, beta, dC, totalFlops);
	cudaCheck(cudaMemset(dC, 0, sizeof(float) * M * N), __FILE__, __LINE__);

	PROFILE_KERNEL(outFileBM, 256, 128, 32, 64, 64, 4, 8, 4, 64,
		M, N, K, alpha, dA, dB, beta, dC, totalFlops);
	cudaCheck(cudaMemset(dC, 0, sizeof(float) * M * N), __FILE__, __LINE__);	
	outFileBM.close();


	std::ofstream outFileBN("BN_profiling.csv");
	outFileBN << "BM,BN,BK,WM,WN,WNITER,TM,TN,THREADS,TimeSec,GFLOPS\n";
	PROFILE_KERNEL(outFileBN, 128, 64, 32, 64, 64, 4, 8, 4, 64,
				   M, N, K, alpha, dA, dB, beta, dC, totalFlops);
	cudaCheck(cudaMemset(dC, 0, sizeof(float) * M * N), __FILE__, __LINE__);

	PROFILE_KERNEL(outFileBN, 128, 128, 32, 64, 64, 4, 8, 4, 64,
				   M, N, K, alpha, dA, dB, beta, dC, totalFlops);
	cudaCheck(cudaMemset(dC, 0, sizeof(float) * M * N), __FILE__, __LINE__);

	PROFILE_KERNEL(outFileBN, 128, 256, 32, 64, 64, 4, 8, 4, 64,
		M, N, K, alpha, dA, dB, beta, dC, totalFlops);
	cudaCheck(cudaMemset(dC, 0, sizeof(float) * M * N), __FILE__, __LINE__);	
	outFileBN.close();

	std::ofstream outFileBK("BK_profiling.csv");
	outFileBK << "BM,BN,BK,WM,WN,WNITER,TM,TN,THREADS,TimeSec,GFLOPS\n";

	PROFILE_KERNEL(outFileBK, 128, 128, 16, 64, 64, 4, 8, 4, 64,
				   M, N, K, alpha, dA, dB, beta, dC, totalFlops);
	cudaCheck(cudaMemset(dC, 0, sizeof(float) * M * N), __FILE__, __LINE__);

	PROFILE_KERNEL(outFileBK, 128, 128, 32, 64, 64, 4, 8, 4, 64,
				   M, N, K, alpha, dA, dB, beta, dC, totalFlops);
	cudaCheck(cudaMemset(dC, 0, sizeof(float) * M * N), __FILE__, __LINE__);

	PROFILE_KERNEL(outFileBK, 128, 128, 64, 64, 64, 4, 8, 4, 64,
		M, N, K, alpha, dA, dB, beta, dC, totalFlops);
	cudaCheck(cudaMemset(dC, 0, sizeof(float) * M * N), __FILE__, __LINE__);
		
	outFileBK.close();


	// std::ofstream outFileWM("WM_profiling.csv");
	// outFileWM << "BM,BN,BK,WM,WN,WNITER,TM,TN,THREADS,TimeSec,GFLOPS\n";

	// PROFILE_KERNEL(outFileWM, 128, 128, 32, 32, 64, 4, 8, 4, 64,
	// 			   M, N, K, alpha, dA, dB, beta, dC, totalFlops);
	// cudaCheck(cudaMemset(dC, 0, sizeof(float) * M * N), __FILE__, __LINE__);

	// PROFILE_KERNEL(outFileWM, 128, 128, 32, 64, 64, 4, 8, 4, 64,
	// 			   M, N, K, alpha, dA, dB, beta, dC, totalFlops);
	// cudaCheck(cudaMemset(dC, 0, sizeof(float) * M * N), __FILE__, __LINE__);

	// PROFILE_KERNEL(outFileWM, 128, 128, 32, 128, 64, 4, 8, 4, 64,
	// 	M, N, K, alpha, dA, dB, beta, dC, totalFlops);
	// cudaCheck(cudaMemset(dC, 0, sizeof(float) * M * N), __FILE__, __LINE__);
		
	// outFileWM.close();


	// std::ofstream outFileWN("WN_profiling.csv");
	// outFileWN << "BM,BN,BK,WM,WN,WNITER,TM,TN,THREADS,TimeSec,GFLOPS\n";

	// PROFILE_KERNEL(outFileWN, 128, 128, 32, 64, 32, 4, 8, 4, 64,
	// 			   M, N, K, alpha, dA, dB, beta, dC, totalFlops);
	// cudaCheck(cudaMemset(dC, 0, sizeof(float) * M * N), __FILE__, __LINE__);

	// PROFILE_KERNEL(outFileWN, 128, 128, 32, 64, 64, 4, 8, 4, 64,
	// 			   M, N, K, alpha, dA, dB, beta, dC, totalFlops);
	// cudaCheck(cudaMemset(dC, 0, sizeof(float) * M * N), __FILE__, __LINE__);

	// PROFILE_KERNEL(outFileWN, 128, 128, 32, 64, 128, 4, 8, 4, 64,
	// 	M, N, K, alpha, dA, dB, beta, dC, totalFlops);
	// cudaCheck(cudaMemset(dC, 0, sizeof(float) * M * N), __FILE__, __LINE__);
		
	// outFileWN.close();


	// std::ofstream outFileWNITER("WNITER_profiling.csv");
	// outFileWNITER << "BM,BN,BK,WM,WN,WNITER,TM,TN,THREADS,TimeSec,GFLOPS\n";

	// PROFILE_KERNEL(outFileWNITER, 128, 128, 32, 64, 64, 1, 8, 4, 64,
	// 			   M, N, K, alpha, dA, dB, beta, dC, totalFlops);
	// cudaCheck(cudaMemset(dC, 0, sizeof(float) * M * N), __FILE__, __LINE__);

	// PROFILE_KERNEL(outFileWNITER, 128, 128, 32, 64, 64, 2, 8, 4, 64,
	// 			   M, N, K, alpha, dA, dB, beta, dC, totalFlops);
	// cudaCheck(cudaMemset(dC, 0, sizeof(float) * M * N), __FILE__, __LINE__);

	// PROFILE_KERNEL(outFileWNITER, 128, 128, 32, 64, 64, 4, 8, 4, 64,
	// 	M, N, K, alpha, dA, dB, beta, dC, totalFlops);
	// cudaCheck(cudaMemset(dC, 0, sizeof(float) * M * N), __FILE__, __LINE__);
		
	// outFileWNITER.close();



	// PROFILE_KERNEL(outFile, 128, 64, 16, 64, 64, 4, 8, 4, 64,
	// 			   M, N, K, alpha, dA, dB, beta, dC, totalFlops);

	// cudaCheck(cudaMemset(dC, 0, sizeof(float) * M * N), __FILE__, __LINE__);

	// PROFILE_KERNEL(outFile, 128, 64, 16, 64, 64, 4, 8, 4, 128,
	// 			   M, N, K, alpha, dA, dB, beta, dC, totalFlops);

	// cudaCheck(cudaMemset(dC, 0, sizeof(float) * M * N), __FILE__, __LINE__);

	// PROFILE_KERNEL(outFile, 128, 64, 16, 64, 64, 4, 8, 4, 256,
	// 			   M, N, K, alpha, dA, dB, beta, dC, totalFlops);

	// cudaCheck(cudaMemset(dC, 0, sizeof(float) * M * N), __FILE__, __LINE__);

	// PROFILE_KERNEL(outFile, 128, 64, 32, 64, 64, 4, 8, 4, 64,
	// 			   M, N, K, alpha, dA, dB, beta, dC, totalFlops);

	// cudaCheck(cudaMemset(dC, 0, sizeof(float) * M * N), __FILE__, __LINE__);

	// PROFILE_KERNEL(outFile, 128, 64, 32, 64, 64, 4, 8, 4, 128,
	// 			   M, N, K, alpha, dA, dB, beta, dC, totalFlops);

	// cudaCheck(cudaMemset(dC, 0, sizeof(float) * M * N), __FILE__, __LINE__);

	// PROFILE_KERNEL(outFile, 128, 64, 32, 64, 64, 4, 8, 4, 256,
	// 			   M, N, K, alpha, dA, dB, beta, dC, totalFlops);

	// cudaCheck(cudaMemset(dC, 0, sizeof(float) * M * N), __FILE__, __LINE__);

	// PROFILE_KERNEL(outFile, 128, 128, 16, 64, 64, 4, 8, 4, 64,
	// 			   M, N, K, alpha, dA, dB, beta, dC, totalFlops);

	// cudaCheck(cudaMemset(dC, 0, sizeof(float) * M * N), __FILE__, __LINE__);

	// PROFILE_KERNEL(outFile, 128, 128, 16, 64, 64, 4, 8, 4, 128,
	// 			   M, N, K, alpha, dA, dB, beta, dC, totalFlops);

	// cudaCheck(cudaMemset(dC, 0, sizeof(float) * M * N), __FILE__, __LINE__);

	// PROFILE_KERNEL(outFile, 128, 128, 16, 64, 64, 4, 8, 4, 256,
	// 			   M, N, K, alpha, dA, dB, beta, dC, totalFlops);

	// cudaCheck(cudaMemset(dC, 0, sizeof(float) * M * N), __FILE__, __LINE__);

	// PROFILE_KERNEL(outFile, 128, 128, 32, 64, 64, 4, 8, 4, 64,
	// 			   M, N, K, alpha, dA, dB, beta, dC, totalFlops);

	// cudaCheck(cudaMemset(dC, 0, sizeof(float) * M * N), __FILE__, __LINE__);

	// PROFILE_KERNEL(outFile, 128, 128, 32, 64, 64, 4, 8, 4, 128,
	// 			   M, N, K, alpha, dA, dB, beta, dC, totalFlops);

	// cudaCheck(cudaMemset(dC, 0, sizeof(float) * M * N), __FILE__, __LINE__);

	// PROFILE_KERNEL(outFile, 128, 128, 32, 64, 64, 4, 8, 4, 256,
	// 			   M, N, K, alpha, dA, dB, beta, dC, totalFlops);

	// cudaCheck(cudaMemset(dC, 0, sizeof(float) * M * N), __FILE__, __LINE__);

	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);

	return 0;
}
