#include <cstdio>
#include <cstdlib>

#include "gemmKernel.cuh"


template <typename T>
__global__ void naive_gemm_kernel(const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C, 
                                    int m, int n, int k, T alpha, T beta)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if ((tx < m) && (ty < n)) {
        T res = 0;
        for (int i = 0; i < k; ++i) {
            res += A[tx * k + i] * B[i * n + ty];
        }

        C[tx * n + ty] = alpha * res + beta * C[tx * n + ty];
    }
}


template <typename T>
class NaiveGemm : public GemmKernel<T> 
{
    using GemmKernel<T>::GemmKernel;
    void launch(const T* d_matA, const T* d_matB, T* d_matC, T alpha, T beta) override {
        naive_gemm_kernel<T><<<this->blocksPerGrid, this->threadsPerBlock>>>(d_matA, d_matB, d_matC, 
            this->m, this->n, this->k, alpha, beta);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
};

int main(int argc, char** argv) 
{
    int m = 1024, n = 1024, k = 1024;
    int numIter = 20;
    if (argc == 4) {
        m = std::atoi(argv[1]);
        n = std::atoi(argv[2]);
        k = std::atoi(argv[3]);
    }

    float alpha = 1.0f, beta = 0.0f;

    float* h_matA = new float[m * k];
    float* h_matB = new float[k * n];
    float* h_matC = new float[m * n];

    for (int i = 0; i < m * k; ++i) h_matA[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < k * n; ++i) h_matB[i] = static_cast<float>(rand()) / RAND_MAX;

    NaiveGemm<float> gemm(m, n, k, numIter, "gemm_naive");
    GemmMetrics metrics = gemm.run(h_matA, h_matB, h_matC, alpha, beta);

    std::cout << metrics.phase << " m=" << metrics.m << " n=" << metrics.n << " k=" << metrics.k
              << " time(ms)=" << metrics.timeMs << " valid=" << metrics.isValid << std::endl;

    writeMetricsCSV(metrics, "bench/gemm_results.csv");

    delete[] h_matA;
    delete[] h_matB;
    delete[] h_matC;

    return 0;
}
