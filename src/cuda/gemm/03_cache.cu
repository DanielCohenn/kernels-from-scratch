#include <cstdio>
#include <cstdlib>

#include "gemmKernel.cuh"


template <typename T>
__global__ void cache_tiled_gemm(const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C,
                                    int m, int n, int k, T alpha, T beta)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    int row_a = 0;
    int col_a = 0;
    
    int row_b = 0;
    int col_b = 0;

    __shared__ float tile_a[TILE * TILE];
    __shared__ float tile_b[TILE * TILE];

    float res = 0.0f;
    for (int i = 0; i < k; i+=TILE) {
        // load elements to tiles in shmem
        row_a = ty;
        col_a = i + threadIdx.x;
        if ((row_a < m) && (col_a < k)) {
            tile_a[threadIdx.x * TILE + threadIdx.y] = A[row_a * k + col_a];
        }
        else {
            tile_a[threadIdx.x * TILE + threadIdx.y] = 0.0f;
        }

        row_b = i + threadIdx.y;
        col_b = tx;
        if ((row_b < k) && (col_b < n)) {
            tile_b[threadIdx.y * TILE + threadIdx.x] = B[row_b * n + col_b];
        }
        else {
            tile_b[threadIdx.y * TILE + threadIdx.x] = 0.0f;
        }

        // Here we will have one single tile from A and B loaded
        __syncthreads();

        // accumulate for each element in C
        for (int j = 0; j < TILE; j+=1) {
            res += tile_a[j * TILE + threadIdx.y] * tile_b[j * TILE + threadIdx.x];
        }

        __syncthreads();
    }

    if ((tx < n) && (ty < m)) {
        C[ty * n + tx] = alpha * res + beta * C[ty * n + tx];
    }
}


template <typename T>
class CacheTiledGemm : public GemmKernel<T>
{
    using GemmKernel<T>::GemmKernel;
    void launch(const T* d_matA, const T* d_matB, T* d_matC, T alpha, T beta) override {
        this->threadsPerBlock = dim3(TILE, TILE);
        this->blocksPerGrid = dim3((this->n + TILE - 1) / TILE,
                              (this->m + TILE - 1) / TILE);

        cache_tiled_gemm<T><<<this->blocksPerGrid, this->threadsPerBlock>>>(d_matA, d_matB, d_matC,
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

    CacheTiledGemm<float> gemm(m, n, k, numIter, "gemm_cache_tiled");
    GemmMetrics metrics = gemm.run(h_matA, h_matB, h_matC, alpha, beta);

    std::cout << metrics.phase << " m=" << metrics.m << " n=" << metrics.n << " k=" << metrics.k
              << " time(ms)=" << metrics.timeMs << " valid=" << metrics.isValid << std::endl;

    writeMetricsCSV(metrics, "bench/gemm_results.csv");

    delete[] h_matA;
    delete[] h_matB;
    delete[] h_matC;

    return 0;
}
