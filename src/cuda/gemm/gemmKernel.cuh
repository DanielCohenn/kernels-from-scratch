#pragma once

#include <string>
#include <iostream>
#include <cmath>
#include <typeinfo>
#include <cuda_runtime.h>

#include "../utils.cuh"


template <typename T>
class GemmKernel 
{
private:

    void hostNaiveGemm(const T* h_matA, const T* h_matB, T alpha, T beta) {
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                T sum = 0;
                for (int p = 0; p < k; ++p) {
                    sum += h_matA[i * k + p] * h_matB[p * n + j];
                }
                h_resReference[i * n + j] = alpha * sum + beta * h_resReference[i * n + j];
            }
        }
    }

    bool validateGemm(const T* h_matC, const T* h_resReference) {
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (std::abs(h_matC[i * n + j] - h_resReference[i * n + j]) > 1e-3) {
                    std::cerr << "Validation failed at (" << i << ", " << j << "): "
                              << h_matC[i * n + j] << " != " << h_resReference[i * n + j] << std::endl;
                    return false;
                }
            }
        }

        return true;
    }

public:
    int m, n, k;
    std::string phase;
    dim3 threadsPerBlock;
    dim3 blocksPerGrid;
    T* h_resReference;

    GemmKernel(int m, int n, int k, std::string phase = "unnamed") : m(m), n(n), k(k), phase(phase) {
        h_resReference = new T[m * n]();
        threadsPerBlock = dim3(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
        blocksPerGrid = dim3((m + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
                              (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    };

    GemmMetrics run(const T* h_matA, const T* h_matB, T* h_matC, T alpha, T beta) {
        T* d_matA, * d_matB, * d_matC;
        GemmMetrics gemmMetrics;
        cudaEvent_t start, stop;
        float ms = 0.0f;

        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaMalloc((void**)&d_matA, m * k * sizeof(T)));
        CUDA_CHECK(cudaMalloc((void**)&d_matB, k * n * sizeof(T)));
        CUDA_CHECK(cudaMalloc((void**)&d_matC, m * n * sizeof(T)));

        CUDA_CHECK(cudaMemcpy(d_matA, h_matA, m * k * sizeof(T), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_matB, h_matB, k * n * sizeof(T), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemset(d_matC, 0.0, m * n * sizeof(T)));

        cudaEventRecord(start);
        launch(d_matA, d_matB, d_matC, alpha, beta);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);

        CUDA_CHECK(cudaMemcpy(h_matC, d_matC, m * n * sizeof(T), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_matA));
        CUDA_CHECK(cudaFree(d_matB));
        CUDA_CHECK(cudaFree(d_matC));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        this->hostNaiveGemm(h_matA, h_matB, alpha, beta);

        gemmMetrics.phase = this->phase;
        gemmMetrics.m = m;
        gemmMetrics.n = n;
        gemmMetrics.k = k;
        gemmMetrics.dtype = typeid(T).name();
        gemmMetrics.timeMs = ms;
        gemmMetrics.isValid = this->validateGemm(h_matC, this->h_resReference);

        return gemmMetrics;
    }

    virtual void launch(const T* d_matA, const T* d_matB, T* d_matC, T alpha, T beta) = 0;

    virtual ~GemmKernel() {
        delete[] h_resReference;
    }
};