#pragma once

#include <stdlib.h>
#include <cuda_runtime.h>
#include <string>
#include <fstream>


#define THREADS_PER_BLOCK 16

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

struct GemmMetrics {
    std::string phase;
    int m, n, k;
    std::string dtype;
    float timeMs;
    bool isValid;
};

inline void writeMetricsCSV(const GemmMetrics& metrics, const std::string& path) {
    bool fileExists = std::ifstream(path).good();
    std::ofstream ofs(path, std::ios::app);
    if (!fileExists) {
        ofs << "phase,m,n,k,dtype,time_ms,is_valid\n";
    }
    ofs << metrics.phase << "," << metrics.m << "," << metrics.n << "," << metrics.k << ","
        << metrics.dtype << "," << metrics.timeMs << "," << metrics.isValid << "\n";
}