#include <string>
#include <iostream>


struct GemmMetrics {
    std::string phase;

    int m, n, k;

    std::string dtype;

    float timeMs;

    bool isValid;
};

template <typename T>
class GemmKernel {
private:
    int m, n, k;
    T* h_resReference;

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
    GemmKernel(int m, int n, int k) : m(m), n(n), k(k) {
        h_resReference = new T[m * n];
    };

    GemmMetrics run(const T* h_matA, const T* h_matB, T* h_matC, T alpha, T beta) {
        // allocate device mem

        // h2d

        // launch kernel

        // d2h

        // sync
    }

    virtual void launch(const T* d_matA, const T* d_matB, T* d_matC, T alpha, T beta) = 0;

    virtual ~GemmKernel() {
        delete[] h_resReference;
    }
};