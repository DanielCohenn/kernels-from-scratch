#include <string>


struct GemmMetrics {
    std::string phase;

    int m, n, k;

    std::string dtype;

    float timeMs;
};

template <typename T>
class GemmKernel {
private:
    int m, n, k;
public:
    GemmKernel(int m, int n, int k) : m(m), n(n), k(k) {};

    GemmMetrics run(const T* h_matA, const T* h_matB, T* h_matC, T alpha, T beta);

    virtual void launch(const T* d_matA, const T* d_matB, T* d_matC, T alpha, T beta) = 0;

    virtual ~GemmKernel() = default;
};