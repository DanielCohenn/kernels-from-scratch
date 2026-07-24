# kernels-from-scratch

Self-study repo for GPU kernel design and AI inference optimization. GPU kernels written
from scratch and optimized step by step — naive → tiled → tensor-core — in CUDA and Triton.
Each kernel is benchmarked against cuBLAS / `torch.compile` and profiled with Nsight Compute.

## Structure

```
src/cuda/     CUDA kernels
src/triton/   Triton kernels
bench/        benchmark harness (vs cuBLAS / torch.compile)
ncu/          Nsight Compute summaries (raw .ncu-rep gitignored — too large/binary to track)
```

## Status

| Kernel | CUDA | Triton | vs cuBLAS / torch.compile | NCU report |
|---|---|---|---|---|
| SGEMM (naive) | ☐ | — | | |
| SGEMM (tiled) | ☐ | ☐ | | |
| SGEMM (tensor-core) | ☐ | ☐ | | |
| Softmax | ☐ | ☐ | | |
| Fused op | ☐ | ☐ | | |

## Build

```
make ARCH=sm_86    # RTX A5000 (daily driver)
make ARCH=sm_120    # RTX PRO 4000 Blackwell (TMA / FP8 / NVFP4 work)
```

## Hardware

Developed on a rented RTX A5000 (24GB, sm_86); tensor-core/FP8/NVFP4 kernels on an RTX PRO
4000 Blackwell (24GB, sm_120 — has single-CTA TMA, no TMEM/tcgen05/WGMMA). Hopper-specific
evidence (WGMMA, FlashAttention-3 style pipelining) gathered in short rented H100 bursts.

## Methodology

Every kernel gets: a correctness check against a reference implementation, a benchmark vs
cuBLAS or `torch.compile`, and an Nsight Compute pass with a roofline placement.
