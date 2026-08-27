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
python build.py     # auto-detects the local GPU's compute capability, syncs .clangd, runs make
make ARCH=sm_86      # manual override: RTX A5000 (daily driver)
make ARCH=sm_120      # manual override: RTX PRO 4000 Blackwell (TMA / FP8 / NVFP4 work)
```

Default build is release: `-O3`, `--use_fast_math`, `-march=native`. For benchmarking, this is what
you want — timings should reflect the fully-optimized kernel. For `cuda-gdb`/stepping through a
kernel, build debug instead (unoptimized, device symbols, no fast-math):

```
make DEBUG=1                  # or: python build.py DEBUG=1
```

Switching between release and debug forces a full rebuild even though the `.cu` sources didn't
change (the Makefile tracks the active flags, not just file mtimes).

## Hardware

Developed on a rented RTX A5000 (24GB, sm_86); tensor-core/FP8/NVFP4 kernels on an RTX PRO
4000 Blackwell (24GB, sm_120 — has single-CTA TMA, no TMEM/tcgen05/WGMMA). Hopper-specific
evidence (WGMMA, FlashAttention-3 style pipelining) gathered in short rented H100 bursts.

## Methodology

Every kernel gets: a correctness check against a reference implementation, a benchmark vs
cuBLAS or `torch.compile`, and an Nsight Compute pass with a roofline placement.
