import argparse
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch

DTYPE_BYTES = {"f": 4, "d": 8, "float32": 4, "float64": 8}


def load_csvs(paths):
    frames = [pd.read_csv(p) for p in paths]
    return pd.concat(frames, ignore_index=True)


def torch_baseline(sizes, dtype=torch.float32, warmup=5, iters=20):
    device = torch.device("cuda")
    rows = []
    for size in sizes:
        a = torch.rand(size, size, device=device, dtype=dtype)
        b = torch.rand(size, size, device=device, dtype=dtype)

        for _ in range(warmup):
            torch.matmul(a, b)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            torch.matmul(a, b)
        end.record()
        torch.cuda.synchronize()

        rows.append({
            "phase": "gemm_pytorch",
            "m": size, "n": size, "k": size,
            "dtype": "float32" if dtype == torch.float32 else "float64",
            "time_ms": start.elapsed_time(end) / iters,
            "is_valid": True,
        })
    return pd.DataFrame(rows)


def tflops(row):
    return 2 * row["m"] * row["n"] * row["k"] / (row["time_ms"] * 1e-3) / 1e12


def mem_bw_gbs(row):
    # idealized lower-bound traffic (each operand touched once) — not actual
    # measured DRAM throughput; use NCU for real achieved bandwidth per kernel.
    itemsize = DTYPE_BYTES.get(str(row["dtype"]), 4)
    bytes_moved = (row["m"] * row["k"] + row["k"] * row["n"] + row["m"] * row["n"]) * itemsize
    return bytes_moved / (row["time_ms"] * 1e-3) / 1e9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csvs", nargs="*", default=["bench/gemm_results.csv"])
    parser.add_argument("--no-pytorch", action="store_true", help="skip torch baseline + speedup subplot")
    parser.add_argument("--out", default="bench/plots/gemm_benchmark.png")
    args = parser.parse_args()

    paths = [p for pattern in args.csvs for p in sorted(glob.glob(pattern))]
    df = load_csvs(paths)
    df = df[(df["m"] == df["n"]) & (df["m"] == df["k"])].copy()
    df["mat_size"] = df["m"]

    if not args.no_pytorch:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available for torch baseline")
        sizes = sorted(df["mat_size"].unique().tolist())
        df = pd.concat([df, torch_baseline(sizes)], ignore_index=True)

    df["tflops"] = df.apply(tflops, axis=1)
    df["mem_bw_gbs"] = df.apply(mem_bw_gbs, axis=1)

    phases = df["phase"].unique()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    for phase in phases:
        sub = df[df["phase"] == phase].sort_values("mat_size")
        ax.plot(sub["mat_size"], sub["tflops"], marker="o", label=phase)
    ax.set_xlabel("Matrix size (n)"); ax.set_ylabel("TFLOPs"); ax.set_xscale("log", base=2)
    ax.set_title("TFLOPs vs matrix size"); ax.legend()

    ax = axes[0, 1]
    for phase in phases:
        sub = df[df["phase"] == phase].sort_values("mat_size")
        ax.plot(sub["mat_size"], sub["mem_bw_gbs"], marker="o", label=phase)
    ax.set_xlabel("Matrix size (n)"); ax.set_ylabel("Mem BW (GB/s, idealized)"); ax.set_xscale("log", base=2)
    ax.set_title("Memory bandwidth vs matrix size"); ax.legend()

    ax = axes[1, 0]
    for phase in phases:
        sub = df[df["phase"] == phase].sort_values("mat_size")
        ax.plot(sub["mat_size"], sub["time_ms"], marker="o", label=phase)
    ax.set_xlabel("Matrix size (n)"); ax.set_ylabel("Time (ms)")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_title("Time vs matrix size"); ax.legend()

    ax = axes[1, 1]
    if "gemm_pytorch" in phases:
        pt = df[df["phase"] == "gemm_pytorch"].set_index("mat_size")["time_ms"]
        for phase in phases:
            if phase == "gemm_pytorch":
                continue
            sub = df[df["phase"] == phase].sort_values("mat_size")
            speedup = pt.reindex(sub["mat_size"]).values / sub["time_ms"].values
            ax.plot(sub["mat_size"], speedup, marker="o", label=phase)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Matrix size (n)"); ax.set_ylabel("Speedup vs PyTorch")
        ax.set_xscale("log", base=2)
        ax.set_title("Speedup vs PyTorch"); ax.legend()
    else:
        ax.text(0.5, 0.5, "no pytorch baseline", ha="center", va="center")

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
