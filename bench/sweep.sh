#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

make

SIZES=(128 192 256 384 512 640 768 896 1024 1280 1536 1792 2048 2560 3072 3584 4096)

for bin in build/gemm/*; do
    [ -x "$bin" ] || continue
    for s in "${SIZES[@]}"; do
        echo "== $bin size=$s =="
        "$bin" "$s" "$s" "$s"
    done
done

echo "Results appended to bench/gemm_results.csv"
