#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

make

SIZES=(128 256 384 512 768 1024 1536 2048 3072 4096 6144 8192)

for bin in build/gemm/*; do
    [ -x "$bin" ] || continue
    for s in "${SIZES[@]}"; do
        echo "== $bin size=$s =="
        "$bin" "$s" "$s" "$s"
    done
done

echo "Results appended to bench/gemm_results.csv"
