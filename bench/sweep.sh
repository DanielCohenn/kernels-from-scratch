#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# --skip-validate: skip hostNaiveGemm's O(m*n*k) single-thread CPU reference per run (~9s at
# just 1024^3, scales cubically). Only use once correctness is already confirmed for these
# kernels at representative sizes — it trusts the last real check instead of re-running it.
SKIP_VALIDATE=0
if [ "${1:-}" = "--skip-validate" ]; then
    SKIP_VALIDATE=1
fi

python3 build.py

SIZES=(128 192 256 384 512 640 768 896 1024 1280 1536 1792 2048 2560 3072 3584 4096)

for bin in build/gemm/*; do
    [ -x "$bin" ] || continue
    for s in "${SIZES[@]}"; do
        echo "== $bin size=$s =="
        if [ "$SKIP_VALIDATE" = "1" ]; then
            GEMM_SKIP_VALIDATE=1 "$bin" "$s" "$s" "$s"
        else
            "$bin" "$s" "$s" "$s"
        fi
    done
done

echo "Results appended to bench/gemm_results.csv"
