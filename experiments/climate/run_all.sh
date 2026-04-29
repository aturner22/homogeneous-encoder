#!/usr/bin/env bash
# Loop run.py over every univariate tensor plus the joint tensor.
# Per-variable hyperparameters live in config.yaml's per_variable: block.
set -euo pipefail
cd "$(dirname "$0")"

for var in u10 tp t2m tensor; do
    echo "=== $var ==="
    python run.py --config config.yaml --var "$var"
done
