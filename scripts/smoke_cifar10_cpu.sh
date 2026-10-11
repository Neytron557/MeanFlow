#!/usr/bin/env bash
# Check training, sampling and checkpoints with a small synthetic CPU run.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
python_bin="${PYTHON:-$repo_dir/.venv/bin/python}"
exec "$python_bin" train.py --dataset cifar10 \
  --smoke --model DiT-Tiny/4 --device cpu \
  --per-device-batch-size 2 --gradient-accumulation-steps 2 \
  --max-train-steps 3 --warmup-steps 2 --num-workers 0 \
  --checkpointing-steps 1 --log-every 1 --output-dir work_dir/cpu_smoke "$@"
