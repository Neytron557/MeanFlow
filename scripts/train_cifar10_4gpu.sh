#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TORCH_HOME="${TORCH_HOME:-$repo_dir/.cache/torch}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
python_bin="${PYTHON:-$repo_dir/.venv/bin/python}"
exec "$python_bin" -m torch.distributed.run \
  --nnodes=1 --nproc_per_node=4 --master-addr=127.0.0.1 \
  --master-port="${MASTER_PORT:-29500}" \
  train.py --dataset cifar10 --config configs/cifar10_4x2080.json "$@"
