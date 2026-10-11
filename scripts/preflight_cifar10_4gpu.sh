#!/usr/bin/env bash
# Check the production model and optimizer with synthetic data on four GPUs.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$repo_dir/scripts/train_cifar10_4gpu.sh" \
  --smoke --max-train-steps 3 --checkpointing-steps 1 --log-every 1 \
  --num-workers 0 --output-dir work_dir/preflight_4gpu \
  --stop-file work_dir/preflight_4gpu/STOP "$@"
