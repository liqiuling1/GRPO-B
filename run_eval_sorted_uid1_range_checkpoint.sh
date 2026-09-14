#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-grpo_b}"

exec conda run --no-capture-output -n "${CONDA_ENV_NAME}" \
  python -u eval_sorted_uid1_range_checkpoint.py "$@"
