#!/bin/bash
set -euo pipefail

# Train on SVAMP while reusing the existing GSM8K-oriented training pipeline.
# The current train_grpo.py expects local arrow rows with "question" and "answer".
# SVAMP uses "question_concat" and "Answer", so this script materializes a small
# compatible arrow file before delegating to run_train.sh.

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

SVAMP_SOURCE="${SVAMP_SOURCE:-ChilleD/SVAMP}"
SVAMP_SPLIT="${SVAMP_SPLIT:-train}"
SVAMP_ARROW="${SVAMP_ARROW:-}"
CONVERTED_DATASET_PATH="${CONVERTED_DATASET_PATH:-data/svamp-${SVAMP_SPLIT}-grpo.arrow}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-./grpo_qwen25_15b_svamp_lora_grpo_baseline}"
PREPARE_ONLY=0

usage() {
  cat <<'EOF'
用法:
  bash run_train_svamp.sh [run_train.sh options]
  bash run_train_svamp.sh --prepare_only

说明:
  这个脚本专门用于 SVAMP 训练。
  它会先把 SVAMP 转成现有 train_grpo.py 需要的 question/answer arrow 格式，
  然后把其余参数原样转交给 run_train.sh。

常用示例:
  bash run_train_svamp.sh

  bash run_train_svamp.sh \
    --base_model Qwen/Qwen2.5-1.5B-Instruct \
    --output_dir ./grpo_qwen25_15b_svamp_lora_grpo_baseline

  CUDA_VISIBLE_DEVICES=0 NUM_PROCESSES=1 bash run_train_svamp.sh \
    --max_completion_length 1024 \
    --batch_size 8 \
    --grad_acc 2 \
    --num_generations 16 \
    --save_steps 10

环境变量:
  SVAMP_ARROW              显式指定原始 SVAMP arrow 文件；不填则用 Hugging Face datasets 加载 ChilleD/SVAMP
  SVAMP_SPLIT              train 或 test，默认 train
  CONVERTED_DATASET_PATH   转换后的 arrow 输出路径，默认 data/svamp-train-grpo.arrow
  BASE_MODEL               默认 Qwen/Qwen2.5-1.5B-Instruct
  OUTPUT_DIR               默认 ./grpo_qwen25_15b_svamp_lora_grpo_baseline
EOF
}

PASSTHROUGH_ARGS=()
for arg in "$@"; do
  case "${arg}" in
    -h|--help)
      usage
      exit 0
      ;;
    --prepare_only)
      PREPARE_ONLY=1
      ;;
    *)
      PASSTHROUGH_ARGS+=("${arg}")
      ;;
  esac
done

python prepare_svamp_grpo_arrow.py \
  --source "${SVAMP_SOURCE}" \
  --split "${SVAMP_SPLIT}" \
  --svamp_arrow "${SVAMP_ARROW}" \
  --output "${CONVERTED_DATASET_PATH}"

if [ "${PREPARE_ONLY}" = "1" ]; then
  exit 0
fi

exec bash run_train.sh \
  --base_model "${BASE_MODEL}" \
  --dataset_path "${CONVERTED_DATASET_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  "${PASSTHROUGH_ARGS[@]}"
