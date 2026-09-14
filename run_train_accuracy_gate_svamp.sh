#!/bin/bash
set -euo pipefail

# Accuracy-gated GRPO training on SVAMP.
# SVAMP is converted to the existing question/answer arrow format first, then
# all training arguments are delegated to run_train_accuracy_gate.sh.

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

SVAMP_SOURCE="${SVAMP_SOURCE:-ChilleD/SVAMP}"
SVAMP_SPLIT="${SVAMP_SPLIT:-train}"
SVAMP_ARROW="${SVAMP_ARROW:-}"
CONVERTED_DATASET_PATH="${CONVERTED_DATASET_PATH:-data/svamp-${SVAMP_SPLIT}-grpo.arrow}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-./grpo_qwen25_15b_svamp_lora_accuracy_gate}"
PREPARE_ONLY=0

usage() {
  cat <<'EOF'
用法:
  bash run_train_accuracy_gate_svamp.sh [run_train_accuracy_gate.sh options]
  bash run_train_accuracy_gate_svamp.sh --prepare_only

说明:
  这个脚本用于 SVAMP + accuracy gate 训练。
  它会先把 SVAMP 转成现有训练代码需要的 question/answer arrow 格式，
  然后调用 run_train_accuracy_gate.sh。
  run_train_accuracy_gate.sh 的训练参数会原样透传，包括截断补采样、accepted 长度过滤和目标区间长度 reward 参数。

示例:
  CUDA_VISIBLE_DEVICES=0 \
  NUM_PROCESSES=1 \
  MAX_COMPLETION_LENGTH=1024 \
  PROMPT_STYLE=short \
  bash run_train_accuracy_gate_svamp.sh \
    --base_model Qwen/Qwen2.5-1.5B-Instruct \
    --output_dir ./grpo_qwen25_15b_svamp_lora_accuracy_gate_1024_1gpu \
    --min_correct_rate 0.375 \
    --max_correct_rate 0.625 \
    --target_update_steps 100 \
    --max_attempt_steps 10000 \
    --batch_size 8 \
    --grad_acc 2 \
    --num_generations 16 \
    --max_completion_length 1024 \
    --accepted_length_min 40 \
    --accepted_length_max 500 \
    --length_reward_weight 0.2 \
    --length_reward_good_min 40 \
    --length_reward_good_max 300 \
    --length_reward_short_min 0 \
    --length_reward_max_penalty 1.0 \
    --rejected_length_only_update 0 \
    --max_retry_truncated_count 6 \
    --max_truncation_retry_rounds 3 \
    --save_steps 10

环境变量:
  SVAMP_ARROW              显式指定原始 SVAMP arrow 文件；不填则用 Hugging Face datasets 加载 ChilleD/SVAMP
  SVAMP_SPLIT              train 或 test，默认 train
  CONVERTED_DATASET_PATH   转换后的 arrow 输出路径，默认 data/svamp-train-grpo.arrow
  MAX_RETRY_TRUNCATED_COUNT       默认 6；也可以用 --max_retry_truncated_count 覆盖
  MAX_TRUNCATION_RETRY_ROUNDS     默认 3；也可以用 --max_truncation_retry_rounds 覆盖
  ACCEPTED_LENGTH_MIN             默认 0；也可以用 --accepted_length_min 覆盖
  ACCEPTED_LENGTH_MAX             默认关闭；也可以用 --accepted_length_max 覆盖
  LENGTH_REWARD_WEIGHT            默认 0；也可以用 --length_reward_weight 覆盖，设 0 关闭
  LENGTH_REWARD_GOOD_MIN          默认 140；也可以用 --length_reward_good_min 覆盖
  LENGTH_REWARD_GOOD_MAX          默认 280；也可以用 --length_reward_good_max 覆盖
  LENGTH_REWARD_SHORT_MIN         默认 0；也可以用 --length_reward_short_min 覆盖
  LENGTH_REWARD_MAX_PENALTY       默认 1.0；也可以用 --length_reward_max_penalty 覆盖
  REJECTED_LENGTH_ONLY_UPDATE     默认 0；也可以用 --rejected_length_only_update 覆盖
  REJECTED_LENGTH_ONLY_ADVANTAGE_SCALE  默认 length_reward_weight；也可以用 --rejected_length_only_advantage_scale 覆盖
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

exec bash run_train_accuracy_gate.sh \
  --base_model "${BASE_MODEL}" \
  --dataset_path "${CONVERTED_DATASET_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  "${PASSTHROUGH_ARGS[@]}"
