#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

mkdir -p logs
TIME="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/train_pvar_uid1_0_to_5188_resume_${TIME}.log"
PID_FILE="logs/train_pvar_uid1_0_to_5188_resume.pid"
LATEST_LOG_LINK="logs/train_pvar_uid1_0_to_5188_resume_latest.log"

RUN_MODE="foreground"

usage() {
  cat <<'EOF'
Usage:
  bash run_resume_train_pvar_uid1_0_to_5188_ckpt300.sh [--mode foreground|background]
  bash run_resume_train_pvar_uid1_0_to_5188_ckpt300.sh [--foreground|--background]

Environment overrides:
  RESUME_CKPT
  OUTPUT_DIR
  TRAIN_SCORES_FILE
  MIN_UID1
  MAX_UID1
  PROMPT_STYLE
  PER_DEVICE_TRAIN_BATCH_SIZE
  GRADIENT_ACCUMULATION_STEPS
  NUM_GENERATIONS
  MAX_STEPS
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --mode)
      if [ $# -lt 2 ]; then
        echo "Error: --mode requires foreground or background"
        usage
        exit 1
      fi
      RUN_MODE="$2"
      shift 2
      ;;
    --foreground)
      RUN_MODE="foreground"
      shift
      ;;
    --background)
      RUN_MODE="background"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unexpected argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [ "${RUN_MODE}" != "foreground" ] && [ "${RUN_MODE}" != "background" ]; then
  echo "Error: invalid mode '${RUN_MODE}'"
  usage
  exit 1
fi

TRAIN_SCORES_FILE="${TRAIN_SCORES_FILE:-outputs/gsm8k_deff_scores_k8_t448_tau0.2_20260412_212149_pvar_sorted.jsonl}"
MIN_UID1="${MIN_UID1:-0}"
MAX_UID1="${MAX_UID1:-5188}"
OUTPUT_DIR="${OUTPUT_DIR:-./grpo_qwen25_15b_gsm8k_lora_pvar_uid1_0_to_5188}"
RESUME_CKPT="${RESUME_CKPT:-${OUTPUT_DIR}/checkpoint-300}"
PROMPT_STYLE="${PROMPT_STYLE:-short}"

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
MAX_STEPS="${MAX_STEPS:-2595}"

if [ ! -d "${RESUME_CKPT}" ]; then
  echo "Error: checkpoint not found: ${RESUME_CKPT}"
  exit 1
fi

if [ ! -f "${TRAIN_SCORES_FILE}" ]; then
  echo "Error: train scores file not found: ${TRAIN_SCORES_FILE}"
  exit 1
fi

if [ -f "${PID_FILE}" ]; then
  OLD_PID="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [ -n "${OLD_PID}" ] && kill -0 "${OLD_PID}" 2>/dev/null; then
    echo "A resume training process is already running with PID ${OLD_PID}."
    echo "Log: $(readlink -f "${LATEST_LOG_LINK}" 2>/dev/null || echo "${LATEST_LOG_LINK}")"
    exit 1
  fi
  rm -f "${PID_FILE}"
fi

ln -sfn "$(basename "${LOG_FILE}")" "${LATEST_LOG_LINK}"

CMD=(
  accelerate launch
  --mixed_precision bf16
  --num_processes 1
  train_grpo.py
  --model_name Qwen/Qwen2.5-1.5B-Instruct
  --output_dir "${OUTPUT_DIR}"
  --seed 42
  --train_split train
  --train_samples -1
  --prompt_style "${PROMPT_STYLE}"
  --train_scores_file "${TRAIN_SCORES_FILE}"
  --min_uid1 "${MIN_UID1}"
  --max_uid1 "${MAX_UID1}"
  --max_steps "${MAX_STEPS}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --learning_rate 1e-5
  --weight_decay 0.0
  --warmup_ratio 0.03
  --num_generations "${NUM_GENERATIONS}"
  --max_completion_length 256
  --temperature 1.0
  --top_p 0.95
  --beta 0.0
  --epsilon 0.2
  --logging_steps 1
  --save_steps 100
  --save_total_limit 1000
  --lora_r 16
  --lora_alpha 32
  --lora_dropout 0.05
  --use_4bit
  --resume_from_checkpoint "${RESUME_CKPT}"
)

{
  echo "==== Resume GRPO Training From checkpoint-300 Start ===="
  echo "Time: ${TIME}"
  echo "Mode: ${RUN_MODE}"
  echo "Log: ${LOG_FILE}"
  echo "Output dir: ${OUTPUT_DIR}"
  echo "Resume from checkpoint: ${RESUME_CKPT}"
  echo "Train scores file: ${TRAIN_SCORES_FILE}"
  echo "Min uid1: ${MIN_UID1}"
  echo "Max uid1: ${MAX_UID1}"
  echo "Max steps: ${MAX_STEPS}"
  echo "Prompt style: ${PROMPT_STYLE}"
} | tee "${LOG_FILE}"

if [ "${RUN_MODE}" = "foreground" ]; then
  "${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
  echo "==== Resume Training Finished ====" | tee -a "${LOG_FILE}"
  exit 0
fi

nohup "${CMD[@]}" >> "${LOG_FILE}" 2>&1 &
TRAIN_PID=$!
echo "${TRAIN_PID}" > "${PID_FILE}"

echo "Started background resume training."
echo "PID: ${TRAIN_PID}"
echo "Log: ${LOG_FILE}"
echo "Latest log link: ${LATEST_LOG_LINK}"
echo "Follow log with: tail -f ${LATEST_LOG_LINK}"
