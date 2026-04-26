#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"

mkdir -p logs
TIME=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/train_${TIME}.log"
PID_FILE="logs/train_latest.pid"
LATEST_LOG_LINK="logs/train_latest.log"

usage() {
  cat <<'EOF'
用法:
  bash run_train.sh [--mode foreground|background] [options] [checkpoint_path]
  bash run_train.sh [--foreground|--background] [options] [checkpoint_path]

选项:
  --base_model MODEL_OR_PATH 基座模型名称或本地路径
  --dataset_path PATH        显式指定 GSM8K train.arrow 路径
  --train_scores_file PATH   评分排序文件 jsonl 路径
  --scores_file PATH         同 --train_scores_file
  --min_uid1 N               仅训练 uid1 >= N 的样本
  --start_uid1 N             同 --min_uid1
  --max_uid1 N               仅训练 uid1 <= N 的样本
  --end_uid1 N               同 --max_uid1
  --output_dir PATH          输出目录
  --max_steps N              最大训练步数；不传时子集任务会自动计算
  --batch_size N             per_device_train_batch_size
  --grad_acc N               gradient_accumulation_steps
  --num_generations N        每个 prompt 的生成数量
  --init_adapter_path PATH   用已有 LoRA adapter 初始化训练
  --prompt_style STYLE       short | fewshot

示例:
  bash run_train.sh
  bash run_train.sh --scores_file outputs/xxx_sorted.jsonl --start_uid1 2287 --output_dir ./my_run
  bash run_train.sh --scores_file outputs/xxx_sorted.jsonl --start_uid1 0 --end_uid1 5188 --output_dir ./my_run
  bash run_train.sh --base_model /abs/path/to/model_snapshot
  bash run_train.sh --dataset_path /abs/path/to/gsm8k-train.arrow
  bash run_train.sh --train_scores_file outputs/xxx.jsonl --min_uid1 1000 --max_uid1 2000 --output_dir ./my_run
  bash run_train.sh --mode foreground
  bash run_train.sh --mode background
  bash run_train.sh --mode background ./grpo_qwen25_15b_gsm8k_lora_grpo_baseline/checkpoint-500
EOF
}

RUN_MODE="foreground"
RESUME_CKPT=""
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
DATASET_PATH="${DATASET_PATH:-}"
BASE_MODEL_SOURCE="local-cache-or-remote-fallback"
DATASET_SOURCE="local-cache-or-remote-fallback"
TRAIN_SCORES_FILE="${TRAIN_SCORES_FILE:-}"
MIN_UID1="${MIN_UID1:-}"
MAX_UID1="${MAX_UID1:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
MAX_STEPS="${MAX_STEPS:-}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-}"
NUM_GENERATIONS="${NUM_GENERATIONS:-}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-}"
INIT_ADAPTER_PATH="${INIT_ADAPTER_PATH:-}"
PROMPT_STYLE="${PROMPT_STYLE:-}"

while [ $# -gt 0 ]; do
  case "$1" in
    --mode)
      if [ $# -lt 2 ]; then
        echo "Error: --mode requires a value: foreground or background"
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
    --base_model)
      if [ $# -lt 2 ]; then
        echo "Error: --base_model requires a value"
        usage
        exit 1
      fi
      BASE_MODEL="$2"
      BASE_MODEL_SOURCE="user-specified"
      shift 2
      ;;
    --dataset_path)
      if [ $# -lt 2 ]; then
        echo "Error: --dataset_path requires a value"
        usage
        exit 1
      fi
      DATASET_PATH="$2"
      DATASET_SOURCE="user-specified"
      shift 2
      ;;
    --train_scores_file)
      if [ $# -lt 2 ]; then
        echo "Error: --train_scores_file requires a value"
        usage
        exit 1
      fi
      TRAIN_SCORES_FILE="$2"
      shift 2
      ;;
    --scores_file)
      if [ $# -lt 2 ]; then
        echo "Error: --scores_file requires a value"
        usage
        exit 1
      fi
      TRAIN_SCORES_FILE="$2"
      shift 2
      ;;
    --min_uid1)
      if [ $# -lt 2 ]; then
        echo "Error: --min_uid1 requires a value"
        usage
        exit 1
      fi
      MIN_UID1="$2"
      shift 2
      ;;
    --start_uid1)
      if [ $# -lt 2 ]; then
        echo "Error: --start_uid1 requires a value"
        usage
        exit 1
      fi
      MIN_UID1="$2"
      shift 2
      ;;
    --max_uid1)
      if [ $# -lt 2 ]; then
        echo "Error: --max_uid1 requires a value"
        usage
        exit 1
      fi
      MAX_UID1="$2"
      shift 2
      ;;
    --end_uid1)
      if [ $# -lt 2 ]; then
        echo "Error: --end_uid1 requires a value"
        usage
        exit 1
      fi
      MAX_UID1="$2"
      shift 2
      ;;
    --output_dir)
      if [ $# -lt 2 ]; then
        echo "Error: --output_dir requires a value"
        usage
        exit 1
      fi
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --max_steps)
      if [ $# -lt 2 ]; then
        echo "Error: --max_steps requires a value"
        usage
        exit 1
      fi
      MAX_STEPS="$2"
      shift 2
      ;;
    --batch_size)
      if [ $# -lt 2 ]; then
        echo "Error: --batch_size requires a value"
        usage
        exit 1
      fi
      PER_DEVICE_TRAIN_BATCH_SIZE="$2"
      shift 2
      ;;
    --grad_acc)
      if [ $# -lt 2 ]; then
        echo "Error: --grad_acc requires a value"
        usage
        exit 1
      fi
      GRADIENT_ACCUMULATION_STEPS="$2"
      shift 2
      ;;
    --num_generations)
      if [ $# -lt 2 ]; then
        echo "Error: --num_generations requires a value"
        usage
        exit 1
      fi
      NUM_GENERATIONS="$2"
      shift 2
      ;;
    --init_adapter_path)
      if [ $# -lt 2 ]; then
        echo "Error: --init_adapter_path requires a value"
        usage
        exit 1
      fi
      INIT_ADAPTER_PATH="$2"
      shift 2
      ;;
    --prompt_style)
      if [ $# -lt 2 ]; then
        echo "Error: --prompt_style requires a value"
        usage
        exit 1
      fi
      PROMPT_STYLE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [ -n "${RESUME_CKPT}" ]; then
        echo "Error: unexpected extra argument: $1"
        usage
        exit 1
      fi
      RESUME_CKPT="$1"
      shift
      ;;
  esac
done

PROMPT_STYLE="${PROMPT_STYLE:-short}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-448}"
OUTPUT_DIR="${OUTPUT_DIR:-./grpo_qwen25_15b_gsm8k_lora_grpo_baseline}"

if [ "${RUN_MODE}" != "foreground" ] && [ "${RUN_MODE}" != "background" ]; then
  echo "Error: invalid mode '${RUN_MODE}', expected 'foreground' or 'background'"
  usage
  exit 1
fi

if [ -n "${DATASET_PATH}" ] && [ ! -f "${DATASET_PATH}" ]; then
  echo "Error: dataset path does not exist: ${DATASET_PATH}" >&2
  exit 1
fi

if [ -n "${TRAIN_SCORES_FILE}" ] && [ ! -f "${TRAIN_SCORES_FILE}" ]; then
  echo "Error: train scores file does not exist: ${TRAIN_SCORES_FILE}" >&2
  exit 1
fi

if [ -n "${INIT_ADAPTER_PATH}" ] && [ ! -d "${INIT_ADAPTER_PATH}" ]; then
  echo "Error: init adapter path does not exist: ${INIT_ADAPTER_PATH}" >&2
  exit 1
fi

RESOLVED_BASE_MODEL=$(python - <<PY
from model_utils import resolve_cached_model_path
print(resolve_cached_model_path(${BASE_MODEL@Q}))
PY
)

if [ -z "${RESOLVED_BASE_MODEL}" ]; then
  echo "Error: failed to resolve base model path for ${BASE_MODEL}" >&2
  exit 1
fi

if [ "${BASE_MODEL_SOURCE}" != "user-specified" ]; then
  if [ -d "${RESOLVED_BASE_MODEL}" ]; then
    BASE_MODEL_SOURCE="local-cache"
  else
    BASE_MODEL_SOURCE="remote-download-fallback"
  fi
fi

RESOLVED_DATASET_PATH=""
if [ -n "${DATASET_PATH}" ]; then
  RESOLVED_DATASET_PATH="${DATASET_PATH}"
else
  RESOLVED_DATASET_PATH=$(python - <<'PY'
from dataset_utils import find_local_gsm8k_arrow
print(find_local_gsm8k_arrow("train") or "")
PY
)
  if [ -n "${RESOLVED_DATASET_PATH}" ]; then
    DATASET_SOURCE="local-cache"
  else
    DATASET_SOURCE="remote-download-fallback"
  fi
fi

SAMPLE_COUNT=""
if [ -n "${TRAIN_SCORES_FILE}" ] && [ -z "${MAX_STEPS}" ]; then
  SAMPLE_COUNT=$(python - <<PY
import json
from pathlib import Path

path = Path(${TRAIN_SCORES_FILE@Q})
min_uid1 = ${MIN_UID1:-None}
max_uid1 = ${MAX_UID1:-None}
count = 0
with path.open("r", encoding="utf-8") as handle:
    for line in handle:
        text = line.strip()
        if not text:
            continue
        row = json.loads(text)
        if min_uid1 is not None or max_uid1 is not None:
            uid1 = int(row["uid1"])
            if min_uid1 is not None and uid1 < min_uid1:
                continue
            if max_uid1 is not None and uid1 > max_uid1:
                continue
        count += 1
print(count)
PY
)

  if [ -n "${SAMPLE_COUNT}" ] && [ "${SAMPLE_COUNT}" -gt 0 ]; then
    PROMPTS_PER_STEP=$(((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS) / NUM_GENERATIONS))
    MAX_STEPS=$(((SAMPLE_COUNT + PROMPTS_PER_STEP - 1) / PROMPTS_PER_STEP))
  fi
fi

MAX_STEPS="${MAX_STEPS:-3737}"

RESUME_ARG=()
if [ -n "${RESUME_CKPT}" ]; then
  RESUME_ARG=(--resume_from_checkpoint "${RESUME_CKPT}")
fi

DATASET_ARG=()
if [ -n "${RESOLVED_DATASET_PATH}" ]; then
  DATASET_ARG=(--dataset_path "${RESOLVED_DATASET_PATH}")
fi
TRAIN_SCORES_ARG=()
if [ -n "${TRAIN_SCORES_FILE}" ]; then
  TRAIN_SCORES_ARG=(--train_scores_file "${TRAIN_SCORES_FILE}")
fi
MIN_UID1_ARG=()
if [ -n "${MIN_UID1}" ]; then
  MIN_UID1_ARG=(--min_uid1 "${MIN_UID1}")
fi
MAX_UID1_ARG=()
if [ -n "${MAX_UID1}" ]; then
  MAX_UID1_ARG=(--max_uid1 "${MAX_UID1}")
fi
INIT_ADAPTER_ARG=()
if [ -n "${INIT_ADAPTER_PATH}" ]; then
  INIT_ADAPTER_ARG=(--init_adapter_path "${INIT_ADAPTER_PATH}")
fi

TRAINING_MODE="fresh_lora"
if [ -n "${RESUME_CKPT}" ]; then
  TRAINING_MODE="resume_checkpoint"
elif [ -n "${INIT_ADAPTER_PATH}" ]; then
  TRAINING_MODE="init_from_lora"
fi

print_training_summary() {
  echo "==== TRL GRPO Unified Training Start ===="
  echo "Time: ${TIME}"
  echo "Log: ${LOG_FILE}"
  echo "Training mode: ${TRAINING_MODE}"
  echo "HF_HOME: ${HF_HOME}"
  echo "Mode: ${RUN_MODE}"
  echo "Train split: train"
  echo "Base model (${BASE_MODEL_SOURCE}): ${BASE_MODEL}"
  echo "Resolved base model: ${RESOLVED_BASE_MODEL}"
  if [ -n "${RESOLVED_DATASET_PATH}" ]; then
    echo "Dataset (${DATASET_SOURCE}): ${RESOLVED_DATASET_PATH}"
  else
    echo "Dataset (${DATASET_SOURCE}): gsm8k train split (load by datasets cache/download)"
  fi
  if [ -n "${TRAIN_SCORES_FILE}" ]; then
    echo "Train scores file: ${TRAIN_SCORES_FILE}"
  else
    echo "Train scores file: none (full dataset order)"
  fi
  if [ -n "${MIN_UID1}" ]; then
    echo "Start uid1: ${MIN_UID1}"
  else
    echo "Start uid1: none"
  fi
  if [ -n "${MAX_UID1}" ]; then
    echo "End uid1: ${MAX_UID1}"
  else
    echo "End uid1: none"
  fi
  if [ -n "${SAMPLE_COUNT}" ]; then
    echo "Sample count: ${SAMPLE_COUNT}"
  else
    echo "Sample count: auto/full dataset"
  fi
  if [ -n "${INIT_ADAPTER_PATH}" ]; then
    echo "Init adapter path: ${INIT_ADAPTER_PATH}"
  else
    echo "Init adapter path: none"
  fi
  if [ -n "${RESUME_CKPT}" ]; then
    echo "Resume from checkpoint: ${RESUME_CKPT}"
  else
    echo "Resume from checkpoint: none"
  fi
  echo "Output dir: ${OUTPUT_DIR}"
  echo "Prompt style version: ${PROMPT_STYLE}"
  echo "Prompt template: ${PROMPT_STYLE}"
  echo "Per-device batch size: ${PER_DEVICE_TRAIN_BATCH_SIZE}"
  echo "Gradient accumulation steps: ${GRADIENT_ACCUMULATION_STEPS}"
  echo "Num generations: ${NUM_GENERATIONS}"
  echo "Max completion length: ${MAX_COMPLETION_LENGTH}"
  echo "Max steps: ${MAX_STEPS}"
  echo "HF_HUB_OFFLINE: ${HF_HUB_OFFLINE}"
  echo "TRANSFORMERS_OFFLINE: ${TRANSFORMERS_OFFLINE}"
  echo "HF_DATASETS_OFFLINE: ${HF_DATASETS_OFFLINE}"
}

print_training_summary

if [ -f "${PID_FILE}" ]; then
  OLD_PID=$(cat "${PID_FILE}" 2>/dev/null || true)
  if [ -n "${OLD_PID}" ] && kill -0 "${OLD_PID}" 2>/dev/null; then
    echo "A training process is already running with PID ${OLD_PID}."
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
  --model_name "${BASE_MODEL}"
  --output_dir "${OUTPUT_DIR}"
  --seed 42
  --train_split train
  "${DATASET_ARG[@]}"
  --train_samples -1
  --prompt_style "${PROMPT_STYLE}"
  "${TRAIN_SCORES_ARG[@]}"
  "${MIN_UID1_ARG[@]}"
  "${MAX_UID1_ARG[@]}"
  --max_steps "${MAX_STEPS}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --learning_rate 1e-5
  --weight_decay 0.0
  --warmup_ratio 0.03
  --num_generations "${NUM_GENERATIONS}"
  --max_completion_length "${MAX_COMPLETION_LENGTH}"
  --temperature 1.0
  --top_p 0.95
  --beta 0.0
  --epsilon 0.2
  --logging_steps 1
  --save_steps 25
  --save_total_limit 1000
  --lora_r 16
  --lora_alpha 32
  --lora_dropout 0.05
  --use_4bit
  "${INIT_ADAPTER_ARG[@]}"
  "${RESUME_ARG[@]}"
)

if [ "${RUN_MODE}" = "foreground" ]; then
  "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
  echo "==== Training Finished ===="
  echo "Output dir: ${OUTPUT_DIR}"
  exit 0
fi

print_training_summary > "${LOG_FILE}"

nohup "${CMD[@]}" >> "${LOG_FILE}" 2>&1 &
TRAIN_PID=$!
echo "${TRAIN_PID}" > "${PID_FILE}"

echo "Started background training."
echo "PID: ${TRAIN_PID}"
echo "Log: ${LOG_FILE}"
echo "Latest log link: ${LATEST_LOG_LINK}"
echo "Follow log with: tail -f ${LATEST_LOG_LINK}"
echo "Stop with: kill ${TRAIN_PID}"
