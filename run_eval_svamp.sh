#!/bin/bash
set -euo pipefail

GPU_IDS="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

usage() {
  cat <<'EOF'
用法:
  bash run_eval_svamp.sh [options] [adapter_path]

说明:
  SVAMP 专用评估脚本。默认评估 SVAMP test split。
  最后一个位置参数是 LoRA checkpoint；加 --no_adapter 则只评估原始 base model。

选项:
  --base_model MODEL_OR_PATH     基座模型，默认 Qwen/Qwen2.5-1.5B-Instruct
  --dataset_path PATH            本地 SVAMP arrow 文件；不传则从 Hugging Face datasets 读取 ChilleD/SVAMP
  --split SPLIT                  train 或 test，默认 test
  --gpu GPU_IDS                  指定显卡，例如 0
  --max_gen_toks N               最大生成 token 数，默认 1536
  --max_samples N                只评前 N 条，默认全量
  --batch_size N                 保留参数，当前逐条生成，默认 1
  --env ENV_NAME                 conda 环境名，默认 grpo_b
  --output_jsonl PATH            保存逐样本预测结果 jsonl
  --prompt_style STYLE           short | fewshot，默认 short
  --no_adapter                   不加载 LoRA adapter
  --no-4bit                      不用 4bit 加载
  --verbose                      打印每条样本详情
  -h, --help                     显示帮助

示例:
  bash run_eval_svamp.sh \
    --base_model /path/to/qwen25-1.5b/snapshot \
    --dataset_path /home/changqingcheng/baseline2.0/svamp-test.arrow \
    --gpu 0 \
    --max_gen_toks 1536 \
    ./grpo_qwen25_15b_svamp_lora_accuracy_gate_1024_1gpu/checkpoint-100
EOF
}

CONDA_ENV_NAME="grpo_b"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
DATASET_PATH="${DATASET_PATH:-}"
SPLIT="test"
MAX_GEN_TOKS="1536"
MAX_SAMPLES="-1"
BATCH_SIZE="1"
OUTPUT_JSONL=""
PROMPT_STYLE="${PROMPT_STYLE:-short}"
ADAPTER_PATH=""
USE_ADAPTER=1
USE_4BIT=1
VERBOSE=0

while [ $# -gt 0 ]; do
  case "$1" in
    --base_model)
      BASE_MODEL="$2"
      shift 2
      ;;
    --dataset_path)
      DATASET_PATH="$2"
      shift 2
      ;;
    --split)
      SPLIT="$2"
      shift 2
      ;;
    --gpu)
      GPU_IDS="$2"
      export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
      shift 2
      ;;
    --max_gen_toks)
      MAX_GEN_TOKS="$2"
      shift 2
      ;;
    --max_samples|--limit)
      MAX_SAMPLES="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --env)
      CONDA_ENV_NAME="$2"
      shift 2
      ;;
    --output_jsonl)
      OUTPUT_JSONL="$2"
      shift 2
      ;;
    --prompt_style)
      PROMPT_STYLE="$2"
      shift 2
      ;;
    --no_adapter)
      USE_ADAPTER=0
      shift
      ;;
    --no-4bit)
      USE_4BIT=0
      shift
      ;;
    --verbose)
      VERBOSE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [ -n "${ADAPTER_PATH}" ]; then
        echo "Error: unexpected extra argument: $1" >&2
        usage
        exit 1
      fi
      ADAPTER_PATH="$1"
      shift
      ;;
  esac
done

if [ "${USE_ADAPTER}" = "1" ] && [ -z "${ADAPTER_PATH}" ]; then
  echo "Error: adapter_path is required unless --no_adapter is set." >&2
  usage
  exit 1
fi

if [ -n "${DATASET_PATH}" ] && [ ! -f "${DATASET_PATH}" ]; then
  echo "Error: dataset path does not exist: ${DATASET_PATH}" >&2
  exit 1
fi

mkdir -p logs
TIME=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/eval_svamp_${TIME}.log"
if [ -z "${OUTPUT_JSONL}" ]; then
  OUTPUT_JSONL="logs/eval_svamp_${TIME}.jsonl"
fi

CMD=(
  conda run
  --no-capture-output
  -n "${CONDA_ENV_NAME}"
  python -u evaluate_svamp.py
  --base_model "${BASE_MODEL}"
  --split "${SPLIT}"
  --max_new_tokens "${MAX_GEN_TOKS}"
  --max_samples "${MAX_SAMPLES}"
  --prompt_style "${PROMPT_STYLE}"
  --report_every 10
  --output_jsonl "${OUTPUT_JSONL}"
)

if [ -n "${DATASET_PATH}" ]; then
  CMD+=(--dataset_path "${DATASET_PATH}")
fi
if [ "${USE_ADAPTER}" = "1" ]; then
  CMD+=(--adapter_path "${ADAPTER_PATH}")
else
  CMD+=(--no_adapter)
fi
if [ "${USE_4BIT}" = "1" ]; then
  CMD+=(--use_4bit)
fi
if [ "${VERBOSE}" = "1" ]; then
  CMD+=(--verbose)
fi

{
  echo "==== SVAMP Evaluation Start ===="
  echo "Time: ${TIME}"
  echo "Conda env: ${CONDA_ENV_NAME}"
  echo "Base model: ${BASE_MODEL}"
  echo "Adapter: $([ "${USE_ADAPTER}" = "1" ] && echo "${ADAPTER_PATH}" || echo disabled)"
  echo "Dataset path: ${DATASET_PATH:-ChilleD/SVAMP:${SPLIT}}"
  echo "GPU: ${GPU_IDS}"
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
  echo "Max gen toks: ${MAX_GEN_TOKS}"
  echo "Max samples: ${MAX_SAMPLES}"
  echo "Batch size argument: ${BATCH_SIZE} (generation is one sample at a time)"
  echo "Prompt style: ${PROMPT_STYLE}"
  echo "4-bit loading: ${USE_4BIT}"
  echo "HF_HOME: ${HF_HOME}"
  echo "HF_HUB_OFFLINE: ${HF_HUB_OFFLINE}"
  echo "TRANSFORMERS_OFFLINE: ${TRANSFORMERS_OFFLINE}"
  echo "HF_DATASETS_OFFLINE: ${HF_DATASETS_OFFLINE}"
  echo "Output jsonl: ${OUTPUT_JSONL}"
  echo "Log file: ${LOG_FILE}"
} | tee "${LOG_FILE}"

"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
