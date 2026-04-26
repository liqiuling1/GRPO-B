#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

usage() {
  cat <<'EOF'
用法:
  bash run_lm_eval_gsm8k.sh [options] [adapter_path]

说明:
  使用 lm-evaluation-harness 的 hf backend 评估 GSM8K。
  默认评估任务为 gsm8k_cot，默认 adapter 为当前训练产物目录。

选项:
  --base_model MODEL_ID       基座模型，默认 Qwen/Qwen2.5-1.5B-Instruct
  --task TASK_NAME            评测任务，默认 gsm8k_cot
  --num_fewshot N             few-shot 示例数，默认 8
  --batch_size N             batch size，默认 1
  --max_gen_toks N           最大生成 token 数，默认 512
  --limit N                  只评前 N 条样本，默认全量
  --env ENV_NAME             conda 环境名，默认 grpo_b
  --output_path PATH         lm-eval 结果输出目录，默认 logs/lm_eval_<task>_<time>
  --no_adapter               不加载 LoRA adapter，直接评估 base model
  --no-chat-template         不传 --apply_chat_template
  --no-4bit                  不在 model_args 中传 load_in_4bit=True
  -h, --help                 显示帮助

示例:
  bash run_lm_eval_gsm8k.sh
  bash run_lm_eval_gsm8k.sh ./grpo_qwen25_15b_gsm8k_lora_grpo_baseline_2500_256_advantage_8X1/checkpoint-2500
  bash run_lm_eval_gsm8k.sh --limit 100 --batch_size 2
  bash run_lm_eval_gsm8k.sh --max_gen_toks 512 ./grpo_qwen25_15b_gsm8k_lora_pvar_uid1_0_to_5188/checkpoint-300
EOF
}

CONDA_ENV_NAME="grpo_b"
BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
TASK_NAME="gsm8k_cot"
NUM_FEWSHOT="8"
BATCH_SIZE="1"
MAX_GEN_TOKS="512"
LIMIT=""
OUTPUT_PATH=""
ADAPTER_PATH="./grpo_qwen25_15b_gsm8k_lora_grpo_baseline_2500_256_advantage_8X1"
DEFAULT_ADAPTER_PATH="${ADAPTER_PATH}"
USE_ADAPTER=1
APPLY_CHAT_TEMPLATE=1
USE_4BIT=1

while [ $# -gt 0 ]; do
  case "$1" in
    --base_model)
      if [ $# -lt 2 ]; then
        echo "Error: --base_model requires a value"
        usage
        exit 1
      fi
      BASE_MODEL="$2"
      shift 2
      ;;
    --task)
      if [ $# -lt 2 ]; then
        echo "Error: --task requires a value"
        usage
        exit 1
      fi
      TASK_NAME="$2"
      shift 2
      ;;
    --num_fewshot)
      if [ $# -lt 2 ]; then
        echo "Error: --num_fewshot requires a value"
        usage
        exit 1
      fi
      NUM_FEWSHOT="$2"
      shift 2
      ;;
    --batch_size)
      if [ $# -lt 2 ]; then
        echo "Error: --batch_size requires a value"
        usage
        exit 1
      fi
      BATCH_SIZE="$2"
      shift 2
      ;;
    --max_gen_toks)
      if [ $# -lt 2 ]; then
        echo "Error: --max_gen_toks requires a value"
        usage
        exit 1
      fi
      MAX_GEN_TOKS="$2"
      shift 2
      ;;
    --limit)
      if [ $# -lt 2 ]; then
        echo "Error: --limit requires a value"
        usage
        exit 1
      fi
      LIMIT="$2"
      shift 2
      ;;
    --env)
      if [ $# -lt 2 ]; then
        echo "Error: --env requires a value"
        usage
        exit 1
      fi
      CONDA_ENV_NAME="$2"
      shift 2
      ;;
    --output_path)
      if [ $# -lt 2 ]; then
        echo "Error: --output_path requires a value"
        usage
        exit 1
      fi
      OUTPUT_PATH="$2"
      shift 2
      ;;
    --no_adapter)
      USE_ADAPTER=0
      shift
      ;;
    --no-chat-template)
      APPLY_CHAT_TEMPLATE=0
      shift
      ;;
    --no-4bit)
      USE_4BIT=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [ "${ADAPTER_PATH}" != "${DEFAULT_ADAPTER_PATH}" ]; then
        echo "Error: unexpected extra argument: $1"
        usage
        exit 1
      fi
      ADAPTER_PATH="$1"
      shift
      ;;
  esac
done

mkdir -p logs
TIME=$(date +"%Y%m%d_%H%M%S")
if [ -z "${OUTPUT_PATH}" ]; then
  OUTPUT_PATH="logs/lm_eval_${TASK_NAME}_${TIME}"
fi
LOG_FILE="logs/lm_eval_${TASK_NAME}_${TIME}.log"

if ! conda run --no-capture-output -n "${CONDA_ENV_NAME}" python -c \
  "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('lm_eval') else 1)" \
  >/dev/null 2>&1; then
  echo "Error: lm-eval is not installed in conda env '${CONDA_ENV_NAME}'."
  echo "Install with:"
  echo "  conda run -n ${CONDA_ENV_NAME} pip install \"lm_eval[hf]\""
  exit 1
fi

TRANSFORMERS_VERSION=$(conda run --no-capture-output -n "${CONDA_ENV_NAME}" python -c \
  "import transformers; print(transformers.__version__)" 2>/dev/null || echo "unknown")

RESOLVED_BASE_MODEL=$(conda run --no-capture-output -n "${CONDA_ENV_NAME}" python - <<PY
from model_utils import resolve_cached_model_path
print(resolve_cached_model_path(${BASE_MODEL@Q}))
PY
)

if [ -z "${RESOLVED_BASE_MODEL}" ]; then
  echo "Error: failed to resolve base model path for ${BASE_MODEL}"
  exit 1
fi

BASE_MODEL_VERSION=$(basename "${RESOLVED_BASE_MODEL}")
ADAPTER_VERSION=""
ADAPTER_CHECKPOINT=""
if [ "${USE_ADAPTER}" = "1" ]; then
  ADAPTER_BASENAME=$(basename "${ADAPTER_PATH}")
  if [[ "${ADAPTER_BASENAME}" == checkpoint-* ]]; then
    ADAPTER_CHECKPOINT="${ADAPTER_BASENAME}"
    ADAPTER_VERSION=$(basename "$(dirname "${ADAPTER_PATH}")")
  else
    ADAPTER_VERSION="${ADAPTER_BASENAME}"
  fi
fi

LM_EVAL_ARGS=()
if conda run --no-capture-output -n "${CONDA_ENV_NAME}" python -m lm_eval run -h >/dev/null 2>&1; then
  LM_EVAL_ARGS=(run)
fi

if [ "${USE_4BIT}" = "1" ]; then
  TRANSFORMERS_MAJOR="${TRANSFORMERS_VERSION%%.*}"
  if [ "${TRANSFORMERS_MAJOR}" = "5" ]; then
    echo "Warning: transformers ${TRANSFORMERS_VERSION} with this lm-eval backend is incompatible with load_in_4bit=True."
    echo "Warning: auto-disabling 4-bit loading for lm-eval."
    USE_4BIT=0
  fi
fi

MODEL_ARGS="pretrained=${RESOLVED_BASE_MODEL},peft=${ADAPTER_PATH},trust_remote_code=True"
if [ "${USE_ADAPTER}" = "0" ]; then
  MODEL_ARGS="pretrained=${RESOLVED_BASE_MODEL},trust_remote_code=True"
fi
if [ "${USE_4BIT}" = "1" ]; then
  MODEL_ARGS="${MODEL_ARGS},load_in_4bit=True"
fi

CMD=(
  conda run
  --no-capture-output
  -n "${CONDA_ENV_NAME}"
  python -m lm_eval
  "${LM_EVAL_ARGS[@]}"
  --model hf
  --model_args "${MODEL_ARGS}"
  --tasks "${TASK_NAME}"
  --device cuda:0
  --num_fewshot "${NUM_FEWSHOT}"
  --batch_size "${BATCH_SIZE}"
  --gen_kwargs "max_gen_toks=${MAX_GEN_TOKS}"
  --output_path "${OUTPUT_PATH}"
  --log_samples
)

if [ "${APPLY_CHAT_TEMPLATE}" = "1" ]; then
  CMD+=(--apply_chat_template)
fi

if [ -n "${LIMIT}" ]; then
  CMD+=(--limit "${LIMIT}")
fi

{
  echo "==== lm-eval GSM8K Start ===="
  echo "Time: ${TIME}"
  echo "Conda env: ${CONDA_ENV_NAME}"
  echo "Transformers: ${TRANSFORMERS_VERSION}"
  echo "Base model: ${BASE_MODEL}"
  echo "Resolved base model: ${RESOLVED_BASE_MODEL}"
  echo "Base model version: ${BASE_MODEL_VERSION}"
  if [ "${USE_ADAPTER}" = "1" ]; then
    echo "Adapter: ${ADAPTER_PATH}"
    echo "Adapter version: ${ADAPTER_VERSION}"
    if [ -n "${ADAPTER_CHECKPOINT}" ]; then
      echo "Checkpoint: ${ADAPTER_CHECKPOINT}"
    else
      echo "Checkpoint: <none>"
    fi
  else
    echo "Adapter: disabled"
  fi
  echo "Model args: ${MODEL_ARGS}"
  echo "Task: ${TASK_NAME}"
  echo "Num fewshot: ${NUM_FEWSHOT}"
  echo "Batch size: ${BATCH_SIZE}"
  echo "Max gen toks: ${MAX_GEN_TOKS}"
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
  echo "HF_HOME: ${HF_HOME}"
  echo "HF_HUB_OFFLINE: ${HF_HUB_OFFLINE}"
  echo "TRANSFORMERS_OFFLINE: ${TRANSFORMERS_OFFLINE}"
  echo "HF_DATASETS_OFFLINE: ${HF_DATASETS_OFFLINE}"
  echo "Output path: ${OUTPUT_PATH}"
  echo "Log file: ${LOG_FILE}"

  if [ -n "${LIMIT}" ]; then
    echo "Limit: ${LIMIT}"
  else
    echo "Limit: full dataset"
  fi

  if [ "${APPLY_CHAT_TEMPLATE}" = "1" ]; then
    echo "Chat template: enabled"
  else
    echo "Chat template: disabled"
  fi

  if [ "${USE_4BIT}" = "1" ]; then
    echo "4-bit loading: enabled"
  else
    echo "4-bit loading: disabled"
  fi

  "${CMD[@]}"

  echo "==== lm-eval GSM8K Finished ===="
} 2>&1 | tee "${LOG_FILE}"
