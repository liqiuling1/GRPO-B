#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BUNDLE_ROOT="${BUNDLE_ROOT:-$HOME/offline_bundle}"
BUNDLE_NAME="${BUNDLE_NAME:-grpo_offline_bundle_${TIMESTAMP}}"
STAGING_DIR="${BUNDLE_ROOT}/${BUNDLE_NAME}"
ARCHIVE_PATH="${BUNDLE_ROOT}/${BUNDLE_NAME}.tar"

HF_HOME_DEFAULT="${HF_HOME:-$HOME/.cache/huggingface}"
HF_HOME_RESOLVED=$(realpath "${HF_HOME_DEFAULT}")

ENV_NAME="${ENV_NAME:-grpo_b}"
ENV_DIR="${ENV_DIR:-$HOME/miniconda3/envs/${ENV_NAME}}"
INCLUDE_ENV="${INCLUDE_ENV:-1}"

BASE_MODEL_REPO="${BASE_MODEL_REPO:-models--Qwen--Qwen2.5-1.5B-Instruct}"

DEFAULT_ADAPTER_NAME="grpo_qwen25_15b_gsm8k_lora_grpo_baseline_2500_256_advantage_8X1"
FALLBACK_ADAPTER_NAME="grpo_qwen25_15b_gsm8k_lora_grpo_baseline_2500_256_advantage_8X1_没有提示词"
ADAPTER_PATH="${ADAPTER_PATH:-}"
INCLUDE_ADAPTER="${INCLUDE_ADAPTER:-1}"

CODE_FILES=(
  "run_lm_eval_gsm8k.sh"
  "run_train.sh"
  "run_smoke_test1.sh"
  "train_grpo.py"
  "dataset_utils.py"
  "model_utils.py"
  "reward_utils.py"
  "instrumented_grpo_trainer.py"
  "requirements_exact.txt"
)

usage() {
  cat <<EOF
Usage:
  bash prepare_offline_bundle.sh [options]

Options:
  --bundle-root PATH          Output directory for the bundle (default: ${BUNDLE_ROOT})
  --bundle-name NAME          Bundle directory / archive name
  --hf-home PATH              Hugging Face cache root (default: ${HF_HOME_RESOLVED})
  --env-name NAME             Conda env name to pack (default: ${ENV_NAME})
  --env-dir PATH              Conda env directory to pack
  --no-env                    Do not pack the conda environment
  --adapter PATH              Adapter directory to include
  --no-adapter                Do not include any adapter
  -h, --help                  Show this help

Environment overrides:
  BUNDLE_ROOT, BUNDLE_NAME, HF_HOME, ENV_NAME, ENV_DIR, INCLUDE_ENV,
  ADAPTER_PATH, INCLUDE_ADAPTER
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --bundle-root)
      BUNDLE_ROOT="$2"
      STAGING_DIR="${BUNDLE_ROOT}/${BUNDLE_NAME}"
      ARCHIVE_PATH="${BUNDLE_ROOT}/${BUNDLE_NAME}.tar"
      shift 2
      ;;
    --bundle-name)
      BUNDLE_NAME="$2"
      STAGING_DIR="${BUNDLE_ROOT}/${BUNDLE_NAME}"
      ARCHIVE_PATH="${BUNDLE_ROOT}/${BUNDLE_NAME}.tar"
      shift 2
      ;;
    --hf-home)
      HF_HOME_RESOLVED=$(realpath "$2")
      shift 2
      ;;
    --env-name)
      ENV_NAME="$2"
      ENV_DIR="$HOME/miniconda3/envs/${ENV_NAME}"
      shift 2
      ;;
    --env-dir)
      ENV_DIR="$2"
      shift 2
      ;;
    --no-env)
      INCLUDE_ENV=0
      shift
      ;;
    --adapter)
      ADAPTER_PATH="$2"
      shift 2
      ;;
    --no-adapter)
      INCLUDE_ADAPTER=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [ ! -d "${HF_HOME_RESOLVED}" ]; then
  echo "HF cache not found: ${HF_HOME_RESOLVED}" >&2
  exit 1
fi

if [ -z "${ADAPTER_PATH}" ] && [ "${INCLUDE_ADAPTER}" = "1" ]; then
  if [ -d "${SCRIPT_DIR}/${DEFAULT_ADAPTER_NAME}" ]; then
    ADAPTER_PATH="${SCRIPT_DIR}/${DEFAULT_ADAPTER_NAME}"
  elif [ -d "${SCRIPT_DIR}/${FALLBACK_ADAPTER_NAME}" ]; then
    ADAPTER_PATH="${SCRIPT_DIR}/${FALLBACK_ADAPTER_NAME}"
  fi
fi

mkdir -p "${BUNDLE_ROOT}"
rm -rf "${STAGING_DIR}"
rm -f "${ARCHIVE_PATH}"

mkdir -p "${STAGING_DIR}/project/GRPO-B"
mkdir -p "${STAGING_DIR}/hf_cache/hub"
mkdir -p "${STAGING_DIR}/hf_cache/datasets"
mkdir -p "${STAGING_DIR}/docs"

echo "Copying minimal project files..."
for rel_path in "${CODE_FILES[@]}"; do
  if [ ! -e "${SCRIPT_DIR}/${rel_path}" ]; then
    echo "Missing required project file: ${SCRIPT_DIR}/${rel_path}" >&2
    exit 1
  fi
  cp -a "${SCRIPT_DIR}/${rel_path}" "${STAGING_DIR}/project/GRPO-B/"
done

echo "Copying Hugging Face model cache..."
for rel_dir in \
  "hub/${BASE_MODEL_REPO}" \
  "hub/datasets--gsm8k" \
  "hub/datasets--openai--gsm8k" \
  "datasets/gsm8k" \
  "datasets/openai___gsm8k"
do
  src="${HF_HOME_RESOLVED}/${rel_dir}"
  if [ ! -e "${src}" ]; then
    echo "Missing required cache directory: ${src}" >&2
    exit 1
  fi
  dest_parent="${STAGING_DIR}/hf_cache/$(dirname "${rel_dir}")"
  mkdir -p "${dest_parent}"
  cp -a "${src}" "${dest_parent}/"
done

ADAPTER_BASENAME=""
if [ "${INCLUDE_ADAPTER}" = "1" ] && [ -n "${ADAPTER_PATH}" ]; then
  if [ ! -d "${ADAPTER_PATH}" ]; then
    echo "Adapter path does not exist: ${ADAPTER_PATH}" >&2
    exit 1
  fi
  echo "Copying adapter directory..."
  cp -a "${ADAPTER_PATH}" "${STAGING_DIR}/project/GRPO-B/"
  ADAPTER_BASENAME=$(basename "${ADAPTER_PATH}")
  if [ "${ADAPTER_BASENAME}" != "${DEFAULT_ADAPTER_NAME}" ] && [ ! -e "${STAGING_DIR}/project/GRPO-B/${DEFAULT_ADAPTER_NAME}" ]; then
    ln -s "${ADAPTER_BASENAME}" "${STAGING_DIR}/project/GRPO-B/${DEFAULT_ADAPTER_NAME}"
  fi
fi

if [ "${INCLUDE_ENV}" = "1" ]; then
  if [ ! -d "${ENV_DIR}" ]; then
    echo "Conda env directory not found: ${ENV_DIR}" >&2
    exit 1
  fi
  echo "Packing conda environment directory..."
  mkdir -p "${STAGING_DIR}/envs"
  tar -C "$(dirname "${ENV_DIR}")" -cf "${STAGING_DIR}/envs/${ENV_NAME}.tar" "$(basename "${ENV_DIR}")"
fi

echo "Writing restore notes..."
cat > "${STAGING_DIR}/docs/README_restore.txt" <<EOF
Offline bundle for:
  - run_lm_eval_gsm8k.sh
  - run_train.sh
  - run_smoke_test1.sh

Contents:
  - project/GRPO-B
  - hf_cache
$(if [ "${INCLUDE_ENV}" = "1" ]; then echo "  - envs/${ENV_NAME}.tar"; fi)
$(if [ -n "${ADAPTER_BASENAME}" ]; then echo "  - adapter: ${ADAPTER_BASENAME}"; fi)

Recommended restore steps on the new WSL machine:

1. Extract the outer bundle:
   tar -xf ${BUNDLE_NAME}.tar

2. Restore the project:
   mkdir -p \$HOME/GRPO-B
   cp -a ${BUNDLE_NAME}/project/GRPO-B/. \$HOME/GRPO-B/

3. Restore Hugging Face cache:
   mkdir -p \$HOME/.cache/huggingface
   cp -a ${BUNDLE_NAME}/hf_cache/. \$HOME/.cache/huggingface/

4. Restore conda env:
$(if [ "${INCLUDE_ENV}" = "1" ]; then cat <<EOSTEP
   mkdir -p \$HOME/miniconda3/envs
   tar -C \$HOME/miniconda3/envs -xf ${BUNDLE_NAME}/envs/${ENV_NAME}.tar
EOSTEP
else
  echo "   Skipped by request."
fi)

5. Run:
   cd \$HOME/GRPO-B
   conda activate ${ENV_NAME}
   bash run_smoke_test1.sh
   bash run_train.sh
   bash run_lm_eval_gsm8k.sh$(if [ -n "${ADAPTER_BASENAME}" ]; then echo " ./$(basename "${ADAPTER_PATH}")"; fi)

Notes:
  - This bundle assumes the new machine also uses WSL / Linux.
  - The packed conda env is a raw copy. It is most reliable when extracted to the same path:
      \$HOME/miniconda3/envs/${ENV_NAME}
  - HF cache is restored under:
      \$HOME/.cache/huggingface
EOF

echo "Creating final tar archive..."
tar -C "${BUNDLE_ROOT}" -cf "${ARCHIVE_PATH}" "${BUNDLE_NAME}"

echo
echo "Bundle ready:"
echo "  staging: ${STAGING_DIR}"
echo "  archive: ${ARCHIVE_PATH}"
