#!/bin/bash
set -euo pipefail

# One manually controlled accuracy-gated GRPO training chunk.
#
# Default behavior:
#   - require the training score JSONL to be specified manually
#   - require the current gate interval to be specified manually
#   - run 50 accepted updates and save checkpoint-50 for manual evaluation
#
# To continue after manual inspection, start a new output directory and pass the
# previous checkpoint as --init_adapter_path through this wrapper.

TRAIN_SCORES_FILE="${TRAIN_SCORES_FILE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-./grpo_qwen25_15b_gsm8k_lora_observation_gate_manual_seed42}"

STAGE_SCHEDULE="${STAGE_SCHEDULE:-}"
TARGET_UPDATE_STEPS="${TARGET_UPDATE_STEPS:-50}"
SAVE_STEPS="${SAVE_STEPS:-50}"

if [ -z "${STAGE_SCHEDULE}" ]; then
  case " $* " in
    *" --stage_schedule "*)
      ;;
    *)
      echo "Error: specify the current gate interval with STAGE_SCHEDULE or --stage_schedule." >&2
      echo "Example:" >&2
      echo "  STAGE_SCHEDULE='0.625-0.875:50' OUTPUT_DIR='./grpo_obs_gate_p_0.625_0.875_round1' bash $0 --mode background" >&2
      exit 1
      ;;
  esac
fi

if [ -z "${TRAIN_SCORES_FILE}" ]; then
  case " $* " in
    *" --scores_file "*|*" --train_scores_file "*)
      ;;
    *)
      echo "Error: specify the training JSONL with TRAIN_SCORES_FILE, --scores_file, or --train_scores_file." >&2
      echo "Example:" >&2
      echo "  TRAIN_SCORES_FILE='outputs/observation_sets_gsm8k_p_intervals_seed42/gsm8k_train_remaining_after_5x128_observe_seed42.jsonl' \\" >&2
      echo "  STAGE_SCHEDULE='0.625-0.875:50' OUTPUT_DIR='./grpo_obs_gate_p_0.625_0.875_round1' bash $0 --mode background" >&2
      exit 1
      ;;
  esac
fi

CMD=(
  bash run_train_accuracy_gate_schedule.sh
  --target_update_steps "${TARGET_UPDATE_STEPS}"
  --save_steps "${SAVE_STEPS}"
  --output_dir "${OUTPUT_DIR}"
)

if [ -n "${TRAIN_SCORES_FILE}" ]; then
  CMD+=(--scores_file "${TRAIN_SCORES_FILE}")
fi

if [ -n "${STAGE_SCHEDULE}" ]; then
  CMD+=(--stage_schedule "${STAGE_SCHEDULE}")
fi

exec "${CMD[@]}" "$@"
