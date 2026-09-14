"""Ordered GRPO training entrypoint.

This intentionally delegates to train_grpo.py. The ordering behavior comes from
passing --train_scores_file: dataset_utils.build_grpo_dataset reads that JSONL
in file order and selects the original GSM8K rows by each row's uid.
"""

from train_grpo import main


if __name__ == "__main__":
    main()
