import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict


CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)$")


def checkpoint_step(path: Path) -> int:
    match = CHECKPOINT_RE.search(path.name)
    return int(match.group(1)) if match else -1


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one explicitly selected accuracy-gate checkpoint on one "
            "GSM8K observation set by re-scoring p for that uid file."
        )
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        required=True,
        help="Explicit checkpoint directory to evaluate, for example ./run/checkpoint-50.",
    )
    parser.add_argument("--uid_file", type=Path, required=True, help="Observation JSONL file to evaluate.")
    parser.add_argument("--label", type=str, default="", help="Short label used in output filenames.")
    parser.add_argument(
        "--eval_output_dir",
        type=Path,
        required=True,
        help="Where evaluation score files, summary, log, and manifest are written.",
    )
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--generation_batch_size", type=int, default=8)
    parser.add_argument("--prompt_batch_size", type=int, default=1)
    parser.add_argument("--prompt_style", type=str, default="short", choices=["short", "fewshot"])
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_4bit", type=int, default=1, choices=[0, 1])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without running evaluation.",
    )
    return parser.parse_args()


def load_summary(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    checkpoint = args.checkpoint_path
    if not checkpoint.is_dir():
        raise SystemExit(f"Checkpoint directory not found: {checkpoint}")
    checkpoint_name = checkpoint.name
    checkpoint_step_value = checkpoint_step(checkpoint)
    eval_output_dir = args.eval_output_dir
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    uid_file = args.uid_file
    if not uid_file.is_file():
        raise SystemExit(f"Observation uid_file not found: {uid_file}")
    label = args.label or uid_file.stem

    out_file = eval_output_dir / f"gsm8k_observation_scores_{checkpoint_name}_{label}.jsonl"
    summary_file = eval_output_dir / f"summary_{checkpoint_name}_{label}.json"
    log_file = eval_output_dir / f"eval_{checkpoint_name}_{label}.log"

    cmd = [
        sys.executable,
        "-u",
        str(repo_root / "gsm8k_p_filter_final_keep_truncated.py"),
        "--base_model",
        args.base_model,
        "--adapter_path",
        str(checkpoint),
        "--split",
        args.split,
        "--max_samples",
        "0",
        "--K",
        str(args.K),
        "--temperature",
        str(args.temperature),
        "--top_p",
        str(args.top_p),
        "--seed",
        str(args.seed),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--generation_batch_size",
        str(args.generation_batch_size),
        "--prompt_batch_size",
        str(args.prompt_batch_size),
        "--prompt_style",
        args.prompt_style,
        "--uid_file",
        str(uid_file),
        "--out",
        str(out_file),
        "--summary_out",
        str(summary_file),
    ]
    if args.use_4bit:
        cmd.append("--use_4bit")
    if args.resume:
        cmd.append("--resume")

    print(f"[eval] {checkpoint} -> {label}", flush=True)
    print(" ".join(cmd), flush=True)
    if not args.dry_run:
        env = os.environ.copy()
        with log_file.open("w", encoding="utf-8") as log_handle:
            result = subprocess.run(
                cmd,
                cwd=str(repo_root),
                env=env,
                text=True,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0:
            raise SystemExit(f"Evaluation failed for {checkpoint}; see {log_file}")

    summary = load_summary(summary_file)
    record = {
        "checkpoint_step": checkpoint_step_value if checkpoint_step_value > 0 else None,
        "checkpoint": str(checkpoint),
        "observation_label": label,
        "uid_file": str(uid_file),
        "scores_file": str(out_file),
        "summary_file": str(summary_file),
        "log_file": str(log_file),
        "num_scored_samples": summary.get("num_scored_samples"),
        "mean_p": summary.get("mean_p"),
        "likely_truncation_rate": summary.get("likely_truncation_rate"),
    }

    manifest_path = eval_output_dir / "observation_eval_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "checkpoint": str(checkpoint),
                "uid_file": str(uid_file),
                "label": label,
                "eval_output_dir": str(eval_output_dir),
                "base_model": args.base_model,
                "K": args.K,
                "seed": args.seed,
                "max_new_tokens": args.max_new_tokens,
                "record": record,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")
    print(f"[done] wrote manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
