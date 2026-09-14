import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object in {path} line {line_no}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def checkpoint_name(path: Path) -> str:
    return path.name.rstrip("/")


def select_uid1_rows(sorted_rows: List[Dict], start_uid1: int, end_uid1: int) -> List[Dict]:
    if start_uid1 > end_uid1:
        raise ValueError(f"start_uid1 ({start_uid1}) must be <= end_uid1 ({end_uid1})")

    selected: List[Dict] = []
    seen_uid1 = set()
    seen_uid = set()
    duplicate_uid1 = []
    duplicate_uid = []

    for row in sorted_rows:
        if "uid1" not in row:
            raise ValueError("Sorted score file is missing required field: uid1")
        if "uid" not in row:
            raise ValueError("Sorted score file is missing required field: uid")
        uid1 = int(row["uid1"])
        uid = str(row["uid"])
        if uid1 in seen_uid1:
            duplicate_uid1.append(uid1)
        seen_uid1.add(uid1)
        if uid in seen_uid:
            duplicate_uid.append(uid)
        seen_uid.add(uid)
        if start_uid1 <= uid1 <= end_uid1:
            selected.append(row)

    if duplicate_uid1:
        sample = ", ".join(str(x) for x in duplicate_uid1[:10])
        raise ValueError(f"Duplicate uid1 values found, e.g. {sample}")
    if duplicate_uid:
        sample = ", ".join(duplicate_uid[:10])
        raise ValueError(f"Duplicate uid values found, e.g. {sample}")
    if not selected:
        raise ValueError(f"No rows selected for uid1 range [{start_uid1}, {end_uid1}]")
    return selected


def enrich_scores(scored_rows: List[Dict], selected_rows: List[Dict]) -> Tuple[List[Dict], int]:
    source_by_uid = {str(row["uid"]): row for row in selected_rows}
    enriched: List[Dict] = []
    missing_uid1 = 0
    for scored in scored_rows:
        uid = str(scored.get("uid"))
        source = source_by_uid.get(uid)
        out = dict(scored)
        if source is None:
            missing_uid1 += 1
        else:
            out["uid1"] = str(source["uid1"])
            if "p" in source:
                out["source_p"] = source["p"]
            if "_num_answers" in source:
                out["source_num_answers"] = source["_num_answers"]
            if "_sample_likely_truncated" in source:
                out["source_sample_likely_truncated"] = source["_sample_likely_truncated"]
        enriched.append(out)
    enriched.sort(key=lambda row: int(row.get("uid1", 10**18)))
    return enriched, missing_uid1


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a checkpoint on a contiguous uid1 range from a sorted GSM8K "
            "difficulty file. uid1 is the sorted position; uid is the original GSM8K row id."
        )
    )
    parser.add_argument("--sorted_scores_file", type=Path, required=True)
    parser.add_argument("--start_uid1", type=int, required=True)
    parser.add_argument("--end_uid1", type=int, required=True)
    parser.add_argument("--checkpoint_path", type=Path, required=True)
    parser.add_argument("--eval_output_dir", type=Path, required=True)
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--generation_batch_size", type=int, default=8)
    parser.add_argument("--prompt_batch_size", type=int, default=1)
    parser.add_argument("--prompt_style", type=str, default="short", choices=["short", "fewshot"])
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--use_4bit", type=int, default=1, choices=[0, 1])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent

    if not args.sorted_scores_file.is_file():
        raise SystemExit(f"Sorted score file not found: {args.sorted_scores_file}")
    if not args.checkpoint_path.is_dir():
        raise SystemExit(f"Checkpoint directory not found: {args.checkpoint_path}")

    args.eval_output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = checkpoint_name(args.checkpoint_path)
    range_label = f"uid1_{args.start_uid1}_{args.end_uid1}"

    sorted_rows = read_jsonl(args.sorted_scores_file)
    selected_rows = select_uid1_rows(sorted_rows, args.start_uid1, args.end_uid1)
    uid_file = args.eval_output_dir / f"{range_label}_uid_file.jsonl"
    raw_scores_file = args.eval_output_dir / f"raw_scores_{ckpt_name}_{range_label}.jsonl"
    enriched_scores_file = args.eval_output_dir / f"scores_{ckpt_name}_{range_label}.jsonl"
    summary_file = args.eval_output_dir / f"summary_{ckpt_name}_{range_label}.json"
    log_file = args.eval_output_dir / f"eval_{ckpt_name}_{range_label}.log"
    manifest_file = args.eval_output_dir / f"manifest_{ckpt_name}_{range_label}.json"

    write_jsonl(uid_file, selected_rows)
    print(f"[prepare] selected rows: {len(selected_rows)}", flush=True)
    print(f"[prepare] uid_file: {uid_file}", flush=True)

    cmd = [
        sys.executable,
        "-u",
        str(repo_root / "gsm8k_p_filter_final_keep_truncated.py"),
        "--base_model",
        args.base_model,
        "--adapter_path",
        str(args.checkpoint_path),
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
        str(raw_scores_file),
        "--summary_out",
        str(summary_file),
    ]
    if args.use_4bit:
        cmd.append("--use_4bit")
    if args.resume:
        cmd.append("--resume")

    print("[eval] " + " ".join(cmd), flush=True)
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
            raise SystemExit(f"Evaluation failed; see log: {log_file}")

        scored_rows = read_jsonl(raw_scores_file)
        enriched_rows, missing_uid1 = enrich_scores(scored_rows, selected_rows)
        write_jsonl(enriched_scores_file, enriched_rows)
    else:
        missing_uid1 = 0

    manifest = {
        "sorted_scores_file": str(args.sorted_scores_file),
        "checkpoint_path": str(args.checkpoint_path),
        "uid1_range": [args.start_uid1, args.end_uid1],
        "num_selected": len(selected_rows),
        "K": args.K,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "uid_file": str(uid_file),
        "raw_scores_file": str(raw_scores_file),
        "scores_file": str(enriched_scores_file),
        "summary_file": str(summary_file),
        "log_file": str(log_file),
        "missing_uid1_after_enrich": missing_uid1,
    }
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[done] scores_file: {enriched_scores_file}", flush=True)
    print(f"[done] summary_file: {summary_file}", flush=True)
    print(f"[done] manifest_file: {manifest_file}", flush=True)


if __name__ == "__main__":
    main()
