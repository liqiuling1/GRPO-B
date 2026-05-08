import argparse
import json
import os
import random
from typing import Dict, List, Set

import numpy as np
import torch
from tqdm import tqdm

from dataset_utils import SUPPORTED_PROMPT_STYLES, build_messages, get_gold_answer, get_question
from gsm8k_p_filter import (
    build_scored_row,
    load_gsm8k_from_local_arrow,
    load_jsonl_rows,
    load_uid_filter,
    normalize_uid_range,
    restore_resume_state,
    sample_answers_batch_safe_text,
    sample_answers_safe_text,
    select_examples,
)
from model_utils import load_model_for_inference, resolve_cached_model_path


def log(message: str) -> None:
    print(message, flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Final GSM8K p scorer that keeps scored rows even when generations are length-truncated."
    )
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--out", type=str, default="outputs/gsm8k_p_scores_final_keep_truncated.jsonl")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--report_every", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--generation_batch_size", type=int, default=8)
    parser.add_argument("--prompt_batch_size", type=int, default=1)
    parser.add_argument("--prompt_style", type=str, default="short", choices=sorted(SUPPORTED_PROMPT_STYLES))
    parser.add_argument("--summary_out", type=str, default="")
    parser.add_argument("--min_uid", "--start_uid", dest="min_uid", type=int, default=None)
    parser.add_argument("--max_uid", "--end_uid", dest="max_uid", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--uid_file",
        type=str,
        default="",
        help="Optional JSONL file containing uid fields; only these dataset rows are scored.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.K < 1:
        raise ValueError("--K must be >= 1.")

    set_seed(args.seed)

    base_model_path = resolve_cached_model_path(args.base_model)

    log("Final keep-truncated run config:")
    log(f"  base_model={args.base_model}")
    log(f"  adapter_path={args.adapter_path or '<none>'}")
    log(f"  split={args.split}")
    log(f"  max_samples={args.max_samples}")
    log(f"  K={args.K}")
    log(f"  max_new_tokens={args.max_new_tokens}")
    log(f"  generation_batch_size={args.generation_batch_size}")
    log(f"  prompt_batch_size={args.prompt_batch_size}")
    log(f"  prompt_style={args.prompt_style}")
    log(f"  min_uid={args.min_uid}")
    log(f"  max_uid={args.max_uid}")
    log(f"  uid_file={args.uid_file or '<none>'}")
    log(f"  resume={args.resume}")
    log("  keep_truncated_scores=True")

    log(f"Loading base model: {args.base_model}")
    if base_model_path != args.base_model:
        log(f"Resolved base model to local cache: {base_model_path}")
    if args.adapter_path:
        log(f"Using adapter: {args.adapter_path}")
    else:
        log("Running without adapter")
    model, tokenizer = load_model_for_inference(
        base_model_name=base_model_path,
        adapter_path=args.adapter_path,
        use_4bit=args.use_4bit,
    )

    log(f"Loading GSM8K split='{args.split}' from local Hugging Face cache if available")
    dataset = load_gsm8k_from_local_arrow(args.split)
    dataset = select_examples(
        dataset=dataset,
        max_samples=args.max_samples,
        seed=args.seed,
        shuffle=args.shuffle,
    )
    target_uid_range = normalize_uid_range(
        dataset_size=len(dataset),
        min_uid=args.min_uid,
        max_uid=args.max_uid,
    )
    target_uid_set = {str(uid) for uid in target_uid_range}
    if args.uid_file:
        uid_filter = load_uid_filter(args.uid_file)
        target_uid_set &= uid_filter
        log(f"Loaded {len(uid_filter)} uid values from: {args.uid_file}")
    log(
        f"Scoring {len(target_uid_set)} samples"
        f" (uid range: {target_uid_range.start}..{target_uid_range.stop - 1})"
        if len(target_uid_range) > 0
        else "Scoring 0 samples (uid range is empty)"
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    rows: List[Dict] = []
    mean_p = 0.0
    total_generations = 0
    total_likely_truncated = 0
    samples_with_any_truncation = 0
    samples_with_all_truncation = 0
    num_scored_rows = 0
    completed_uids: Set[str] = set()
    diagnostics_complete = True

    if args.resume:
        existing_rows = [
            row
            for row in load_jsonl_rows(args.out)
            if str(row.get("uid")) in target_uid_set
        ]
        if existing_rows:
            resume_state = restore_resume_state(existing_rows)
            rows = list(existing_rows)
            completed_uids = resume_state["completed_uids"]
            mean_p = resume_state["mean_p"]
            total_generations = resume_state["total_generations"]
            total_likely_truncated = resume_state["total_likely_truncated"]
            samples_with_any_truncation = resume_state["samples_with_any_truncation"]
            samples_with_all_truncation = resume_state["samples_with_all_truncation"]
            num_scored_rows = resume_state["num_scored_rows"]
            diagnostics_complete = resume_state["diagnostics_complete"]
            log(f"Resume enabled: loaded {len(existing_rows)} completed rows from {args.out}")
        else:
            log("Resume enabled: no existing rows found, starting a fresh run")

    out_mode = "a" if args.resume and os.path.exists(args.out) else "w"

    with open(args.out, out_mode, encoding="utf-8") as handle:
        progress = tqdm(total=len(target_uid_set), desc="Final scoring GSM8K", dynamic_ncols=True)
        if completed_uids:
            progress.update(len(completed_uids & target_uid_set))

        for batch_start in range(0, len(dataset), max(1, args.prompt_batch_size)):
            batch_end = min(batch_start + max(1, args.prompt_batch_size), len(dataset))
            batch_examples = [dataset[i] for i in range(batch_start, batch_end)]
            questions = [get_question(example) for example in batch_examples]
            gold_answers = [get_gold_answer(example) for example in batch_examples]
            messages_batch = [build_messages(question, prompt_style=args.prompt_style) for question in questions]
            pending_local_indices = [
                local_idx
                for local_idx in range(len(batch_examples))
                if str(batch_start + local_idx) in target_uid_set
                and str(batch_start + local_idx) not in completed_uids
            ]

            if not pending_local_indices:
                continue

            pending_messages_batch = [messages_batch[idx] for idx in pending_local_indices]

            try:
                answer_infos_batch = sample_answers_batch_safe_text(
                    model=model,
                    tokenizer=tokenizer,
                    messages_batch=pending_messages_batch,
                    K=args.K,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    repetition_penalty=args.repetition_penalty,
                    generation_batch_size=args.generation_batch_size,
                )
            except torch.cuda.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                answer_infos_batch = []
                for messages in pending_messages_batch:
                    try:
                        answer_infos_batch.append(
                            sample_answers_safe_text(
                                model=model,
                                tokenizer=tokenizer,
                                messages=messages,
                                K=args.K,
                                temperature=args.temperature,
                                top_p=args.top_p,
                                max_new_tokens=args.max_new_tokens,
                                repetition_penalty=args.repetition_penalty,
                                generation_batch_size=args.generation_batch_size,
                            )
                        )
                    except torch.cuda.OutOfMemoryError:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        answer_infos_batch.append([])

            answer_infos_by_local_idx = {
                local_idx: answer_infos
                for local_idx, answer_infos in zip(pending_local_indices, answer_infos_batch)
            }

            for local_idx in pending_local_indices:
                question = questions[local_idx]
                gold_answer = gold_answers[local_idx]
                answer_infos = answer_infos_by_local_idx[local_idx]
                uid = batch_start + local_idx

                if not answer_infos:
                    row = {
                        "uid": str(uid),
                        "question": question,
                        "answer_gold": gold_answer,
                        "error": "oom",
                    }
                    rows.append(row)
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    completed_uids.add(str(uid))
                    progress.update(1)
                    continue

                row = build_scored_row(
                    idx=uid,
                    split=args.split,
                    question=question,
                    gold_answer=gold_answer,
                    answer_infos=answer_infos,
                )

                num_scored_rows += 1
                total_generations += row["_num_answers"]
                total_likely_truncated += row["_sample_likely_truncated"]
                if row["_sample_likely_truncated"] > 0:
                    samples_with_any_truncation += 1
                if row["_sample_likely_truncated"] == row["_num_answers"]:
                    samples_with_all_truncation += 1

                mean_p += row["p"]
                rows.append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                completed_uids.add(str(uid))

                progress.update(1)
                progress.set_postfix(
                    mean_p=f"{mean_p / max(num_scored_rows, 1):.4f}",
                    trunc_rate=f"{total_likely_truncated / max(total_generations, 1):.3f}",
                )
                if args.report_every > 0 and (uid + 1) % args.report_every == 0:
                    log(
                        f"[progress] {uid + 1}/{len(dataset)} "
                        f"mean_p={mean_p / max(num_scored_rows, 1):.4f} "
                        f"trunc_rate={total_likely_truncated / max(total_generations, 1):.4f}"
                    )

        progress.close()

    log(f"Saved final scored rows to: {args.out}")

    summary = {
        "split": args.split,
        "num_samples": len(rows),
        "num_scored_samples": num_scored_rows,
        "min_uid": args.min_uid,
        "max_uid": args.max_uid,
        "K": args.K,
        "max_new_tokens": args.max_new_tokens,
        "prompt_style": args.prompt_style,
        "mean_p": mean_p / max(num_scored_rows, 1),
        "resume": args.resume,
        "uid_file": args.uid_file or None,
        "keep_truncated_scores": True,
        "diagnostics_complete": diagnostics_complete,
    }
    if diagnostics_complete:
        summary.update(
            {
                "total_generations": total_generations,
                "likely_truncated_generations": total_likely_truncated,
                "likely_truncation_rate": total_likely_truncated / max(total_generations, 1),
                "samples_with_any_truncation": samples_with_any_truncation,
                "samples_with_all_truncation": samples_with_all_truncation,
                "sample_any_truncation_rate": samples_with_any_truncation / max(num_scored_rows, 1),
                "sample_all_truncation_rate": samples_with_all_truncation / max(num_scored_rows, 1),
            }
        )
    else:
        summary.update(
            {
                "total_generations": None,
                "likely_truncated_generations": None,
                "likely_truncation_rate": None,
                "samples_with_any_truncation": None,
                "samples_with_all_truncation": None,
                "sample_any_truncation_rate": None,
                "sample_all_truncation_rate": None,
                "warning": "Resumed from an older output file without per-sample diagnostics; truncation-related summary fields are unavailable until a fresh run or a complete state file is present.",
            }
        )
    log(f"Summary: {json.dumps(summary, ensure_ascii=False)}")

    if args.summary_out:
        os.makedirs(os.path.dirname(args.summary_out) or ".", exist_ok=True)
        with open(args.summary_out, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        log(f"Saved summary to: {args.summary_out}")


if __name__ == "__main__":
    main()
