import argparse
import glob
import json
import os
import random
from typing import Dict, List, Set

import numpy as np
import torch
from datasets import Dataset
from tqdm import tqdm

from dataset_utils import (
    GSM8K_COT_STOP_STRINGS,
    SUPPORTED_PROMPT_STYLES,
    build_messages,
    get_gold_answer,
    get_question,
    load_gsm8k,
)
from model_utils import load_model_for_inference, resolve_cached_model_path, truncate_on_stop_strings
from reward_utils import score_prediction_against_answer


def log(message: str) -> None:
    print(message, flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_input_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_gsm8k_from_local_arrow(split: str) -> Dataset:
    hf_home = os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
    patterns = [
        os.path.join(hf_home, "datasets", "openai___gsm8k", "main", "0.0.0", "*", f"gsm8k-{split}.arrow"),
        os.path.join(hf_home, "datasets", "gsm8k", "main", "0.0.0", "*", f"gsm8k-{split}.arrow"),
    ]

    candidates: List[str] = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern))

    if not candidates:
        return load_gsm8k(split)

    arrow_path = max(candidates, key=os.path.getmtime)
    log(f"Loading GSM8K split='{split}' from local arrow file: {arrow_path}")
    return Dataset.from_file(arrow_path)


def compute_p(answers: List[str], gold_answer: str) -> float:
    if not answers:
        return 0.0
    return sum(score_prediction_against_answer(answer, gold_answer) for answer in answers) / len(answers)


def derive_truncation_out_path(out_path: str, truncation_out: str) -> str:
    if truncation_out:
        return truncation_out
    if out_path.endswith(".jsonl"):
        return out_path[:-6] + ".truncated.jsonl"
    return out_path + ".truncated.jsonl"


def load_jsonl_rows(path: str) -> List[Dict]:
    rows: List[Dict] = []
    if not path or not os.path.exists(path):
        return rows

    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                log(f"[warn] Skipping invalid JSONL line {line_no} in {path}")
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def load_uid_filter(path: str) -> Set[str]:
    rows = load_jsonl_rows(path)
    uids: Set[str] = set()
    for row in rows:
        uid = row.get("uid")
        if uid is not None:
            uids.add(str(uid))
    return uids


def restore_resume_state(existing_rows: List[Dict]) -> Dict:
    completed_uids: Set[str] = set()
    mean_p = 0.0
    total_generations = 0
    total_likely_truncated = 0
    samples_with_any_truncation = 0
    samples_with_all_truncation = 0
    num_scored_rows = 0
    diagnostics_complete = True

    for row in existing_rows:
        uid = row.get("uid")
        if uid is not None:
            completed_uids.add(str(uid))

        if "p" in row:
            mean_p += row["p"]
            num_scored_rows += 1

        has_truncation_diagnostics = all(
            key in row
            for key in [
                "_sample_likely_truncated",
                "_num_answers",
            ]
        )
        if has_truncation_diagnostics:
            total_generations += row["_num_answers"]
            total_likely_truncated += row["_sample_likely_truncated"]
            if row["_sample_likely_truncated"] > 0:
                samples_with_any_truncation += 1
            if row["_sample_likely_truncated"] == row["_num_answers"]:
                samples_with_all_truncation += 1
        elif "p" in row:
            diagnostics_complete = False

    return {
        "completed_uids": completed_uids,
        "mean_p": mean_p,
        "total_generations": total_generations,
        "total_likely_truncated": total_likely_truncated,
        "samples_with_any_truncation": samples_with_any_truncation,
        "samples_with_all_truncation": samples_with_all_truncation,
        "num_scored_rows": num_scored_rows,
        "diagnostics_complete": diagnostics_complete,
    }


@torch.no_grad()
def sample_answers_batch_safe_text(
    model,
    tokenizer,
    messages_batch: List[List[Dict[str, str]]],
    K: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    repetition_penalty: float = 1.0,
    generation_batch_size: int = 1,
) -> List[List[Dict]]:
    rendered_prompts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        for messages in messages_batch
    ]
    inputs = tokenizer(rendered_prompts, return_tensors="pt", padding=True)
    device = get_input_device(model)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_length = inputs["input_ids"].shape[1]

    answers_batch: List[List[Dict]] = [[] for _ in rendered_prompts]
    do_sample = temperature > 0.0
    generation_batch_size = max(1, generation_batch_size)

    for start_idx in range(0, K, generation_batch_size):
        batch_k = min(generation_batch_size, K - start_idx)
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "num_return_sequences": batch_k,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "repetition_penalty": repetition_penalty,
            "use_cache": True,
            "do_sample": do_sample,
        }
        if do_sample:
            generation_kwargs.update(
                {
                    "temperature": temperature,
                    "top_p": top_p,
                }
            )

        output_ids = model.generate(**inputs, **generation_kwargs)
        for prompt_idx in range(len(rendered_prompts)):
            for seq_idx in range(batch_k):
                output_idx = prompt_idx * batch_k + seq_idx
                completion_ids = output_ids[output_idx, prompt_length:]
                text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
                text = truncate_on_stop_strings(text, stop_strings=GSM8K_COT_STOP_STRINGS)
                generated_token_count = int(completion_ids.shape[0])
                ended_with_eos = bool(
                    generated_token_count > 0
                    and tokenizer.eos_token_id is not None
                    and completion_ids[-1].item() == tokenizer.eos_token_id
                )
                answer_info = {
                    "text": text,
                    "generated_tokens": generated_token_count,
                    "ended_with_eos": ended_with_eos,
                    "likely_length_truncated": bool(generated_token_count >= max_new_tokens and not ended_with_eos),
                }
                answers_batch[prompt_idx].append(answer_info)

        del output_ids

    return answers_batch


@torch.no_grad()
def sample_answers_safe_text(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    K: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    repetition_penalty: float = 1.0,
    generation_batch_size: int = 1,
) -> List[Dict]:
    return sample_answers_batch_safe_text(
        model=model,
        tokenizer=tokenizer,
        messages_batch=[messages],
        K=K,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        generation_batch_size=generation_batch_size,
    )[0]


def build_scored_row(
    idx: int,
    split: str,
    question: str,
    gold_answer: str,
    answer_infos: List[Dict],
) -> Dict:
    answers = [item["text"] for item in answer_infos]
    p = compute_p(answers, gold_answer)
    sample_likely_truncated = sum(1 for item in answer_infos if item["likely_length_truncated"])
    return {
        "uid": str(idx),
        "split": split,
        "question": question,
        "answer_gold": gold_answer,
        "p": p,
        "_sample_likely_truncated": sample_likely_truncated,
        "_num_answers": len(answer_infos),
    }


def build_truncation_row(
    row: Dict,
    answer_infos: List[Dict],
) -> Dict | None:
    if row["_sample_likely_truncated"] <= 0:
        return None
    return {"uid": row["uid"]}


def select_examples(dataset, max_samples: int, seed: int, shuffle: bool):
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    if max_samples is not None and max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset


def normalize_uid_range(dataset_size: int, min_uid: int | None, max_uid: int | None) -> range:
    if min_uid is not None and min_uid < 0:
        raise ValueError("--min_uid/--start_uid must be >= 0.")
    if max_uid is not None and max_uid < 0:
        raise ValueError("--max_uid/--end_uid must be >= 0.")
    if min_uid is not None and max_uid is not None and min_uid > max_uid:
        raise ValueError(f"min_uid ({min_uid}) must be <= max_uid ({max_uid}).")

    start_uid = 0 if min_uid is None else min_uid
    end_uid = (dataset_size - 1) if max_uid is None else min(max_uid, dataset_size - 1)

    if dataset_size <= 0 or start_uid >= dataset_size or start_uid > end_uid:
        return range(0)
    return range(start_uid, end_uid + 1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Probability-only GSM8K scorer using a text Qwen model."
    )
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--out", type=str, default="outputs/gsm8k_p_scores.jsonl")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=160)
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--report_every", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--generation_batch_size", type=int, default=8)
    parser.add_argument("--prompt_batch_size", type=int, default=2)
    parser.add_argument("--prompt_style", type=str, default="short", choices=sorted(SUPPORTED_PROMPT_STYLES))
    parser.add_argument("--summary_out", type=str, default="")
    parser.add_argument("--min_uid", "--start_uid", dest="min_uid", type=int, default=None)
    parser.add_argument("--max_uid", "--end_uid", dest="max_uid", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--truncation_out",
        type=str,
        default="",
        help="JSONL path for samples with at least one likely length-truncated generation.",
    )
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

    log("Run config:")
    log(f"  base_model={args.base_model}")
    log(f"  adapter_path={args.adapter_path or '<none>'}")
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
    truncation_out = derive_truncation_out_path(args.out, args.truncation_out)
    os.makedirs(os.path.dirname(truncation_out) or ".", exist_ok=True)

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
        existing_rows = load_jsonl_rows(args.out)
        existing_source = args.out
        existing_truncation_rows = load_jsonl_rows(truncation_out)

        existing_rows = [
            row for row in existing_rows if str(row.get("uid")) in target_uid_set
        ]
        existing_truncation_uids = {
            str(row.get("uid"))
            for row in existing_truncation_rows
            if row.get("uid") is not None and str(row.get("uid")) in target_uid_set
        }

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
            log(
                f"Resume enabled: loaded {len(existing_rows)} completed rows "
                f"from {existing_source}"
            )
        else:
            log("Resume enabled: no existing rows found, starting a fresh run")
        if existing_truncation_uids:
            completed_uids |= existing_truncation_uids
            log(
                f"Resume enabled: loaded {len(existing_truncation_uids)} truncated uid rows "
                f"from {truncation_out}"
            )

    out_mode = "a" if args.resume and os.path.exists(args.out) else "w"
    truncation_mode = "a" if args.resume and os.path.exists(truncation_out) else "w"

    with open(args.out, out_mode, encoding="utf-8") as handle, open(truncation_out, truncation_mode, encoding="utf-8") as truncation_handle:
        progress = tqdm(total=len(target_uid_set), desc="Scoring GSM8K", dynamic_ncols=True)
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
                if str(batch_start + local_idx) in target_uid_set and str(batch_start + local_idx) not in completed_uids
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
                idx = batch_start + local_idx + 1
                if not answer_infos:
                    row = {
                        "uid": str(idx - 1),
                        "question": question,
                        "answer_gold": gold_answer,
                        "error": "oom",
                    }
                    rows.append(row)
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    completed_uids.add(str(idx - 1))
                    progress.update(1)
                    continue

                row = build_scored_row(
                    idx=idx - 1,
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

                truncation_row = build_truncation_row(
                    row=row,
                    answer_infos=answer_infos,
                )
                if truncation_row is not None:
                    truncation_handle.write(json.dumps(truncation_row, ensure_ascii=False) + "\n")
                else:
                    rows.append(row)
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                completed_uids.add(str(idx - 1))

                progress.update(1)
                progress.set_postfix(
                    mean_p=f"{mean_p / max(num_scored_rows, 1):.4f}",
                    trunc_rate=f"{total_likely_truncated / max(total_generations, 1):.3f}",
                )
                if args.report_every > 0 and idx % args.report_every == 0:
                    log(
                        f"[progress] {idx}/{len(dataset)} "
                        f"mean_p={mean_p / max(num_scored_rows, 1):.4f} "
                        f"trunc_rate={total_likely_truncated / max(total_generations, 1):.4f}"
                    )

        progress.close()

    log(f"Saved scored rows to: {args.out}")
    log(f"Saved truncation rows to: {truncation_out}")

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
        "truncation_out": truncation_out,
        "uid_file": args.uid_file or None,
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
