import argparse
import glob
import json
import os
import random
import re
from collections import Counter
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from dataset_utils import GSM8K_COT_STOP_STRINGS, build_messages, get_gold_answer, get_question, load_gsm8k
from model_utils import load_model_for_inference, resolve_cached_model_path, truncate_on_stop_strings
from reward_utils import extract_final_answer, extract_last_number, normalize_text_answer, score_prediction_against_answer

FINAL_ANSWER_RE = re.compile(r"The answer is (\-?[0-9\.\,]+)\.", flags=re.IGNORECASE)


def log(message: str) -> None:
    print(message, flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_embedder(name: str, device: str) -> Callable[[List[str]], np.ndarray]:
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(name, device=device)

        def encode(texts: List[str]) -> np.ndarray:
            return model.encode(texts, normalize_embeddings=True)

        return encode
    except ModuleNotFoundError:
        resolved_name = resolve_cached_model_path(name)
        local_only = os.path.isdir(resolved_name)
        tokenizer = AutoTokenizer.from_pretrained(
            resolved_name,
            trust_remote_code=True,
            local_files_only=local_only,
        )
        model = AutoModel.from_pretrained(
            resolved_name,
            trust_remote_code=True,
            local_files_only=local_only,
        )
        target_device = torch.device(device)
        model.to(target_device)
        model.eval()

        @torch.no_grad()
        def encode(texts: List[str]) -> np.ndarray:
            inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(target_device) for k, v in inputs.items()}
            outputs = model(**inputs)
            hidden = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            return pooled.cpu().numpy()

        return encode


def get_input_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_cached_model_path(model_name_or_path: str) -> str:
    if os.path.isdir(model_name_or_path):
        return model_name_or_path

    hf_home = os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
    repo_dir = os.path.join(
        hf_home,
        "hub",
        f"models--{model_name_or_path.replace('/', '--')}",
    )
    if not os.path.isdir(repo_dir):
        return model_name_or_path

    ref_path = os.path.join(repo_dir, "refs", "main")
    if os.path.isfile(ref_path):
        with open(ref_path, "r", encoding="utf-8") as handle:
            revision = handle.read().strip()
        snapshot_dir = os.path.join(repo_dir, "snapshots", revision)
        if os.path.isdir(snapshot_dir):
            return snapshot_dir

    snapshots_dir = os.path.join(repo_dir, "snapshots")
    if os.path.isdir(snapshots_dir):
        candidates = sorted(
            os.path.join(snapshots_dir, name)
            for name in os.listdir(snapshots_dir)
        )
        for candidate in reversed(candidates):
            if os.path.isdir(candidate):
                return candidate

    return model_name_or_path


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


def canonicalize_answer_text(text: str) -> str:
    final_answer = extract_final_answer(text)
    number = extract_last_number(final_answer)
    if number is not None:
        return number
    return normalize_text_answer(final_answer)


def compute_p(answers: List[str], gold_answer: str) -> float:
    if not answers:
        return 0.0
    return sum(score_prediction_against_answer(answer, gold_answer) for answer in answers) / len(answers)


def compute_maj(canonical_answers: List[str]) -> float:
    if not canonical_answers:
        return 0.0
    counts = Counter(canonical_answers)
    return max(counts.values()) / len(canonical_answers)


def compute_div_from_embs(embs: np.ndarray) -> float:
    k = embs.shape[0]
    if k <= 1:
        return 0.0
    cos = embs @ embs.T
    avg_cos = (cos.sum() - k) / (k * (k - 1))
    return float(1.0 - avg_cos)


def compute_deff(p: float, div: float, maj: float, tau: float) -> float:
    if maj < tau:
        return 0.0
    return float(4.0 * p * (1.0 - p) * div)


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
                answers_batch[prompt_idx].append(
                    {
                        "text": text,
                        "generated_tokens": generated_token_count,
                        "ended_with_eos": ended_with_eos,
                        "likely_length_truncated": bool(generated_token_count >= max_new_tokens and not ended_with_eos),
                        "has_final_answer_format": bool(FINAL_ANSWER_RE.search(text)),
                        "has_numeric_final_answer": extract_last_number(extract_final_answer(text)) is not None,
                    }
                )

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
    tau: float,
    encode: Callable[[List[str]], np.ndarray],
) -> Dict:
    answers = [item["text"] for item in answer_infos]
    answers_canonical = [canonicalize_answer_text(answer) for answer in answers]
    answer_counter = Counter(answers_canonical)
    p = compute_p(answers, gold_answer)
    maj = compute_maj(answers_canonical)
    sample_likely_truncated = sum(1 for item in answer_infos if item["likely_length_truncated"])
    sample_missing_final_format = sum(1 for item in answer_infos if not item["has_final_answer_format"])
    sample_missing_numeric_final = sum(1 for item in answer_infos if not item["has_numeric_final_answer"])

    if len(set(answers_canonical)) <= 1:
        div = 0.0
    else:
        embed_texts = [text if text else "<empty>" for text in answers_canonical]
        embs = encode(embed_texts)
        div = compute_div_from_embs(embs)

    p_times_one_minus_p = p * (1.0 - p)
    score_p = 4.0 * p_times_one_minus_p
    gate = 1.0 if maj >= tau else 0.0
    score_wo_div = score_p * gate
    score_wo_maj = score_p * div
    deff = compute_deff(p, div, maj, tau)

    return {
        "uid": str(idx),
        "split": split,
        "question": question,
        "answer_gold": gold_answer,
        "p": p,
        "div": div,
        "maj": maj,
        "p_times_one_minus_p": p_times_one_minus_p,
        "deff_wo_div": score_wo_div,
        "deff_wo_maj": score_wo_maj,
        "deff": deff,
        "_sample_likely_truncated": sample_likely_truncated,
        "_sample_missing_final_format": sample_missing_final_format,
        "_sample_missing_numeric_final": sample_missing_numeric_final,
        "_num_answers": len(answer_infos),
    }


def select_examples(dataset, max_samples: int, seed: int, shuffle: bool):
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    if max_samples is not None and max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset


def filter_rows(rows: List[Dict], min_deff: Optional[float], keep_top_n: int) -> List[Dict]:
    kept = rows
    if min_deff is not None:
        kept = [row for row in kept if row.get("deff", 0.0) >= min_deff]
    kept = sorted(kept, key=lambda row: row.get("deff", 0.0), reverse=True)
    if keep_top_n > 0:
        kept = kept[:keep_top_n]
    return kept


def parse_args():
    parser = argparse.ArgumentParser(
        description="DEFF-style GSM8K scorer and filter using a text Qwen model."
    )
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--out", type=str, default="outputs/gsm8k_deff_scores.jsonl")
    parser.add_argument("--filtered_out", type=str, default="")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=160)
    parser.add_argument("--tau", type=float, default=0.4)
    parser.add_argument("--st_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedder_device", type=str, default="cpu")
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--report_every", type=int, default=50)
    parser.add_argument("--min_deff", type=float, default=None)
    parser.add_argument("--keep_top_n", type=int, default=0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--generation_batch_size", type=int, default=4)
    parser.add_argument("--prompt_batch_size", type=int, default=2)
    parser.add_argument("--summary_out", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.K < 2:
        raise ValueError("--K must be >= 2, otherwise maj/div are not meaningful.")

    set_seed(args.seed)

    base_model_path = resolve_cached_model_path(args.base_model)
    st_model_path = resolve_cached_model_path(args.st_model)

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
    log(f"Scoring {len(dataset)} samples")

    log(f"Loading sentence embedder: {args.st_model} on {args.embedder_device}")
    if st_model_path != args.st_model:
        log(f"Resolved embedder to local cache: {st_model_path}")
    encode = build_embedder(st_model_path, device=args.embedder_device)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    rows: List[Dict] = []
    mean_deff = 0.0
    mean_p = 0.0
    total_generations = 0
    total_likely_truncated = 0
    total_missing_final_format = 0
    total_missing_numeric_final = 0
    samples_with_any_truncation = 0
    samples_with_all_truncation = 0

    with open(args.out, "w", encoding="utf-8") as handle:
        progress = tqdm(total=len(dataset), desc="Scoring GSM8K", dynamic_ncols=True)
        for batch_start in range(0, len(dataset), max(1, args.prompt_batch_size)):
            batch_end = min(batch_start + max(1, args.prompt_batch_size), len(dataset))
            batch_examples = [dataset[i] for i in range(batch_start, batch_end)]
            questions = [get_question(example) for example in batch_examples]
            gold_answers = [get_gold_answer(example) for example in batch_examples]
            messages_batch = [build_messages(question) for question in questions]

            try:
                answer_infos_batch = sample_answers_batch_safe_text(
                    model=model,
                    tokenizer=tokenizer,
                    messages_batch=messages_batch,
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
                for messages in messages_batch:
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

            for local_idx, (question, gold_answer, answer_infos) in enumerate(zip(questions, gold_answers, answer_infos_batch)):
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
                    progress.update(1)
                    continue

                row = build_scored_row(
                    idx=idx - 1,
                    split=args.split,
                    question=question,
                    gold_answer=gold_answer,
                    answer_infos=answer_infos,
                    tau=args.tau,
                    encode=encode,
                )

                total_generations += row["_num_answers"]
                total_likely_truncated += row["_sample_likely_truncated"]
                total_missing_final_format += row["_sample_missing_final_format"]
                total_missing_numeric_final += row["_sample_missing_numeric_final"]
                if row["_sample_likely_truncated"] > 0:
                    samples_with_any_truncation += 1
                if row["_sample_likely_truncated"] == row["_num_answers"]:
                    samples_with_all_truncation += 1

                mean_deff += row["deff"]
                mean_p += row["p"]

                public_row = {k: v for k, v in row.items() if not k.startswith("_")}
                rows.append(public_row)
                handle.write(json.dumps(public_row, ensure_ascii=False) + "\n")

                progress.update(1)
                progress.set_postfix(
                    mean_deff=f"{mean_deff / idx:.4f}",
                    mean_p=f"{mean_p / idx:.4f}",
                    trunc_rate=f"{total_likely_truncated / max(total_generations, 1):.3f}",
                )
                if args.report_every > 0 and idx % args.report_every == 0:
                    log(
                        f"[progress] {idx}/{len(dataset)} "
                        f"mean_deff={mean_deff / idx:.4f} mean_p={mean_p / idx:.4f} "
                        f"trunc_rate={total_likely_truncated / max(total_generations, 1):.4f}"
                    )

        progress.close()

    log(f"Saved scored rows to: {args.out}")

    summary = {
        "split": args.split,
        "num_samples": len(rows),
        "num_scored_samples": sum(1 for row in rows if "deff" in row),
        "K": args.K,
        "max_new_tokens": args.max_new_tokens,
        "mean_deff": mean_deff / max(len([row for row in rows if "deff" in row]), 1),
        "mean_p": mean_p / max(len([row for row in rows if "deff" in row]), 1),
        "total_generations": total_generations,
        "likely_truncated_generations": total_likely_truncated,
        "likely_truncation_rate": total_likely_truncated / max(total_generations, 1),
        "missing_final_answer_format_rate": total_missing_final_format / max(total_generations, 1),
        "missing_numeric_final_answer_rate": total_missing_numeric_final / max(total_generations, 1),
        "samples_with_any_truncation": samples_with_any_truncation,
        "samples_with_all_truncation": samples_with_all_truncation,
        "sample_any_truncation_rate": samples_with_any_truncation / max(len(rows), 1),
        "sample_all_truncation_rate": samples_with_all_truncation / max(len(rows), 1),
    }
    log(f"Summary: {json.dumps(summary, ensure_ascii=False)}")

    if args.summary_out:
        os.makedirs(os.path.dirname(args.summary_out) or ".", exist_ok=True)
        with open(args.summary_out, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        log(f"Saved summary to: {args.summary_out}")

    if args.filtered_out:
        filtered_rows = filter_rows(
            rows=[row for row in rows if "deff" in row],
            min_deff=args.min_deff,
            keep_top_n=args.keep_top_n,
        )
        os.makedirs(os.path.dirname(args.filtered_out) or ".", exist_ok=True)
        with open(args.filtered_out, "w", encoding="utf-8") as handle:
            for row in filtered_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        log(f"Saved filtered rows to: {args.filtered_out} ({len(filtered_rows)} samples)")


if __name__ == "__main__":
    main()
