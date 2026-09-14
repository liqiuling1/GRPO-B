import argparse
import json
from pathlib import Path
from typing import Dict

from datasets import Dataset, load_dataset
from tqdm import tqdm

from dataset_utils import GSM8K_COT_STOP_STRINGS, build_messages
from model_utils import generate_n_responses, load_model_for_inference
from reward_utils import (
    extract_final_answer,
    extract_last_number,
    normalize_text_answer,
    score_prediction_against_answer,
)


def log(message: str) -> None:
    print(message, flush=True)


def get_svamp_question(example: Dict) -> str:
    question = (example.get("question_concat") or example.get("question") or "").strip()
    if question:
        return question
    body = (example.get("Body") or "").strip()
    tail = (example.get("Question") or "").strip()
    return f"{body} {tail}".strip()


def get_svamp_answer(example: Dict) -> str:
    return str(example.get("Answer", example.get("answer", ""))).strip()


def load_svamp_dataset(split: str, dataset_path: str) -> Dataset:
    if dataset_path:
        log(f"Loading SVAMP from local arrow file: {dataset_path}")
        return Dataset.from_file(dataset_path)
    log(f"Loading SVAMP from Hugging Face: ChilleD/SVAMP split={split}")
    return load_dataset("ChilleD/SVAMP", split=split)


def evaluate(
    model,
    tokenizer,
    dataset: Dataset,
    max_samples: int,
    max_new_tokens: int,
    prompt_style: str,
    verbose: bool,
    report_every: int,
    output_jsonl: str,
) -> float:
    if max_samples is not None and max_samples > 0:
        subset = dataset.select(range(min(max_samples, len(dataset))))
    else:
        subset = dataset

    output_handle = None
    if output_jsonl:
        output_path = Path(output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_handle = output_path.open("w", encoding="utf-8")

    total = 0.0
    count = 0
    progress = tqdm(subset, total=len(subset), desc="svamp-eval", dynamic_ncols=True)
    try:
        for idx, example in enumerate(progress, start=1):
            question = get_svamp_question(example)
            gold_answer = get_svamp_answer(example)
            messages = build_messages(question, prompt_style=prompt_style)

            response = generate_n_responses(
                model=model,
                tokenizer=tokenizer,
                messages=messages,
                n=1,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                do_sample=False,
                stop_strings=GSM8K_COT_STOP_STRINGS,
            )[0]

            score = score_prediction_against_answer(response, gold_answer)
            total += score
            count += 1
            accuracy = total / count
            progress.set_postfix(accuracy=f"{accuracy:.4f}", count=f"{count}/{len(subset)}")

            pred_final = extract_final_answer(response)
            pred_num = extract_last_number(pred_final)
            gold_num = extract_last_number(gold_answer)

            if output_handle is not None:
                output_handle.write(
                    json.dumps(
                        {
                            "idx": idx - 1,
                            "id": example.get("ID", example.get("svamp_id", "")),
                            "type": example.get("Type", example.get("svamp_type", "")),
                            "question": question,
                            "gold_answer": gold_answer,
                            "prediction": response,
                            "pred_final": pred_final,
                            "pred_number": pred_num,
                            "gold_number": gold_num,
                            "score": score,
                            "running_accuracy": accuracy,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            if report_every > 0 and count % report_every == 0:
                tqdm.write(f"[progress] {count}/{len(subset)} accuracy={accuracy:.4f}")

            if verbose:
                tqdm.write("=" * 80)
                tqdm.write(f"[Example {idx}]")
                tqdm.write(f"Question: {question}")
                tqdm.write(f"Gold answer: {gold_answer}")
                tqdm.write(f"Model response: {response}")
                tqdm.write(f"Parsed final answer: {pred_final}")
                tqdm.write(f"Parsed pred number: {pred_num}")
                tqdm.write(f"Parsed gold number: {gold_num}")
                tqdm.write(f"Normalized pred: {normalize_text_answer(pred_final)}")
                tqdm.write(f"Normalized gold: {normalize_text_answer(gold_answer)}")
                tqdm.write(f"Score: {score}")
    finally:
        if output_handle is not None:
            output_handle.close()

    return total / max(count, 1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter_path", type=str, default="")
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--max_new_tokens", type=int, default=1536)
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--no_adapter", action="store_true")
    parser.add_argument("--prompt_style", type=str, default="short", choices=["short", "fewshot"])
    parser.add_argument("--report_every", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output_jsonl", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    adapter_path = None if args.no_adapter or not args.adapter_path else args.adapter_path

    log(f"Base model: {args.base_model}")
    if adapter_path:
        log(f"Adapter: {adapter_path}")
    else:
        log("Adapter: disabled")
    log(f"Prompt style: {args.prompt_style}")
    log(f"Max new tokens: {args.max_new_tokens}")

    model, tokenizer = load_model_for_inference(
        base_model_name=args.base_model,
        adapter_path=adapter_path,
        use_4bit=args.use_4bit,
    )
    dataset = load_svamp_dataset(split=args.split, dataset_path=args.dataset_path)
    eval_count = min(args.max_samples, len(dataset)) if args.max_samples > 0 else len(dataset)
    log(f"Evaluating SVAMP examples: {eval_count}/{len(dataset)}")

    accuracy = evaluate(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        prompt_style=args.prompt_style,
        verbose=args.verbose,
        report_every=args.report_every,
        output_jsonl=args.output_jsonl,
    )
    log(f"svamp_accuracy={accuracy:.4f}")


if __name__ == "__main__":
    main()
