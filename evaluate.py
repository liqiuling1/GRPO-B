import argparse

from datasets import Dataset
from tqdm import tqdm

from dataset_utils import GSM8K_COT_STOP_STRINGS, build_messages, get_gold_answer, get_question, load_gsm8k
from model_utils import generate_n_responses, load_model_for_inference
from reward_utils import extract_final_answer, extract_last_number, score_prediction_against_answer


def log(message: str) -> None:
    print(message, flush=True)


def evaluate(
    model,
    tokenizer,
    dataset,
    max_samples: int = -1,
    max_new_tokens: int = 256,
    verbose: bool = False,
    report_every: int = 0,
    prompt_style: str = "short",
):
    if max_samples is not None and max_samples > 0:
        subset = dataset.select(range(min(max_samples, len(dataset))))
    else:
        subset = dataset

    total = 0.0
    count = 0
    progress = tqdm(subset, total=len(subset), desc="eval", dynamic_ncols=True)

    for idx, example in enumerate(progress, start=1):
        question = get_question(example)
        gold_answer = get_gold_answer(example)
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
        progress.set_postfix(
            accuracy=f"{(total / count):.4f}",
            count=f"{count}/{len(subset)}",
        )
        if report_every > 0 and count % report_every == 0:
            tqdm.write(
                f"[progress] {count}/{len(subset)} "
                f"accuracy={total / count:.4f}"
            )

        if verbose:
            pred_final = extract_final_answer(response)
            pred_num = extract_last_number(pred_final)
            gold_num = extract_last_number(gold_answer)

            tqdm.write("=" * 80)
            tqdm.write(f"[Example {idx}]")
            tqdm.write("Question:")
            tqdm.write(question)
            tqdm.write("\nModel response:")
            tqdm.write(response)
            tqdm.write("\nParsed final answer:")
            tqdm.write(pred_final)
            tqdm.write("\nGold answer:")
            tqdm.write(gold_answer)
            tqdm.write(f"\nParsed pred number: {pred_num}")
            tqdm.write(f"Parsed gold number: {gold_num}")
            tqdm.write(f"Score: {score}")

    return total / max(count, 1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="./grpo_qwen25_15b_gsm8k_lora_grpo_baseline",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--no_adapter", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--report_every", type=int, default=10)
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--prompt_style", type=str, default="short", choices=["short", "fewshot"])
    return parser.parse_args()


def main():
    args = parse_args()
    adapter_path = None if args.no_adapter else args.adapter_path

    log(f"Loading model and tokenizer from base_model='{args.base_model}'")
    if adapter_path:
        log(f"Using adapter: {adapter_path}")
    else:
        log("Running without adapter")
    model, tokenizer = load_model_for_inference(
        base_model_name=args.base_model,
        adapter_path=adapter_path,
        use_4bit=args.use_4bit,
    )

    if args.dataset_path:
        log(f"Loading dataset from local file: {args.dataset_path}")
        dataset = Dataset.from_file(args.dataset_path)
        dataset_desc = args.dataset_path
    else:
        log(f"Loading dataset split='{args.split}'")
        dataset = load_gsm8k(args.split)
        dataset_desc = args.split

    eval_count = min(args.max_samples, len(dataset)) if args.max_samples > 0 else len(dataset)
    log(f"Evaluating {eval_count} examples from dataset='{dataset_desc}'")
    log(f"Prompt style: {args.prompt_style}")
    accuracy = evaluate(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        verbose=args.verbose,
        report_every=args.report_every,
        prompt_style=args.prompt_style,
    )
    log(f"accuracy={accuracy:.4f}")


if __name__ == "__main__":
    main()
