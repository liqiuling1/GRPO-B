import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from datasets import Dataset, load_dataset

GSM8K_COT_FEWSHOT_EXAMPLES = [
    (
        "There are 15 trees in the grove. Grove workers will plant trees in the grove today. "
        "After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "There are 15 trees originally. Then there were 21 trees after some more were planted. "
        "So there must have been 21 - 15 = 6. The answer is 6.",
    ),
    (
        "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.",
    ),
    (
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. "
        "After eating 35, they had 74 - 35 = 39. The answer is 39.",
    ),
    (
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. "
        "How many lollipops did Jason give to Denny?",
        "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. "
        "So he gave Denny 20 - 12 = 8. The answer is 8.",
    ),
    (
        "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. "
        "How many toys does he have now?",
        "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. "
        "5 + 4 = 9. The answer is 9.",
    ),
    (
        "There were nine computers in the server room. Five more computers were installed each day, "
        "from monday to thursday. How many computers are now in the server room?",
        "There were originally 9 computers. For each of 4 days, 5 more computers were added. "
        "So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.",
    ),
    (
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. "
        "How many golf balls did he have at the end of wednesday?",
        "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. "
        "After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.",
    ),
    (
        "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
        "So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.",
    ),
]
GSM8K_COT_STOP_STRINGS = ["Q:", "</s>", "<|im_end|>"]
SUPPORTED_PROMPT_STYLES = {"short", "fewshot"}


def load_gsm8k(split: str = "train"):
    arrow_path = find_local_gsm8k_arrow(split)
    if arrow_path is not None:
        print(f"Loading GSM8K split='{split}' from local arrow file: {arrow_path}")
        return Dataset.from_file(arrow_path)
    return load_dataset("gsm8k", "main", split=split)


def load_gsm8k_from_path(dataset_path: str):
    dataset_file = Path(dataset_path).expanduser().resolve()
    if not dataset_file.is_file():
        raise FileNotFoundError(f"GSM8K dataset file not found: {dataset_file}")
    print(f"Loading GSM8K from explicit local arrow file: {dataset_file}")
    return Dataset.from_file(str(dataset_file))


def _candidate_hf_homes() -> List[Path]:
    repo_root = Path(__file__).resolve().parent
    homes: List[Path] = []
    seen: set[Path] = set()

    for raw_path in [
        os.environ.get("HF_HOME"),
        str(repo_root / "hf_cache"),
        str(Path.home() / ".cache" / "huggingface"),
    ]:
        if not raw_path:
            continue
        path = Path(raw_path).expanduser().resolve()
        if path not in seen:
            homes.append(path)
            seen.add(path)
    return homes


def find_local_gsm8k_arrow(split: str) -> Optional[str]:
    candidates: List[str] = []
    for hf_home in _candidate_hf_homes():
        patterns = [
            hf_home / "datasets" / "openai___gsm8k" / "main" / "0.0.0" / "*" / f"gsm8k-{split}.arrow",
            hf_home / "datasets" / "gsm8k" / "main" / "0.0.0" / "*" / f"gsm8k-{split}.arrow",
        ]
        for pattern in patterns:
            candidates.extend(glob.glob(str(pattern)))

    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def load_jsonl_rows(path: str) -> List[Dict]:
    rows: List[Dict] = []
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {input_path}: {exc}") from exc
    return rows


def get_question(example: Dict) -> str:
    return example["question"].strip()


def get_gold_answer(example: Dict) -> str:
    return example["answer"]


def build_prompt(question: str, prompt_style: str = "short") -> str:
    question = question.strip()
    if prompt_style not in SUPPORTED_PROMPT_STYLES:
        raise ValueError(
            f"Unsupported prompt_style='{prompt_style}'. "
            f"Expected one of: {sorted(SUPPORTED_PROMPT_STYLES)}."
        )

    if prompt_style == "short":
        return (
            "Solve the following grade school math problem. "
            "Show your reasoning briefly, and end with a final sentence exactly in the form "
            "'The answer is <number>.'\n\n"
            f"Question: {question}"
        )

    parts: List[str] = []
    for example_question, example_answer in GSM8K_COT_FEWSHOT_EXAMPLES:
        parts.append(f"Q: {example_question}\n\nA: {example_answer}")
    parts.append(f"Q: {question}\n\nA:")
    return "\n\n".join(parts)


def build_messages(question: str, prompt_style: str = "short") -> List[Dict[str, str]]:
    return [{"role": "user", "content": build_prompt(question, prompt_style=prompt_style)}]


def build_grpo_dataset(
    split: str = "train",
    max_samples: Optional[int] = None,
    seed: int = 42,
    selected_rows_path: Optional[str] = None,
    min_uid1: Optional[int] = None,
    max_uid1: Optional[int] = None,
    prompt_style: str = "short",
    dataset_path: Optional[str] = None,
) -> Dataset:
    if selected_rows_path:
        if min_uid1 is not None and max_uid1 is not None and min_uid1 > max_uid1:
            raise ValueError(f"min_uid1 ({min_uid1}) must be <= max_uid1 ({max_uid1}).")
        selected_rows = load_jsonl_rows(selected_rows_path)
        if min_uid1 is not None or max_uid1 is not None:
            filtered_rows: List[Dict] = []
            for row in selected_rows:
                if "uid1" not in row:
                    raise ValueError(
                        f"Row is missing 'uid1' in {selected_rows_path}; cannot filter with "
                        f"min_uid1={min_uid1}, max_uid1={max_uid1}."
                    )
                uid1 = int(row["uid1"])
                if min_uid1 is not None and uid1 < min_uid1:
                    continue
                if max_uid1 is not None and uid1 > max_uid1:
                    continue
                filtered_rows.append(row)
            selected_rows = filtered_rows

        selected_indices: List[int] = []
        for row in selected_rows:
            if "uid" not in row:
                raise ValueError(f"Row is missing 'uid' in {selected_rows_path}; cannot recover original GSM8K order.")
            selected_indices.append(int(row["uid"]))

        if max_samples is not None and max_samples > 0:
            selected_indices = selected_indices[:max_samples]

        raw_dataset = load_gsm8k_from_path(dataset_path) if dataset_path else load_gsm8k(split)
        if not selected_indices:
            return Dataset.from_list([])

        max_index = len(raw_dataset) - 1
        for idx in selected_indices:
            if idx < 0 or idx > max_index:
                raise ValueError(
                    f"Selected uid={idx} from {selected_rows_path} is out of range for split='{split}' "
                    f"with dataset size {len(raw_dataset)}."
                )

        raw_dataset = raw_dataset.select(selected_indices)

        def _convert_selected(example: Dict) -> Dict:
            question = get_question(example)
            answer = get_gold_answer(example)
            return {
                "prompt": build_messages(question, prompt_style=prompt_style),
                "question": question,
                "answer": answer,
            }

        return raw_dataset.map(
            _convert_selected,
            remove_columns=raw_dataset.column_names,
            keep_in_memory=True,
            load_from_cache_file=False,
        )

    raw_dataset = load_gsm8k_from_path(dataset_path) if dataset_path else load_gsm8k(split)

    if max_samples is not None and max_samples > 0:
        max_samples = min(max_samples, len(raw_dataset))
        raw_dataset = raw_dataset.shuffle(seed=seed).select(range(max_samples))

    def _convert(example: Dict) -> Dict:
        question = get_question(example)
        answer = get_gold_answer(example)
        return {
            "prompt": build_messages(question, prompt_style=prompt_style),
            "question": question,
            "answer": answer,
        }

    return raw_dataset.map(_convert, remove_columns=raw_dataset.column_names)
