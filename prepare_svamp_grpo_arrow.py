import argparse
from pathlib import Path

from datasets import Dataset, load_dataset
from datasets.arrow_writer import ArrowWriter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="ChilleD/SVAMP")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--svamp_arrow", type=str, default="")
    parser.add_argument("--output", type=str, default="data/svamp-train-grpo.arrow")
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.svamp_arrow:
        raw_dataset = Dataset.from_file(args.svamp_arrow)
        source_desc = args.svamp_arrow
    else:
        raw_dataset = load_dataset(args.source, split=args.split)
        source_desc = f"{args.source}:{args.split}"

    def convert(example):
        question = (example.get("question_concat") or "").strip()
        if not question:
            body = (example.get("Body") or "").strip()
            tail = (example.get("Question") or "").strip()
            question = f"{body} {tail}".strip()

        return {
            "question": question,
            "answer": str(example["Answer"]).strip(),
            "svamp_id": str(example.get("ID", "")),
            "svamp_type": str(example.get("Type", "")),
        }

    converted = Dataset.from_list([convert(example) for example in raw_dataset])
    with ArrowWriter(path=str(output_path), schema=converted._data.table.schema) as writer:
        writer.write_table(converted._data.table)
        writer.finalize()

    print(f"SVAMP source: {source_desc}")
    print(f"Converted rows: {len(converted)}")
    print(f"Converted dataset: {output_path}")
    print("Columns: question, answer, svamp_id, svamp_type")


if __name__ == "__main__":
    main()
