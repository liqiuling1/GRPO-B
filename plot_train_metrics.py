import argparse
import ast
import csv
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional


DEFAULT_METRICS = [
    "rewards/gsm8k_correctness_reward/mean",
    "rewards/final_answer_format_reward/mean",
    "reward",
    "reward_std",
    "frac_reward_zero_std",
    "entropy",
    "completions/mean_length",
    "completions/clipped_ratio",
    "grad_norm",
    "update_proxy",
    "advantage/std",
    "advantage/abs_mean",
    "advantage/zero_frac",
    "prompt/advantage_zero_group_frac",
    "prompt/correctness_reward_std_mean",
    "prompt/correctness_reward_zero_std_frac",
    "prompt/correctness_adv_zero_group_frac",
    "loss",
    "learning_rate",
    "step_time",
]

DIAGNOSTIC_METRICS = [
    "grad_norm",
    "update_proxy",
    "reward_std",
    "frac_reward_zero_std",
    "advantage/std",
    "advantage/abs_mean",
    "advantage/zero_frac",
    "prompt/advantage_zero_group_frac",
    "prompt/correctness_reward_std_mean",
    "prompt/correctness_reward_zero_std_frac",
    "prompt/correctness_adv_zero_group_frac",
]


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def parse_float(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def clean_line(line: str) -> str:
    return ANSI_ESCAPE_RE.sub("", line).strip()


def is_metric_dict(line: str) -> bool:
    return line.startswith("{") and line.endswith("}") and "'epoch'" in line


def parse_log(log_path: Path) -> List[Dict[str, float]]:
    records: List[Dict[str, float]] = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            for part in raw_line.split("\r"):
                line = clean_line(part)
                if not is_metric_dict(line):
                    continue
                payload = ast.literal_eval(line)
                record: Dict[str, float] = {}
                for key, value in payload.items():
                    record[key] = parse_float(value)
                if "epoch" not in record:
                    continue
                grad_norm = record.get("grad_norm", math.nan)
                learning_rate = record.get("learning_rate", math.nan)
                if not math.isnan(grad_norm) and not math.isnan(learning_rate):
                    record["update_proxy"] = grad_norm * learning_rate
                records.append(record)

    for index, record in enumerate(records, start=1):
        record["step"] = float(index)
    return records


def collect_metric_names(records: Iterable[Dict[str, float]]) -> List[str]:
    metric_names = set()
    for record in records:
        metric_names.update(record.keys())
    metric_names.discard("step")
    return sorted(metric_names)


def export_csv(records: List[Dict[str, float]], output_path: Path) -> None:
    metric_names = ["step"] + collect_metric_names(records)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=metric_names)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def sanitize_metric_name(metric: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", metric).strip("_")


def moving_average(values: List[float], window: int) -> List[float]:
    if window <= 1:
        return values[:]

    smoothed: List[float] = []
    history: List[float] = []
    for value in values:
        history.append(value)
        if len(history) > window:
            history.pop(0)
        valid = [v for v in history if not math.isnan(v)]
        smoothed.append(sum(valid) / len(valid) if valid else math.nan)
    return smoothed


def plot_metric(
    records: List[Dict[str, float]],
    metric: str,
    output_dir: Path,
    smooth_window: int,
) -> Optional[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it with `pip install matplotlib` "
            "or `conda install matplotlib`."
        ) from exc

    xs: List[float] = []
    ys: List[float] = []
    for record in records:
        value = record.get(metric, math.nan)
        if math.isnan(value):
            continue
        xs.append(record["step"])
        ys.append(value)

    if not xs:
        return None

    plt.figure(figsize=(10, 5))
    plt.plot(xs, ys, linewidth=1.2, alpha=0.35, label="raw")
    if smooth_window > 1 and len(ys) >= 2:
        plt.plot(xs, moving_average(ys, smooth_window), linewidth=2.0, label=f"ma{smooth_window}")
        plt.legend()
    plt.title(metric)
    plt.xlabel("Step")
    plt.ylabel(metric)
    plt.grid(alpha=0.25, linestyle="--")
    plt.tight_layout()

    output_path = output_dir / f"{sanitize_metric_name(metric)}.png"
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def plot_diagnostics_overview(
    records: List[Dict[str, float]],
    metrics: List[str],
    output_dir: Path,
    smooth_window: int,
) -> Optional[Path]:
    available_metrics = [metric for metric in metrics if any(not math.isnan(record.get(metric, math.nan)) for record in records)]
    if not available_metrics:
        return None

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(available_metrics), 1, figsize=(12, 3.2 * len(available_metrics)), sharex=True)
    if len(available_metrics) == 1:
        axes = [axes]

    for axis, metric in zip(axes, available_metrics):
        xs: List[float] = []
        ys: List[float] = []
        for record in records:
            value = record.get(metric, math.nan)
            if math.isnan(value):
                continue
            xs.append(record["step"])
            ys.append(value)

        if not xs:
            continue

        axis.plot(xs, ys, linewidth=1.0, alpha=0.3, label="raw")
        if smooth_window > 1 and len(ys) >= 2:
            axis.plot(xs, moving_average(ys, smooth_window), linewidth=1.8, label=f"ma{smooth_window}")
            axis.legend(loc="upper right")
        axis.set_ylabel(metric)
        axis.grid(alpha=0.25, linestyle="--")

    axes[-1].set_xlabel("Step")
    fig.suptitle("GRPO Diagnostics Overview")
    fig.tight_layout()

    output_path = output_dir / "diagnostics_overview.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def write_summary(records: List[Dict[str, float]], metrics: List[str], output_path: Path) -> None:
    lines = []
    if not records:
        lines.append("No metric records found.")
    else:
        lines.append(f"records={len(records)}")
        lines.append(f"first_step=1")
        lines.append(f"last_step={int(records[-1]['step'])}")
        for metric in metrics:
            values = [record.get(metric, math.nan) for record in records]
            values = [value for value in values if not math.isnan(value)]
            if not values:
                continue
            lines.append(
                f"{metric}: min={min(values):.6g}, max={max(values):.6g}, "
                f"last={values[-1]:.6g}, mean={sum(values) / len(values):.6g}"
            )
        zero_grad_steps = sum(1 for record in records if math.isclose(record.get("grad_norm", math.nan), 0.0, abs_tol=1e-12))
        if zero_grad_steps:
            lines.append(f"zero_grad_steps={zero_grad_steps}")
        all_have_zero_std = [
            record
            for record in records
            if math.isclose(record.get("grad_norm", math.nan), 0.0, abs_tol=1e-12)
            and not math.isnan(record.get("frac_reward_zero_std", math.nan))
        ]
        if all_have_zero_std:
            matched = sum(
                1
                for record in all_have_zero_std
                if math.isclose(record.get("frac_reward_zero_std", math.nan), 1.0, abs_tol=1e-12)
            )
            lines.append(f"zero_grad_with_frac_reward_zero_std_eq_1={matched}/{len(all_have_zero_std)}")
        zero_adv_records = [
            record
            for record in records
            if math.isclose(record.get("advantage/zero_frac", math.nan), 1.0, abs_tol=1e-12)
        ]
        if zero_adv_records:
            lines.append(f"advantage_zero_frac_eq_1_steps={len(zero_adv_records)}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--metrics", nargs="*", default=None)
    parser.add_argument("--list_metrics", action="store_true")
    parser.add_argument("--smooth_window", type=int, default=25)
    parser.add_argument("--csv_only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    log_file = Path(args.log_file)
    records = parse_log(log_file)
    if not records:
        raise ValueError(f"No metric dicts were found in {log_file}")

    available_metrics = collect_metric_names(records)
    if args.list_metrics:
        for metric in available_metrics:
            print(metric)
        return

    metrics = args.metrics or [metric for metric in DEFAULT_METRICS if metric in available_metrics]
    missing_metrics = [metric for metric in metrics if metric not in available_metrics]
    if missing_metrics:
        raise ValueError(f"Unknown metrics: {missing_metrics}")

    output_dir = Path(args.output_dir) if args.output_dir else Path("plots") / log_file.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="matplotlib-", dir="/tmp"))

    csv_path = output_dir / "metrics.csv"
    export_csv(records, csv_path)
    write_summary(records, metrics, output_dir / "summary.txt")

    if args.csv_only:
        print(f"Saved CSV to {csv_path}")
        print(f"Saved summary to {output_dir / 'summary.txt'}")
        return

    created_files = []
    diagnostics_path = plot_diagnostics_overview(
        records=records,
        metrics=DIAGNOSTIC_METRICS,
        output_dir=output_dir,
        smooth_window=args.smooth_window,
    )
    if diagnostics_path is not None:
        created_files.append(diagnostics_path)
    for metric in metrics:
        output_path = plot_metric(
            records=records,
            metric=metric,
            output_dir=output_dir,
            smooth_window=args.smooth_window,
        )
        if output_path is not None:
            created_files.append(output_path)

    print(f"Saved CSV to {csv_path}")
    print(f"Saved summary to {output_dir / 'summary.txt'}")
    for path in created_files:
        print(path)


if __name__ == "__main__":
    main()
