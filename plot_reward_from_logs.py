import argparse
import math
import tempfile
from pathlib import Path
from typing import List, Tuple

from plot_train_metrics import moving_average, parse_log


def infer_steps(records, steps_per_epoch: int) -> List[int]:
    steps: List[int] = []
    previous = 0
    for index, record in enumerate(records, start=1):
        epoch = record.get("epoch", math.nan)
        if math.isnan(epoch):
            step = previous + 1
        else:
            step = max(1, int(round(epoch * steps_per_epoch)))
        if step <= previous and len(records) <= steps_per_epoch:
            step = previous + 1
        steps.append(step)
        previous = step
    return steps


def get_metric_value(record, metric: str) -> float:
    if metric == "weighted_reward":
        correctness = record.get("rewards/gsm8k_correctness_reward/mean", math.nan)
        formatting = record.get("rewards/final_answer_format_reward/mean", math.nan)
        if math.isnan(correctness) or math.isnan(formatting):
            return math.nan
        return correctness + 0.1 * formatting
    return record.get(metric, math.nan)


def load_metric_series(log_path: Path, steps_per_epoch: int, metric: str) -> Tuple[List[int], List[float]]:
    records = parse_log(log_path)
    steps = infer_steps(records, steps_per_epoch)
    xs: List[int] = []
    ys: List[float] = []
    for step, record in zip(steps, records):
        value = get_metric_value(record, metric)
        if math.isnan(value):
            continue
        xs.append(step)
        ys.append(value)
    return xs, ys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", help="Training log files to plot.")
    parser.add_argument("--output", default="plots/reward_from_logs.png")
    parser.add_argument("--metric", default="reward", help="Metric to plot. Use weighted_reward for correctness + 0.1 * format.")
    parser.add_argument("--steps_per_epoch", type=int, default=2594)
    parser.add_argument("--smooth_window", type=int, default=25)
    return parser.parse_args()


def main():
    args = parse_args()

    import os

    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="matplotlib-", dir="/tmp"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(12, 5.5))
    for log_name in args.logs:
        log_path = Path(log_name)
        xs, ys = load_metric_series(log_path, args.steps_per_epoch, args.metric)
        if not xs:
            continue
        label = log_path.stem
        axis.plot(xs, ys, linewidth=0.9, alpha=0.25, label=f"{label} raw")
        if args.smooth_window > 1 and len(ys) >= 2:
            axis.plot(xs, moving_average(ys, args.smooth_window), linewidth=2.0, label=f"{label} ma{args.smooth_window}")

    axis.set_title(f"{args.metric} vs Training Step")
    axis.set_xlabel("Global step inferred from epoch")
    axis.set_ylabel(args.metric)
    if args.metric == "weighted_reward":
        axis.set_ylim(-0.05, 1.15)
    elif args.metric == "reward":
        axis.set_ylim(-0.05, 2.05)
    axis.grid(alpha=0.25, linestyle="--")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()
