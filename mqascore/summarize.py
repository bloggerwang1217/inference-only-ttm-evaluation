"""Summarize MQAScore (Qwen3-Omni) over runs: overall and per dimension, with 95% t-intervals.

Reads <run-dir>/mqascore/qwen3omni.jsonl for each --runs entry (label=run_dir). For every run it reports
  - MQAScore: the repo's item score (unweighted mean of P(Yes) over all concepts of a caption), averaged over captions;
  - each dimension: the item's mean P(Yes) over its concepts of that dimension, averaged over the captions that have
    such concepts (the paper reports dimensions separately and skips captions without the dimension);
and, against --paired-with, the mean of the caption-by-caption difference with a 95% t-interval (every run scores the
same concepts, since they come from the captions). Writes --out as CSV and prints the table.

    python mqascore/summarize.py
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

DIMENSIONS = ["genre", "instrument", "mood_theme", "vocal", "context"]


def mean_ci(x, level=0.95):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float(x.mean()) if len(x) else float("nan"), float("nan"), float("nan")
    half = stats.t.ppf((1 + level) / 2, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))
    return x.mean(), x.mean() - half, x.mean() + half


def load(run_dir):
    items = {}
    with open(Path(run_dir) / "mqascore" / "qwen3omni.jsonl") as f:
        for line in f:
            r = json.loads(line)
            items[r["id"]] = {"MQAScore": r["mean_score"], **r["attribute_mean"]}
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=["5=outputs/sweep/scale_5", "7=outputs/baseline_seed_per_caption",
                                                      "9=outputs/sweep/scale_9"])
    parser.add_argument("--paired-with", default="7")
    parser.add_argument("--out", default="outputs/mqascore/summary.csv")
    args = parser.parse_args()

    runs = dict(r.split("=", 1) for r in args.runs)
    data = {label: load(d) for label, d in runs.items()}
    ids = sorted(data[args.paired_with], key=int)
    for label, items in data.items():
        assert sorted(items, key=int) == ids, f"run {label} does not have the same captions as {args.paired_with}"

    rows = []
    for metric in ["MQAScore"] + DIMENSIONS:
        have = [i for i in ids if metric in data[args.paired_with][i]]
        for label in runs:
            x = [data[label][i][metric] for i in have]
            m, lo, hi = mean_ci(x)
            row = {"metric": metric, "guidance_scale": label, "n_captions": len(have), "mean": m, "ci_low": lo, "ci_high": hi}
            if label != args.paired_with:
                dm, dlo, dhi = mean_ci([data[label][i][metric] - data[args.paired_with][i][metric] for i in have])
                row.update({"diff_mean": dm, "diff_ci_low": dlo, "diff_ci_high": dhi})
            rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "guidance_scale", "n_captions", "mean", "ci_low", "ci_high",
                                               "diff_mean", "diff_ci_low", "diff_ci_high"])
        writer.writeheader()
        writer.writerows(rows)

    p = args.paired_with
    print(f"{'metric':<11} {'scale':>5} {'n':>4}  {'mean [95% CI]':>24}  {'minus scale ' + p + ' [95% CI]':>30}")
    for r in rows:
        diff = (f"{r['diff_mean']:+.3f} [{r['diff_ci_low']:+.3f}, {r['diff_ci_high']:+.3f}]" if "diff_mean" in r else "—")
        print(f"{r['metric']:<11} {r['guidance_scale']:>5} {r['n_captions']:>4}  "
              f"{r['mean']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]  {diff:>30}")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
