"""Table of the guidance-scale sweep: one row per scale, from the summary.json and per_clip.csv of each run.

Columns: FD-VGG; KL-PASST-softmax and LAION-CLAP-Score as mean and 95% t-interval; the same two as the mean
difference from the run at --paired-with, clip by clip (every run uses seed k for caption k, so clip k of two runs
differs only in the guidance scale), with a 95% t-interval; and the number of clips that were clipped when saved,
counted as clips with a sample at 16-bit full scale. Writes the table to --out as CSV and prints it.

    python summarize_sweep.py
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import stats

DEFAULT_RUNS = {0: "outputs/uncond_seed_per_caption", 7: "outputs/baseline_seed_per_caption",
                **{s: f"outputs/sweep/scale_{s}" for s in [1, 3, 5, 9, 11, 13]}}
METRICS = ["KL-PASST-softmax", "LAION-CLAP-Score"]


def mean_ci(values, level=0.95):
    n = len(values)
    half = stats.t.ppf((1 + level) / 2, n - 1) * values.std(ddof=1) / math.sqrt(n)
    return values.mean(), values.mean() - half, values.mean() + half


def per_clip(run_dir):
    with open(Path(run_dir) / "eval_cache" / "per_clip.csv", newline="") as f:
        rows = {r["name"]: r for r in csv.DictReader(f)}
    names = sorted(rows, key=int)
    return names, {m: np.array([float(rows[k][m]) for k in names]) for m in METRICS}


def clipped_clips(run_dir):
    wavs = sorted((Path(run_dir) / "wav").glob("*.wav"), key=lambda p: int(p.stem))
    return sum(int(np.abs(sf.read(p, dtype="int16")[0].astype(np.int32)).max() >= 32767) for p in wavs), len(wavs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", default=None, help="scale=run_dir pairs (default: the sweep runs)")
    parser.add_argument("--paired-with", type=float, default=7, help="scale whose run the differences are taken from")
    parser.add_argument("--out", default="outputs/sweep/summary.csv")
    args = parser.parse_args()

    runs = ({float(k): v for k, v in (r.split("=", 1) for r in args.runs)} if args.runs
            else {float(k): v for k, v in DEFAULT_RUNS.items()})
    base_names, base = per_clip(runs[args.paired_with])

    table = []
    for scale in sorted(runs):
        run_dir = runs[scale]
        with open(Path(run_dir) / "eval_cache" / "summary.json") as f:
            summary = json.load(f)
        names, values = per_clip(run_dir)
        assert names == base_names, f"{run_dir} does not hold the same clips as the paired run"
        clipped, n = clipped_clips(run_dir)
        row = {"guidance_scale": f"{scale:g}", "run_dir": run_dir, "FD-VGG": f"{summary['FD-VGG']:.4f}"}
        for m in METRICS:
            mean, lo, hi = mean_ci(values[m])
            dmean, dlo, dhi = mean_ci(values[m] - base[m])
            row.update({m: f"{mean:.4f}", f"{m} CI low": f"{lo:.4f}", f"{m} CI high": f"{hi:.4f}",
                        f"{m} minus scale {args.paired_with:g}": f"{dmean:+.4f}",
                        f"{m} difference CI low": f"{dlo:+.4f}", f"{m} difference CI high": f"{dhi:+.4f}"})
        row["clipped clips"] = f"{clipped}/{n}"
        table.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    p = f"{args.paired_with:g}"
    print(f"{'scale':>5} | {'FD-VGG':>7} | {'KL mean [95% CI]':>24} | {'KL - scale ' + p + ' [95% CI]':>30} | "
          f"{'CLAP mean [95% CI]':>24} | {'CLAP - scale ' + p + ' [95% CI]':>30} | clipped")
    for r in table:
        kl, cl = METRICS
        print(f"{r['guidance_scale']:>5} | {r['FD-VGG']:>7} | "
              f"{r[kl]} [{r[kl + ' CI low']}, {r[kl + ' CI high']}] | "
              f"{r[kl + ' minus scale ' + p]} [{r[kl + ' difference CI low']}, {r[kl + ' difference CI high']}] | "
              f"{r[cl]} [{r[cl + ' CI low']}, {r[cl + ' CI high']}] | "
              f"{r[cl + ' minus scale ' + p]} [{r[cl + ' difference CI low']}, {r[cl + ' difference CI high']}] | "
              f"{r['clipped clips']}")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
