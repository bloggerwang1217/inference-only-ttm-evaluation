"""Summarize one av-benchmark run: FD-VGG, plus KL-PASST-softmax and LAION-CLAP-Score with 95% confidence intervals.

av-benchmark reports KL and CLAP only as means over the paired clips. This script recomputes the value of
each pair from the feature caches with the same formulas (av_bench/metrics/kl.py and av_bench/evaluate.py),
checks that their means match av-benchmark's output_metrics.json, and adds a 95% t-interval for each mean.
FD-VGG compares the two sets as a whole, so it has no per-pair values and gets no interval here.

The value of each pair is saved to <run-cache>/per_clip.csv, and the summary to <run-cache>/summary.json.

    python summarize_eval.py --ref-cache outputs/reference/cache --run-cache outputs/baseline/eval_cache
"""

import argparse
import csv
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from scipy import stats


def load(cache, name):
    return torch.load(cache / f"{name}.pth", weights_only=True)


def mean_ci(values, level=0.95):
    """Mean, standard deviation, and the half-width of the t-interval for the mean."""
    n = len(values)
    sd = values.std(unbiased=True).item()
    half = stats.t.ppf((1 + level) / 2, n - 1) * sd / math.sqrt(n)
    return values.mean().item(), sd, half


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-cache", type=Path, default=Path("outputs/reference/cache"))
    parser.add_argument("--run-cache", type=Path, required=True)
    args = parser.parse_args()
    ref, run = args.ref_cache, args.run_cache

    ref_logits, ref_text, ref_vggish = (load(ref, n) for n in ["passt_logits", "clap_laion_text", "vggish_features"])
    run_logits, run_audio, run_vggish = (load(run, n) for n in ["passt_logits", "clap_laion_audio", "vggish_features"])

    # every cache must hold the same clips, so that each generated clip has its reference and caption
    names = sorted(run_logits, key=int)
    for cache, name, d in [(ref, "passt_logits", ref_logits), (ref, "clap_laion_text", ref_text),
                           (ref, "vggish_features", ref_vggish), (run, "clap_laion_audio", run_audio),
                           (run, "vggish_features", run_vggish)]:
        assert set(d) == set(names), f"{cache / name}.pth holds {len(d)} clips, not the same {len(names)} as the run"

    # KL(reference || generated) between the softmax distributions over the 527 AudioSet classes of PaSST
    gt = torch.stack([ref_logits[k] for k in names]).log_softmax(dim=1)
    pred = torch.stack([run_logits[k] for k in names]).log_softmax(dim=1)
    kl = (gt.exp() * (gt - pred)).sum(dim=1)

    # cosine similarity between the CLAP embeddings of caption k and generated clip k
    clap = F.cosine_similarity(torch.stack([ref_text[k] for k in names]),
                               torch.stack([run_audio[k] for k in names]), dim=-1)

    with open(run / "output_metrics.json") as f:
        metrics = json.load(f)
    summary = {"num_pairs": len(names), "FD-VGG": metrics["FD-VGG"]}
    for key, values in [("KL-PASST-softmax", kl), ("LAION-CLAP-Score", clap)]:
        mean, sd, half = mean_ci(values)
        assert math.isclose(mean, metrics[key], rel_tol=1e-4), f"{key}: per-pair mean {mean} != av-benchmark {metrics[key]}"
        summary[key] = {"mean": mean, "sd": sd, "ci95": [mean - half, mean + half]}

    with open(run / "per_clip.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "KL-PASST-softmax", "LAION-CLAP-Score"])
        writer.writerows([k, f"{a:.6f}", f"{b:.6f}"] for k, a, b in zip(names, kl.tolist(), clap.tolist()))
    with open(run / "summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print(f"{len(names)} pairs")
    print(f"FD-VGG: {summary['FD-VGG']:.4f}")
    for key in ["KL-PASST-softmax", "LAION-CLAP-Score"]:
        s = summary[key]
        lo, hi = s["ci95"]
        print(f"{key}: {s['mean']:.4f} (95% CI {lo:.4f} to {hi:.4f}, SD {s['sd']:.4f})")


if __name__ == "__main__":
    main()
