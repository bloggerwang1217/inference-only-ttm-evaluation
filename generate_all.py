"""Generate one clip per test caption with Stable Audio Open (sampling loop and CFG from generate.py / cfg.py).

Each caption k is saved as <out-dir>/wav/k.wav (for av-benchmark, which reads wav/flac only) and
<out-dir>/mp3/k.mp3 (for submission). The arguments are saved to <out-dir>/config.json.

    python generate_all.py --out-dir outputs/baseline
    python generate_all.py --out-dir outputs/apg --apg --norm-threshold 0   # APG; per-step norms to apg_norms.json
"""

import argparse
import csv
import json
from pathlib import Path

import soundfile as sf
import torch
from diffusers import StableAudioPipeline

from generate import add_apg_args, generate, make_apg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions", default="test.csv")
    parser.add_argument("--out-dir", default="outputs/baseline")
    parser.add_argument("--scale", type=float, default=7.0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0, help="same seed for every caption")
    parser.add_argument("--seed-per-caption", action="store_true",
                        help="caption k uses seed k instead of --seed, so every caption starts from its own noise")
    add_apg_args(parser)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "wav").mkdir(parents=True, exist_ok=True)
    (out_dir / "mp3").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    with open(args.captions, newline="") as f:
        rows = list(csv.DictReader(f))

    pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    sr = pipe.vae.config.sampling_rate

    apg = make_apg(args)
    peaks, norms = {}, {}
    for n, row in enumerate(rows, 1):
        seed = int(row["id"]) if args.seed_per_caption else args.seed
        audio = generate(pipe, row["caption"], args.scale, args.steps, args.seconds, seed, apg)
        if apg is not None:
            norms[row["id"]] = apg.norms
        audio = audio.T.float().cpu().numpy()  # (samples, channels)
        sf.write(out_dir / "wav" / f"{row['id']}.wav", audio, sr)
        sf.write(out_dir / "mp3" / f"{row['id']}.mp3", audio, sr, format="MP3")
        peaks[row["id"]] = float(abs(audio).max())
        print(f"[{n}/{len(rows)}] id={row['id']} peak={peaks[row['id']]:.3f}")

    if apg is not None:
        with open(out_dir / "apg_norms.json", "w") as f:
            json.dump(norms, f)
    clipped = [k for k, p in peaks.items() if p > 1.0]
    print(f"Saved {len(rows)} clips to {out_dir}; {len(clipped)} with peak > 1.0 (clipped on write): {clipped}")


if __name__ == "__main__":
    main()
