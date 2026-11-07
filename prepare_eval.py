"""Build the reference set for av-benchmark from the retrieval result.

For each caption k, the MusicCaps clip retrieved for it (retrieval.json) is decoded and saved as
<out-dir>/wav/k.wav, so that av-benchmark pairs it with the generated k.wav by filename. A clip
retrieved for several captions is saved once per caption. The audio is written as 32-bit float wav
at its original sample rate and channels, so the decoded mp3 is neither re-quantized nor clipped.

The captions are saved as <out-dir>/captions.csv with the columns `name` and `caption`, the format
read by av-benchmark's extract_text.py.

    python prepare_eval.py --out-dir outputs/reference
"""

import argparse
import csv
import json
from pathlib import Path

import soundfile as sf
import torchaudio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions", default="test.csv")
    parser.add_argument("--retrieval", default="retrieval.json")
    parser.add_argument("--audio-dir", default="data/audio_mp3")
    parser.add_argument("--out-dir", default="outputs/reference")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "wav").mkdir(parents=True, exist_ok=True)

    with open(args.captions, newline="") as f:
        rows = list(csv.DictReader(f))
    with open(args.retrieval) as f:
        retrieval = json.load(f)

    for row in rows:
        waveform, sr = torchaudio.load(Path(args.audio_dir) / f"{retrieval[row['id']]}.mp3")
        sf.write(out_dir / "wav" / f"{row['id']}.wav", waveform.T.numpy(), sr, subtype="FLOAT")

    with open(out_dir / "captions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "caption"])
        writer.writeheader()
        writer.writerows({"name": row["id"], "caption": row["caption"]} for row in rows)

    print(f"Saved {len(rows)} reference clips and captions to {out_dir}")


if __name__ == "__main__":
    main()
