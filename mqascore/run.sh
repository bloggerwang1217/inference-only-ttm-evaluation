#!/usr/bin/env bash
# Score one generated run with MQAScore (third_party/MQAScore, unmodified) and Qwen3-Omni, using the tags from
# mqascore/tags.py. Writes <run-dir>/mqascore/items.jsonl (the scorer's input) and qwen3omni.jsonl (its output, with
# qwen3omni.jsonl.meta.json recording the settings). Extra arguments go to score.py, e.g. --limit 2 for a dry run.
# Qwen3-Omni in bf16 needs about 70 GB of GPU memory; mqascore/score_qwen3omni.py shards it across the visible GPUs under the
# caps in MAX_MEMORY (default for GPUs 0-2: 40GiB,19GiB,19GiB).
# Run from the deepmir_hw3_eval environment:
#   CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2 bash mqascore/run.sh outputs/baseline_seed_per_caption
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_DIR=${1:?usage: bash mqascore/run.sh <run-dir> [score.py arguments]}
TAGS=${TAGS:-outputs/mqascore/tags.jsonl}
OUT_DIR=$RUN_DIR/mqascore
mkdir -p "$OUT_DIR"

python - "$TAGS" "$RUN_DIR" "$OUT_DIR/items.jsonl" <<'EOF'
import json, sys
from pathlib import Path
tags_path, run_dir, out_path = sys.argv[1:]
with open(tags_path) as f, open(out_path, "w") as g:
    for line in f:
        item = json.loads(line)
        audio = Path(run_dir, "wav", f"{item['id']}.wav").resolve()
        assert audio.exists(), audio
        g.write(json.dumps({"id": item["id"], "audio": str(audio), "text": item["text"], "tags": item["tags"]},
                           ensure_ascii=False) + "\n")
EOF

# --fix-seconds 10 is Qwen3-Omni's default length; it is passed explicitly because score.py crashes on its own
# default value "default" (float("default") is evaluated as the fallback of dict.get).
python mqascore/score_qwen3omni.py --max-memory "${MAX_MEMORY:-40GiB,19GiB,19GiB}" --model qwen3omni \
    --input "$OUT_DIR/items.jsonl" --output "$OUT_DIR/qwen3omni.jsonl" \
    --cache-dir "$OUT_DIR/audio_cache" --fix-seconds 10 "${@:2}"
