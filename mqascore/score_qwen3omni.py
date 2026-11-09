"""Run third_party/MQAScore/scripts/score.py with Qwen3-Omni placed on the GPUs under a memory cap per GPU.

score.py loads Qwen3-Omni with device_map="auto". On GPUs 0-2 (48 + 24 + 24 GB) that filled GPU 1 to 22.5 GB, and
GPU 1 then ran out of memory while transformers converted the expert weights of the last MoE layers. This script
computes the device map with the caps in --max-memory, one per visible GPU, puts lm_head on the thinker's first GPU,
and passes the map to the same backend class; everything else is score.py unchanged, and all other arguments go to
score.py.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2 python mqascore/score_qwen3omni.py --max-memory 40GiB,19GiB,19GiB \
        --model qwen3omni --input items.jsonl --output qwen3omni.jsonl
"""

import argparse
import collections
import sys
from pathlib import Path

import torch
from accelerate import infer_auto_device_map, init_empty_weights
from transformers import AutoConfig, Qwen3OmniMoeForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "third_party" / "MQAScore" / "scripts"))
import score  # noqa: E402  (third_party/MQAScore/scripts/score.py; also puts the MQAScore root on sys.path)
from mqascore.models.qwen3_omni import Qwen3OmniBackend  # noqa: E402


def capped_device_map(max_memory):
    config = AutoConfig.from_pretrained(Qwen3OmniBackend.model_id)
    with init_empty_weights():
        model = Qwen3OmniMoeForConditionalGeneration._from_config(config, dtype=torch.bfloat16)
    caps = dict(enumerate(max_memory.split(",")))
    device_map = infer_auto_device_map(model, max_memory=caps, dtype=torch.bfloat16,
                                       no_split_module_classes=model._no_split_modules)
    off_gpu = sorted({d for d in device_map.values() if not isinstance(d, int)})
    if off_gpu:
        raise SystemExit(f"the model does not fit under --max-memory {max_memory}: some modules went to {off_gpu}")
    # The backend puts the Yes/No token ids on model.thinker.device (the GPU of the thinker's first module) and
    # indexes the logits with them, so the logits must come out on that GPU.
    device_map["thinker.lm_head"] = device_map[next(iter(device_map))]
    return device_map


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--max-memory", required=True, help="memory cap per visible GPU, e.g. 40GiB,19GiB,19GiB")
    args, score_args = parser.parse_known_args()
    if "qwen3omni" not in score_args:
        raise SystemExit("this wrapper only places Qwen3-Omni; pass --model qwen3omni")

    device_map = capped_device_map(args.max_memory)
    print("modules per GPU:", dict(sorted(collections.Counter(device_map.values()).items())), flush=True)
    score.load_backend = lambda name, **kwargs: Qwen3OmniBackend(device_map=device_map, **kwargs)
    sys.argv = [score.__file__] + score_args
    score.main()


if __name__ == "__main__":
    main()
