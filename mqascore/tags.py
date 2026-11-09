"""Split each test caption into MQAScore tags with Qwen3-4B-Instruct-2507, run with transformers instead of vLLM.

The instruction (third_party/MQAScore/instructions/instruction_llm_tag_extraction.txt), the model, greedy decoding,
max_new_tokens and the JSON parsing (extract_json) are the same as third_party/MQAScore/scripts/extract_tags.py.
Writes one line per caption, {"id", "text", "tags"}, to --output; captions whose output cannot be parsed go to
<output>.failures.jsonl with the raw output. The arguments are saved to <output>.config.json.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python mqascore/tags.py
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "third_party" / "MQAScore" / "scripts"))
from extract_tags import extract_json  # noqa: E402  (this import also puts the MQAScore root on sys.path)
from mqascore.prompts import tag_extraction_instruction  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions", default="test.csv")
    parser.add_argument("--output", default="outputs/mqascore/tags.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_name(out.name + ".config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    with open(args.captions, newline="") as f:
        rows = list(csv.DictReader(f))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    instruction = tag_extraction_instruction()

    records, failures = [], []
    for n, row in enumerate(rows, 1):
        messages = [{"role": "system", "content": instruction}, {"role": "user", "content": row["caption"]}]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=True,
                                               return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        raw = tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        item = {"id": row["id"], "text": row["caption"]}
        try:
            records.append({**item, "tags": extract_json(raw)})
            status = "ok"
        except (ValueError, json.JSONDecodeError) as e:
            failures.append({**item, "error": str(e), "raw_output": raw})
            status = f"failed: {e}"
        print(f"[{n}/{len(rows)}] id={row['id']} {status}", flush=True)

    with open(out, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    fail_path = out.with_name(out.name + ".failures.jsonl")
    if failures:
        with open(fail_path, "w") as f:
            for r in failures:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved tags for {len(records)} captions to {out}; {len(failures)} failed"
          + (f" (see {fail_path})" if failures else ""))


if __name__ == "__main__":
    main()
