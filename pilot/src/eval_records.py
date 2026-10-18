"""MLX evaluation that writes per-item records in the same schema as colab/pilot_t4c.py.

Records (one JSON line per test item): id, gold, pred, tokens, text, twins[{kind, gold, pred, tokens,
text}], and for the first 100 items with a parseable answer trunc_0 / trunc_50 {raw, pred}.
Analyse with src/analyze.py.

Usage: python -I src/eval_records.py --model mlx-community/Qwen3-4B-4bit [--adapter DIR] \
           --test data/test.jsonl --out results/rerun_mlx/records/base.jsonl
"""
import argparse
import json
import os
import sys
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import batch_generate

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import chat_prompt, load_jsonl, parse_answer  # noqa: E402


def generate(model, tokenizer, prompts, max_tokens, batch_size):
    outs = []
    for i in range(0, len(prompts), batch_size):
        toks = [tokenizer.encode(p, add_special_tokens=False) for p in prompts[i:i + batch_size]]
        res = batch_generate(model, tokenizer, toks, max_tokens=max_tokens, completion_batch_size=batch_size,
                             prefill_batch_size=batch_size, return_token_ids=True)
        outs.extend(zip(res.texts, [len(t) for t in res.token_ids]))
        mx.clear_cache()
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--trunc_n", type=int, default=100)
    args = ap.parse_args()

    t0 = time.time()
    model, tokenizer = load(args.model, adapter_path=args.adapter)
    test = load_jsonl(args.test)
    flat, index = [], []
    for k, r in enumerate(test):
        flat.append(chat_prompt(tokenizer, r["prompt"])); index.append((k, -1))
        for j, tw in enumerate(r.get("twins", [])):
            flat.append(chat_prompt(tokenizer, tw["prompt"])); index.append((k, j))
    gens = dict(zip(index, generate(model, tokenizer, flat, 512, args.batch_size)))
    records = []
    for k, r in enumerate(test):
        text, n = gens[(k, -1)]
        rec = {"id": r["id"], "gold": r["answer"], "pred": parse_answer(text), "tokens": n, "text": text, "twins": []}
        for j, tw in enumerate(r.get("twins", [])):
            ttext, tn = gens[(k, j)]
            rec["twins"].append({"kind": tw["kind"], "gold": tw["answer"], "pred": parse_answer(ttext),
                                 "tokens": tn, "text": ttext})
        records.append(rec)
    cand = [k for k, rec in enumerate(records) if rec["pred"] is not None][:args.trunc_n]
    for frac in (0.0, 0.5):
        prompts = []
        for k in cand:
            text = records[k]["text"]
            reasoning = text[:text.upper().rfind("FINAL ANSWER")]
            prompts.append(chat_prompt(tokenizer, test[k]["prompt"]) + reasoning[:int(len(reasoning) * frac)]
                           + "\n\nFINAL ANSWER:")
        for (o, _), k in zip(generate(model, tokenizer, prompts, 24, args.batch_size), cand):
            records[k][f"trunc_{int(frac * 100)}"] = {"raw": o, "pred": parse_answer("FINAL ANSWER:" + o)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    acc = sum(rec["pred"] == sorted(rec["gold"]) for rec in records) / len(records)
    print(json.dumps({"out": args.out, "acc": acc, "minutes": round((time.time() - t0) / 60, 1)}), flush=True)


if __name__ == "__main__":
    main()
