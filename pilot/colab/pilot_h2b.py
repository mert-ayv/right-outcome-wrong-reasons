"""H2b pilot (see ../PREREGISTRATION_H2B.md): finer, gated proof-step reward after an SFT warm start.

Derived from pilot_h2.py. All prompts carry the v2 step instruction (src/proof_steps_v2.py).

Jobs:
  --job base                             evaluate the untrained model
  --job sft                              warm start: SFT on solver-written derivations for 400 training
                                         cases (100 steps), then evaluate
  --job grpo --reward outcome  --seed S  from the SFT adapter: reward 1 for a correct final answer
  --job grpo --reward process  --seed S  from the SFT adapter: 0.5 x outcome + 0.5 x step_score_v2
GRPO: 120 steps on the other 600 training cases; checkpoint 120 is evaluated.

Run with: UNSLOTH_VLLM_NO_FLASHINFER=1 VLLM_USE_FLASHINFER_SAMPLER=0 H1_OUT=... python -u pilot_h2b.py --job ...
"""
import argparse
import json
import os
import re
import shutil
import time
import zipfile

ap = argparse.ArgumentParser()
ap.add_argument("--job", required=True, choices=["base", "sft", "grpo"])
ap.add_argument("--reward", default="outcome", choices=["outcome", "process"])
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--steps", type=int, default=120)
ap.add_argument("--sft_steps", type=int, default=100)
args_cli = ap.parse_args()

OUT = os.environ.get("H1_OUT", "/content/h1_rerun")  # Kaggle: /kaggle/working/h1_rerun
os.makedirs(f"{OUT}/records", exist_ok=True)
_final = {"base": "h2b-base", "sft": f"h2b-sft-step{args_cli.sft_steps}"}.get(
    args_cli.job, f"h2b-{args_cli.reward}-s{args_cli.seed}-step{args_cli.steps}")
if os.path.exists(f"{OUT}/records/{_final}.jsonl"):
    print(f"SKIP {_final}: records already exist", flush=True)
    raise SystemExit(0)
with zipfile.ZipFile("pilot_data.zip") as z:
    z.extractall("data")
train = [json.loads(l) for l in open("data/train.jsonl")]
test = [json.loads(l) for l in open("data/test.jsonl")]

import sys  # noqa: E402
sys.path.insert(0, os.getcwd())
from proof_steps_v2 import full_derivation_correct, gold_steps_v2, solver_trace, step_score_v2, tag_prompt_v2  # noqa: E402
import random  # noqa: E402

for r in train + test:
    r["prompt"] = tag_prompt_v2(r["prompt"])
    r["steps"] = gold_steps_v2(r["case"])[0]
    for tw in r.get("twins", []):
        tw["prompt"] = tag_prompt_v2(tw["prompt"])
_order = list(range(len(train)))
random.Random(0).shuffle(_order)
train_sft = [train[i] for i in _order[:400]]
train_rl = [train[i] for i in _order[400:]]

from unsloth import FastLanguageModel  # noqa: E402

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-4B", max_seq_length=1700, load_in_4bit=True,
    fast_inference=True, max_lora_rank=16, gpu_memory_utilization=0.55)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=32, lora_dropout=0.0, random_state=args_cli.seed,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth")

from vllm import SamplingParams  # noqa: E402

JURISDICTIONS = ["Arizona", "Illinois", "Oregon", "Pennsylvania", "Wisconsin", "Federal"]


def parse_answer(text):
    idx = text.upper().rfind("FINAL ANSWER")
    if idx < 0:
        return None
    tail = text[idx + len("FINAL ANSWER"):].lstrip(" :*\n").split("\n")[0]
    found = {j for j in JURISDICTIONS if re.search(rf"\b{j}\b", tail, re.IGNORECASE)}
    return sorted(found) if found else None


def chat_prompt(user_text):
    return tokenizer.apply_chat_template([{"role": "user", "content": user_text}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)


def generate(prompts, max_tokens, lora=None):
    outs = model.fast_generate(prompts, sampling_params=SamplingParams(temperature=0.0, max_tokens=max_tokens),
                               lora_request=lora)
    return [(o.outputs[0].text, len(o.outputs[0].token_ids)) for o in outs]


def refresh_zip():
    if os.path.exists(f"{OUT}.zip"):
        os.remove(f"{OUT}.zip")
    shutil.make_archive(OUT, "zip", OUT)


def evaluate(tag, lora=None, trunc_n=100):
    if os.path.exists(f"{OUT}/records/{tag}.jsonl"):
        print(f"SKIP evaluation {tag}: records already exist", flush=True)
        return
    t0 = time.time()
    flat, index = [], []
    for k, r in enumerate(test):
        flat.append(chat_prompt(r["prompt"])); index.append((k, -1))
        for j, tw in enumerate(r.get("twins", [])):
            flat.append(chat_prompt(tw["prompt"])); index.append((k, j))
    gens = dict(zip(index, generate(flat, 512, lora)))
    records = []
    for k, r in enumerate(test):
        text, n = gens[(k, -1)]
        rec = {"id": r["id"], "gold": r["answer"], "pred": parse_answer(text), "tokens": n, "text": text, "twins": [],
               "step_score": step_score_v2(text, r["steps"]), "full_derivation": full_derivation_correct(text, r["steps"])}
        for j, tw in enumerate(r.get("twins", [])):
            ttext, tn = gens[(k, j)]
            rec["twins"].append({"kind": tw["kind"], "gold": tw["answer"], "pred": parse_answer(ttext),
                                 "tokens": tn, "text": ttext})
        records.append(rec)
    # Early answering on the first trunc_n originals whose full answer parsed
    cand = [k for k, rec in enumerate(records) if rec["pred"] is not None][:trunc_n]
    for frac in (0.0, 0.5):
        prompts = []
        for k in cand:
            text = records[k]["text"]
            reasoning = text[:text.upper().rfind("FINAL ANSWER")]
            prompts.append(chat_prompt(test[k]["prompt"]) + reasoning[:int(len(reasoning) * frac)] + "\n\nFINAL ANSWER:")
        for (o, _), k in zip(generate(prompts, 24, lora), cand):
            records[k][f"trunc_{int(frac * 100)}"] = {"raw": o, "pred": parse_answer("FINAL ANSWER:" + o)}
    with open(f"{OUT}/records/{tag}.jsonl", "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    # Self-contained summary (CF on the model's own correct items; joint CF is computed offline)
    ok = [rec for rec in records if rec["pred"] == sorted(rec["gold"])]
    op = [tw["pred"] != rec["pred"] for rec in ok for tw in rec["twins"] if tw["kind"] == "operative"]
    inv = [tw["pred"] == rec["pred"] for rec in ok for tw in rec["twins"] if tw["kind"] == "invariant"]
    summ = {"tag": tag, "acc": len(ok) / len(records),
            "cf_own_operative": sum(op) / max(1, len(op)), "cf_own_invariant": sum(inv) / max(1, len(inv)),
            "pairs": [len(op), len(inv)], "minutes": round((time.time() - t0) / 60, 1),
            "step_score": sum(rec["step_score"] for rec in records) / len(records),
            "full_derivation": sum(rec["full_derivation"] for rec in records) / len(records)}
    for frac in (0, 50):
        tr = [rec[f"trunc_{frac}"] for rec in records if f"trunc_{frac}" in rec]
        summ[f"trunc{frac}_agree"] = sum(t["pred"] == rec["pred"] for rec in records
                                         for t in [rec.get(f"trunc_{frac}")] if t) / max(1, len(tr))
        summ[f"trunc{frac}_parse_fail"] = sum(t["pred"] is None for t in tr) / max(1, len(tr))
    with open(f"{OUT}/summaries.jsonl", "a") as f:
        f.write(json.dumps(summ) + "\n")
    refresh_zip()
    print("RESULT", json.dumps(summ), flush=True)


if args_cli.job == "base":
    evaluate("h2b-base")
else:
    from datasets import Dataset
    from transformers import TrainerCallback

    eval_steps = [args_cli.sft_steps] if args_cli.job == "sft" else [args_cli.steps]
    run = "h2b-sft" if args_cli.job == "sft" else f"h2b-{args_cli.reward}-s{args_cli.seed}"

    class EvalOnSave(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            if state.global_step in eval_steps:
                evaluate(f"{run}-step{state.global_step}",
                         model.load_lora(f"{OUT}/{run}/checkpoint-{state.global_step}"))

    if args_cli.job == "sft":
        from trl import SFTConfig, SFTTrainer
        ds = Dataset.from_list([{"prompt": chat_prompt(r["prompt"]),
                                 "completion": solver_trace(r["steps"], sorted(r["answer"]))} for r in train_sft])
        cfg = SFTConfig(
            output_dir=f"{OUT}/{run}", seed=0, max_steps=args_cli.sft_steps, save_steps=args_cli.sft_steps,
            logging_steps=10, per_device_train_batch_size=4, gradient_accumulation_steps=1, learning_rate=1e-4,
            optim="adamw_8bit", report_to="none", max_length=1700)
        trainer = SFTTrainer(model=model, processing_class=tokenizer, args=cfg, train_dataset=ds,
                             callbacks=[EvalOnSave()])
    else:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        from trl import GRPOConfig, GRPOTrainer
        init = f"{OUT}/h2b-sft/checkpoint-{args_cli.sft_steps}/adapter_model.safetensors"
        if not os.path.exists(init):
            raise SystemExit(f"Error: SFT warm start missing: {init}")
        res = set_peft_model_state_dict(model, load_file(init))
        print("Loaded SFT warm start; unexpected keys:", len(getattr(res, "unexpected_keys", []) or []), flush=True)
        ds = Dataset.from_list([{"prompt": chat_prompt(r["prompt"]), "answer": r["answer"],
                                 "steps": json.dumps(r["steps"])} for r in train_rl]).shuffle(seed=args_cli.seed)

        def outcome_reward(prompts, completions, answer, **kwargs):
            return [1.0 if parse_answer(c) == sorted(a) else 0.0 for c, a in zip(completions, answer)]

        def process_reward(prompts, completions, answer, steps, **kwargs):
            return [0.5 * (1.0 if parse_answer(c) == sorted(a) else 0.0) + 0.5 * step_score_v2(c, json.loads(st))
                    for c, a, st in zip(completions, answer, steps)]

        cfg = GRPOConfig(
            output_dir=f"{OUT}/{run}", seed=args_cli.seed, max_steps=args_cli.steps, save_steps=60, logging_steps=5,
            per_device_train_batch_size=4, gradient_accumulation_steps=8, num_generations=8,
            max_prompt_length=1300, max_completion_length=400, temperature=1.0,
            learning_rate=2e-5, beta=0.0, optim="adamw_8bit", report_to="none")
        trainer = GRPOTrainer(model=model, processing_class=tokenizer,
                              reward_funcs=[outcome_reward if args_cli.reward == "outcome" else process_reward],
                              args=cfg, train_dataset=ds, callbacks=[EvalOnSave()])
    import glob
    resume = bool(glob.glob(f"{OUT}/{run}/checkpoint-*"))
    if resume:
        print("RESUME from last checkpoint in", f"{OUT}/{run}", flush=True)
    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume)
    print("TRAIN DONE", run, round((time.time() - t0) / 60, 1), "min", flush=True)
