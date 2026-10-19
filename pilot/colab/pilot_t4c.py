"""H1 pilot reruns on a free Colab T4 (see ../PREREGISTRATION.md). One job per process.

Jobs:
  --job base             evaluate the untrained model
  --job grpo --seed S    outcome-reward GRPO, 60 steps, evaluate checkpoints 30 and 60
  --job sft --seed S     answer-only SFT on 240 examples, evaluate checkpoint 60

Every evaluation writes per-item records (raw outputs, parsed answers, token counts, early-answering
outputs with parse status) to OUT/records/<tag>.jsonl and a summary line to OUT/summaries.jsonl,
then refreshes OUT.zip so results can be downloaded at any point. Metrics that compare two models
(CF on jointly correct items, bootstrap) are computed offline by ../src/analyze.py.

Resumable: a job whose final records exist is skipped, finished evaluations are skipped, and
training resumes from the last checkpoint in OUT/<run>/. Point OUT at Google Drive (the notebook
symlinks /content/h1_rerun to Drive) so a dropped session loses at most one checkpoint interval.

Run with: UNSLOTH_VLLM_NO_FLASHINFER=1 VLLM_USE_FLASHINFER_SAMPLER=0 python -u pilot_t4c.py --job ...
"""
import argparse
import json
import os
import re
import shutil
import time
import zipfile

ap = argparse.ArgumentParser()
ap.add_argument("--job", required=True, choices=["base", "grpo", "sft"])
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--steps", type=int, default=60)
args_cli = ap.parse_args()

OUT = os.environ.get("H1_OUT", "/content/h1_rerun")  # Kaggle: /kaggle/working/h1_rerun
os.makedirs(f"{OUT}/records", exist_ok=True)
_final = "base" if args_cli.job == "base" else f"{args_cli.job}-s{args_cli.seed}-step{args_cli.steps}"
if os.path.exists(f"{OUT}/records/{_final}.jsonl"):
    print(f"SKIP {_final}: records already exist", flush=True)
    raise SystemExit(0)
with zipfile.ZipFile("pilot_data.zip") as z:
    z.extractall("data")
train = [json.loads(l) for l in open("data/train.jsonl")]
test = [json.loads(l) for l in open("data/test.jsonl")]

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
        rec = {"id": r["id"], "gold": r["answer"], "pred": parse_answer(text), "tokens": n, "text": text, "twins": []}
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
            "pairs": [len(op), len(inv)], "minutes": round((time.time() - t0) / 60, 1)}
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
    evaluate("base")
else:
    from datasets import Dataset
    from transformers import TrainerCallback

    eval_steps = [30, 60] if args_cli.job == "grpo" else [60]
    run = f"{args_cli.job}-s{args_cli.seed}"

    class EvalOnSave(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            if state.global_step in eval_steps:
                evaluate(f"{run}-step{state.global_step}",
                         model.load_lora(f"{OUT}/{run}/checkpoint-{state.global_step}"))

    if args_cli.job == "grpo":
        from trl import GRPOConfig, GRPOTrainer
        ds = Dataset.from_list([{"prompt": chat_prompt(r["prompt"]), "answer": r["answer"]} for r in train]
                               ).shuffle(seed=args_cli.seed)

        def outcome_reward(prompts, completions, answer, **kwargs):
            return [1.0 if parse_answer(c) == sorted(a) else 0.0 for c, a in zip(completions, answer)]

        cfg = GRPOConfig(
            output_dir=f"{OUT}/{run}", seed=args_cli.seed, max_steps=args_cli.steps, save_steps=30, logging_steps=5,
            per_device_train_batch_size=4, gradient_accumulation_steps=8, num_generations=8,
            max_prompt_length=1200, max_completion_length=400, temperature=1.0,
            learning_rate=2e-5, beta=0.0, optim="adamw_8bit", report_to="none")
        trainer = GRPOTrainer(model=model, processing_class=tokenizer, reward_funcs=[outcome_reward],
                              args=cfg, train_dataset=ds, callbacks=[EvalOnSave()])
    else:
        from trl import SFTConfig, SFTTrainer
        n = args_cli.steps * 4
        ds = Dataset.from_list([{"prompt": chat_prompt(r["prompt"]),
                                 "completion": "FINAL ANSWER: " + ", ".join(sorted(r["answer"]))}
                                for r in train]).shuffle(seed=args_cli.seed).select(range(n))
        cfg = SFTConfig(
            output_dir=f"{OUT}/{run}", seed=args_cli.seed, max_steps=args_cli.steps, save_steps=30, logging_steps=5,
            per_device_train_batch_size=4, gradient_accumulation_steps=1, learning_rate=2e-5,
            optim="adamw_8bit", report_to="none", max_length=1700)
        trainer = SFTTrainer(model=model, processing_class=tokenizer, args=cfg, train_dataset=ds,
                             callbacks=[EvalOnSave()])
    import glob
    resume = bool(glob.glob(f"{OUT}/{run}/checkpoint-*"))
    if resume:
        print("RESUME from last checkpoint in", f"{OUT}/{run}", flush=True)
    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume)
    print("TRAIN DONE", run, round((time.time() - t0) / 60, 1), "min", flush=True)
