# Generated from h1_pilot_t4.ipynb; run with: python -u pilot_t4.py
import torch
import os, json, re, zipfile, time
SEED = 1
STEPS = 60
EVAL_STEPS = [30, 60]
OUT = "/content/h1_pilot"
os.makedirs(OUT, exist_ok=True)
with zipfile.ZipFile("pilot_data.zip") as z:
    z.extractall("data")
train = [json.loads(l) for l in open("data/train.jsonl")]
test = [json.loads(l) for l in open("data/test.jsonl")]
print(len(train), len(test))

from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-4B", max_seq_length=1700, load_in_4bit=True,
    fast_inference=True, max_lora_rank=16, gpu_memory_utilization=0.55)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=32, lora_dropout=0.0, random_state=SEED,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth")

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

from vllm import SamplingParams

def generate(prompts, max_tokens, lora=None):
    outs = model.fast_generate(prompts, sampling_params=SamplingParams(temperature=0.0, max_tokens=max_tokens),
                               lora_request=lora)
    return [(o.outputs[0].text, len(o.outputs[0].token_ids)) for o in outs]

def evaluate(items, lora=None, trunc_n=100):
    flat, index = [], []
    for k, r in enumerate(items):
        flat.append(chat_prompt(r["prompt"])); index.append((k, None))
        for j, tw in enumerate(r.get("twins", [])):
            flat.append(chat_prompt(tw["prompt"])); index.append((k, j))
    gens = dict(zip(index, generate(flat, 512, lora)))
    ok_n = 0; tok_r, tok_w = [], []
    op_t = op_c = inv_t = inv_k = 0
    for k, r in enumerate(items):
        text, n = gens[(k, None)]
        pred = parse_answer(text); ok = pred == sorted(r["answer"])
        ok_n += ok; (tok_r if ok else tok_w).append(n)
        if not ok:
            continue
        for j, tw in enumerate(r.get("twins", [])):
            tpred = parse_answer(gens[(k, j)][0])
            if tw["kind"] == "operative":
                op_t += 1; op_c += tpred != pred
            else:
                inv_t += 1; inv_k += tpred == pred
    res = {"acc": ok_n / len(items),
           "cf_operative_changed": op_c / op_t if op_t else float("nan"),
           "cf_invariant_kept": inv_k / inv_t if inv_t else float("nan"), "cf_pairs": [op_t, inv_t],
           "tokens_right": sum(tok_r) / max(1, len(tok_r)), "tokens_wrong": sum(tok_w) / max(1, len(tok_w))}
    res["cf_balanced"] = (res["cf_operative_changed"] + res["cf_invariant_kept"]) / 2
    cand = [k for k in range(len(items)) if parse_answer(gens[(k, None)][0]) is not None][:trunc_n]
    for frac in (0.0, 0.5):
        prompts = []
        for k in cand:
            text = gens[(k, None)][0]
            reasoning = text[:text.upper().rfind("FINAL ANSWER")]
            prompts.append(chat_prompt(items[k]["prompt"]) + reasoning[:int(len(reasoning) * frac)] + "\n\nFINAL ANSWER:")
        short = generate(prompts, 24, lora)
        agree = sum(parse_answer("FINAL ANSWER:" + o) == parse_answer(gens[(k, None)][0]) for (o, _), k in zip(short, cand))
        res[f"trunc_agree_{int(frac * 100)}"] = agree / max(1, len(cand))
    return res

# Base model already evaluated in the previous session on the same 300 items (greedy, same code):
results = {"base": {"acc": 0.3333, "cf_operative_changed": 0.4634, "cf_invariant_kept": 0.62, "cf_pairs": [41, 100],
                    "cf_balanced": 0.5417, "trunc_agree_0": 0.06, "trunc_agree_50": 0.18}}

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

ds = Dataset.from_list([{"prompt": chat_prompt(r["prompt"]), "answer": r["answer"]} for r in train]).shuffle(seed=SEED)

def outcome_reward(prompts, completions, answer, **kwargs):
    return [1.0 if parse_answer(c) == sorted(a) else 0.0 for c, a in zip(completions, answer)]

args = GRPOConfig(
    output_dir=f"{OUT}/grpo-s{SEED}", seed=SEED, max_steps=STEPS, save_steps=30, logging_steps=5,
    per_device_train_batch_size=4, gradient_accumulation_steps=8, num_generations=8,
    max_prompt_length=1200, max_completion_length=400, temperature=1.0,
    learning_rate=2e-5, beta=0.0, optim="adamw_8bit", report_to="none")
from transformers import TrainerCallback

class EvalOnSave(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        step = state.global_step
        if step in EVAL_STEPS:
            path = f"{OUT}/grpo-s{SEED}/checkpoint-{step}"
            t1 = time.time()
            results[f"step{step}"] = evaluate(test, model.load_lora(path))
            print(f"RESULT step{step}", json.dumps(results[f"step{step}"]), round((time.time() - t1) / 60, 1), "min", flush=True)
            json.dump(results, open(f"{OUT}/results.json", "w"), indent=1)

trainer = GRPOTrainer(model=model, processing_class=tokenizer, reward_funcs=[outcome_reward],
                      args=args, train_dataset=ds, callbacks=[EvalOnSave()])
t0 = time.time(); trainer.train(); print("train minutes", round((time.time() - t0) / 60, 1))

base = results["base"]; last = results[f"step{max(s for s in EVAL_STEPS if f'step{s}' in results)}"]
d_acc = 100 * (last["acc"] - base["acc"]); d_cf = 100 * (last["cf_balanced"] - base["cf_balanced"])
print(f"VERDICT accuracy {base['acc']:.3f} -> {last['acc']:.3f} ({d_acc:+.1f} pts); "
      f"CF {base['cf_balanced']:.3f} -> {last['cf_balanced']:.3f} ({d_cf:+.1f} pts); "
      f"trunc0 {base['trunc_agree_0']:.2f} -> {last['trunc_agree_0']:.2f}")
