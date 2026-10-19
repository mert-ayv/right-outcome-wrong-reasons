"""Outcome-reward GRPO with LoRA in MLX (arm "outcome-GRPO" of the H1 pilot).

One on-policy update per batch, so the importance ratio is 1 and the objective reduces to the
group-normalised policy gradient; no KL term. Reward: 1 if the final jurisdiction set matches the
gold label, else 0. The loss is averaged over all completion tokens in the batch (DAPO-style
token-level normalisation). Adapters are saved every --save_every steps in mlx_lm format, so
evaluate.py can load them with --adapter.

Usage: python -I src/grpo.py --model mlx-community/Qwen3-1.7B-4bit --train data/train.jsonl \
           --out runs/grpo-s0 --steps 150 --seed 0
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.generate import batch_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.utils import linear_to_lora_layers

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import chat_prompt, load_jsonl, reward  # noqa: E402


def save_adapter(model, out_dir, lora_cfg, num_layers):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out_dir / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters())))
    with open(out_dir / "adapter_config.json", "w") as f:
        json.dump({"fine_tune_type": "lora", "num_layers": num_layers, "lora_parameters": lora_cfg}, f)


def sequence_logprobs(model, tokens, prompt_lens, lengths, start):
    """Sum of log-probs of completion tokens and their count, per padded sequence.

    The vocabulary projection runs only from position `start` (shortest prompt in the chunk)
    onward, which avoids materialising logits for the long shared prompt."""
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    hidden = model.model(inputs)[:, start:]
    if model.args.tie_word_embeddings:
        logits = model.model.embed_tokens.as_linear(hidden)
    else:
        logits = model.lm_head(hidden)
    targets = targets[:, start:]
    logp = -nn.losses.cross_entropy(logits.astype(mx.float32), targets, reduction="none")
    pos = mx.arange(start, start + targets.shape[1])[None, :]
    mask = (pos >= (prompt_lens[:, None] - 1)) & (pos < (lengths[:, None] - 1))
    return (logp * mask).sum(axis=1), mask.sum(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--prompts_per_step", type=int, default=4)
    ap.add_argument("--group_size", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=400)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lora_scale", type=float, default=20.0, help="2.0 matches PEFT alpha 32 / r 16")
    ap.add_argument("--num_layers", type=int, default=-1)
    ap.add_argument("--micro_batch", type=int, default=4)
    ap.add_argument("--save_every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    mx.random.seed(args.seed)
    mx.set_cache_limit(1 << 30)  # cap MLX's buffer cache at 1 GB to keep the peak low on 16 GB machines
    model, tokenizer = load(args.model)
    model.freeze()
    lora_cfg = {"rank": args.rank, "scale": args.lora_scale, "dropout": 0.0}
    num_layers = args.num_layers if args.num_layers > 0 else len(model.layers)
    linear_to_lora_layers(model, num_layers, lora_cfg)
    n_train = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"trainable parameters: {n_train / 1e6:.2f}M", flush=True)

    optimizer = optim.Adam(learning_rate=args.lr)
    data = load_jsonl(args.train)
    order = list(range(len(data)))
    random.shuffle(order)
    sampler = make_sampler(temp=args.temp)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def loss_fn(model, tokens, prompt_lens, lengths, adv, total_tokens, start):
        lp, _ = sequence_logprobs(model, tokens, prompt_lens, lengths, start)
        return -(adv * lp).sum() / total_tokens

    grad_fn = nn.value_and_grad(model, loss_fn)
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "log.jsonl"), "a")
    save_adapter(model, os.path.join(args.out, "step_0"), lora_cfg, num_layers)
    cursor = 0
    for step in range(1, args.steps + 1):
        t0 = time.time()
        batch = []
        for _ in range(args.prompts_per_step):
            batch.append(data[order[cursor % len(order)]])
            cursor += 1
        prompt_tokens = [tokenizer.encode(chat_prompt(tokenizer, r["prompt"]), add_special_tokens=False) for r in batch]
        rollout_prompts = [p for p in prompt_tokens for _ in range(args.group_size)]
        res = batch_generate(model, tokenizer, rollout_prompts, max_tokens=args.max_tokens, sampler=sampler,
                             completion_batch_size=min(8, len(rollout_prompts)), prefill_batch_size=1, prefill_step_size=512,
                             return_token_ids=True)
        mx.clear_cache()
        t_gen = time.time() - t0

        rewards = [reward(text, batch[i // args.group_size]["answer"]) for i, text in enumerate(res.texts)]
        advs, informative = [], 0
        for g in range(args.prompts_per_step):
            rs = rewards[g * args.group_size:(g + 1) * args.group_size]
            mean = sum(rs) / len(rs)
            std = (sum((r - mean) ** 2 for r in rs) / len(rs)) ** 0.5
            informative += std > 0
            advs.extend([(r - mean) / (std + 1e-4) if std > 0 else 0.0 for r in rs])

        seqs = [(rollout_prompts[i] + res.token_ids[i], len(rollout_prompts[i]), advs[i])
                for i in range(len(advs)) if advs[i] != 0.0]
        loss_val = 0.0
        if seqs:
            total_tokens = float(sum(len(s) - pl for s, pl, _ in seqs))
            accum = None
            for m in range(0, len(seqs), args.micro_batch):
                chunk = seqs[m:m + args.micro_batch]
                maxlen = max(len(s) for s, _, _ in chunk)
                tokens = mx.array([s + [pad] * (maxlen - len(s)) for s, _, _ in chunk])
                prompt_lens = mx.array([pl for _, pl, _ in chunk])
                lengths = mx.array([len(s) for s, _, _ in chunk])
                adv = mx.array([a for _, _, a in chunk], dtype=mx.float32)
                start = min(pl for _, pl, _ in chunk) - 1
                loss, grads = grad_fn(model, tokens, prompt_lens, lengths, adv, total_tokens, start)
                mx.eval(loss, grads)
                loss_val += loss.item()
                accum = grads if accum is None else tree_map_add(accum, grads)
                mx.clear_cache()
            optimizer.update(model, accum)
            mx.eval(model.trainable_parameters(), optimizer.state)
            mx.clear_cache()

        entry = {"step": step, "reward": sum(rewards) / len(rewards), "informative_groups": informative,
                 "loss": loss_val, "mean_completion_tokens": sum(len(t) for t in res.token_ids) / len(res.token_ids),
                 "sec_gen": round(t_gen, 1), "sec_total": round(time.time() - t0, 1)}
        log.write(json.dumps(entry) + "\n"); log.flush()
        print(json.dumps(entry), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            save_adapter(model, os.path.join(args.out, f"step_{step}"), lora_cfg, num_layers)


def tree_map_add(a, b):
    from mlx.utils import tree_map
    return tree_map(lambda x, y: x + y, a, b)


if __name__ == "__main__":
    main()
