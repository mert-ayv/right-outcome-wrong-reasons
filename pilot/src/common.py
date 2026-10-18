"""Shared helpers: prompt formatting, answer parsing, reward."""
import json
import re

JURISDICTIONS = ["Arizona", "Illinois", "Oregon", "Pennsylvania", "Wisconsin", "Federal"]


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def chat_prompt(tokenizer, user_text):
    """Qwen3 chat prompt in non-thinking mode; the visible answer carries the reasoning."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def parse_answer(text):
    """Return the sorted jurisdiction set after the last 'FINAL ANSWER:', or None if absent."""
    idx = text.upper().rfind("FINAL ANSWER")
    if idx < 0:
        return None
    tail = text[idx + len("FINAL ANSWER"):].lstrip(" :*\n").split("\n")[0]
    found = {j for j in JURISDICTIONS if re.search(rf"\b{j}\b", tail, re.IGNORECASE)}
    return sorted(found) if found else None


def reward(text, gold):
    """Outcome reward: 1 for an exact jurisdiction-set match, 0 otherwise."""
    pred = parse_answer(text)
    return 1.0 if pred is not None and pred == sorted(gold) else 0.0
