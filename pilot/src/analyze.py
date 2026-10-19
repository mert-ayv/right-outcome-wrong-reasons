"""Compute the pre-registered metrics (../PREREGISTRATION.md) from per-item records.

Reads <records_dir>/base.jsonl and every other <tag>.jsonl written by colab/pilot_t4c.py, compares
each run with the base model on jointly correct items, runs a paired item-level bootstrap and
applies the decision rule to the step-60 checkpoints.

Usage: python -I src/analyze.py --records results/rerun/records --out results/rerun/analysis.json
"""
import argparse
import glob
import json
import os
import random


def load(path):
    with open(path) as f:
        return {r["id"]: r for r in map(json.loads, f)}


def correct(rec):
    return rec["pred"] is not None and rec["pred"] == sorted(rec["gold"])


def cf_rates(model, ids):
    """Operative-changed and invariant-kept rates of `model` over the given item ids."""
    op, inv = [], []
    for i in ids:
        rec = model[i]
        for tw in rec["twins"]:
            if tw["kind"] == "operative":
                op.append(tw["pred"] != rec["pred"])
            else:
                inv.append(tw["pred"] == rec["pred"])
    op_rate = sum(op) / len(op) if op else float("nan")
    inv_rate = sum(inv) / len(inv) if inv else float("nan")
    return op_rate, inv_rate, (op_rate + inv_rate) / 2, (len(op), len(inv))


def trunc_stats(model, frac):
    rows = [(rec[f"trunc_{frac}"], rec["pred"]) for rec in model.values() if f"trunc_{frac}" in rec]
    if not rows:
        return {}
    parsed = [(t, p) for t, p in rows if t["pred"] is not None]
    return {f"trunc{frac}_n": len(rows),
            f"trunc{frac}_agree": sum(t["pred"] == p for t, p in rows) / len(rows),
            f"trunc{frac}_parse_fail": 1 - len(parsed) / len(rows),
            f"trunc{frac}_agree_parsed": sum(t["pred"] == p for t, p in parsed) / len(parsed) if parsed else float("nan")}


def compare(base, model, n_boot=10000, seed=0):
    ids = sorted(base)
    joint = [i for i in ids if correct(base[i]) and correct(model[i])]
    b_op, b_inv, b_cf, b_pairs = cf_rates(base, joint)
    m_op, m_inv, m_cf, m_pairs = cf_rates(model, joint)
    own_m = [i for i in ids if correct(model[i])]
    own_b = [i for i in ids if correct(base[i])]
    acc_b = len(own_b) / len(ids)
    acc_m = len(own_m) / len(ids)
    res = {
        "acc_base": acc_b, "acc": acc_m, "d_acc": acc_m - acc_b,
        "joint_items": len(joint), "joint_pairs": m_pairs,
        "cf_joint_base": b_cf, "cf_joint": m_cf, "d_cf_joint": m_cf - b_cf,
        "joint_operative_base": b_op, "joint_operative": m_op,
        "joint_invariant_base": b_inv, "joint_invariant": m_inv,
        "cf_own": cf_rates(model, own_m)[2], "cf_own_base": cf_rates(base, own_b)[2],
        "own_pairs": cf_rates(model, own_m)[3],
        "tokens_right": _mean(model[i]["tokens"] for i in own_m),
        "tokens_wrong": _mean(model[i]["tokens"] for i in ids if not correct(model[i])),
        **trunc_stats(model, 0), **trunc_stats(model, 50),
    }
    rng = random.Random(seed)
    d_acc, d_cf = [], []
    for _ in range(n_boot):
        sample = [rng.choice(ids) for _ in ids]
        d_acc.append(sum(correct(model[i]) for i in sample) / len(sample)
                     - sum(correct(base[i]) for i in sample) / len(sample))
        js = [i for i in sample if correct(base[i]) and correct(model[i])]
        mb, mm = cf_rates(base, js)[2], cf_rates(model, js)[2]
        if mb == mb and mm == mm:  # skip resamples without both pair kinds
            d_cf.append(mm - mb)
    res["d_acc_ci95"] = _ci(d_acc)
    res["d_cf_joint_ci95"] = _ci(d_cf)
    return res


def verdict(r):
    """Pre-registered rule, applied to one step-60 comparison."""
    lo, hi = r["d_cf_joint_ci95"]
    if r["d_acc"] >= 0.05 and r["d_cf_joint"] <= -0.05 and hi < 0:
        return "supported"
    if r["d_cf_joint"] > -0.02:
        return "refuted"
    return "inconclusive"


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def _ci(xs):
    xs = sorted(xs)
    return [xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1]] if xs else [float("nan")] * 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--base", default="base", help="tag of the reference run, e.g. h2-base")
    ap.add_argument("--only", default="", help="compare only runs whose tag starts with this prefix")
    args = ap.parse_args()
    base = load(os.path.join(args.records, f"{args.base}.jsonl"))
    out = {"base": {"acc": sum(map(correct, base.values())) / len(base), **trunc_stats(base, 0), **trunc_stats(base, 50)},
           "runs": {}, "verdicts": {}}
    for path in sorted(glob.glob(os.path.join(args.records, "*.jsonl"))):
        tag = os.path.basename(path)[:-6]
        if tag == args.base or not tag.startswith(args.only):
            continue
        out["runs"][tag] = compare(base, load(path), args.n_boot)
        if tag.startswith("grpo") and tag.endswith("step60"):
            out["verdicts"][tag] = verdict(out["runs"][tag])
    v = list(out["verdicts"].values())
    if v:
        out["h1_overall"] = ("supported" if len(v) >= 2 and all(x == "supported" for x in v) else
                             "refuted" if len(v) >= 2 and all(x == "refuted" for x in v) else "pending")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    for tag, r in out["runs"].items():
        print(f"{tag:18s} acc {r['acc']:.3f} ({r['d_acc']:+.3f}, CI {r['d_acc_ci95'][0]:+.3f}..{r['d_acc_ci95'][1]:+.3f})  "
              f"CF_joint {r['cf_joint_base']:.3f}->{r['cf_joint']:.3f} ({r['d_cf_joint']:+.3f}, CI "
              f"{r['d_cf_joint_ci95'][0]:+.3f}..{r['d_cf_joint_ci95'][1]:+.3f}, n={r['joint_items']})  "
              f"trunc0 {r.get('trunc0_agree', float('nan')):.2f} (parse fail {r.get('trunc0_parse_fail', float('nan')):.2f})")
    print("verdicts:", out["verdicts"], "overall:", out.get("h1_overall"))


if __name__ == "__main__":
    main()
