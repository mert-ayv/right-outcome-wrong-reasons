"""Balanced training set for the H2c pilot (PREREGISTRATION_H2C.md).

In the original training data the two strongest shortcuts, "first state named" and "state on day
730 before the petition", are right on 80% and 90% of cases. This draws 1,000 cases from a fresh
OpenExempt pool so that each is right on 56%: 400 where only the first state is right, 400 where only
the day-730 state is right, all cases where neither is right (37 in the seed-7 pool) and the rest
where both are right.

  python -I src/ae_data.py --vendor vendor/OpenExempt --out POOL --train 8000 --test 0 --seed 7
  python -I src/balance_train.py --pool POOL/train.jsonl --out data_h2c/train.jsonl
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shortcut_analysis import shortcuts  # noqa: E402


def cell(r):
    sc = shortcuts(r["case"])
    ok = lambda k: sorted(sc[k]) == sorted(r["answer"])  # noqa: E731
    return ok("first state named"), ok("state on day 730 before the petition")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--per_cell", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cells = {}
    for r in map(json.loads, open(args.pool)):
        cells.setdefault(cell(r), []).append(r)
    rng = random.Random(args.seed)
    for v in cells.values():
        rng.shuffle(v)
    pick = cells[(True, False)][:args.per_cell] + cells[(False, True)][:args.per_cell] + cells.get((False, False), [])
    pick += cells[(True, True)][:args.n - len(pick)]
    rng.shuffle(pick)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in pick:
            f.write(json.dumps(r) + "\n")
    n = len(pick)
    fs = sum(cell(r)[0] for r in pick)
    d730 = sum(cell(r)[1] for r in pick)
    print(f"{n} cases: first state right {fs} ({100 * fs / n:.0f}%), day-730 state right {d730} ({100 * d730 / n:.0f}%)")


if __name__ == "__main__":
    main()
