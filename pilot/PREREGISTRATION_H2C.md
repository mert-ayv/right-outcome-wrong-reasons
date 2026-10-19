# Pre-registration: H2c pilot (H2b on a training set where the shortcuts do not pay)

**Written 2026-10-19, committed before any H2c run started** (see this file's git history).

## Why

In H2b (`PREREGISTRATION_H2B.md`, `results/h2b/VERDICT.md`) both rewards drove the model to the
"first state named" shortcut, and derivations the solver accepted were copied unchanged to the
operative twins. In the training data that shortcut was right on 80% of cases and "state on day 730
before the petition" on 90%, so a template derivation earned full outcome and step reward on most
cases. On the 42 test cases where the first-state shortcut fails, the SFT warm start answered 13
correctly and GRPO brought this down to 0 to 5. H2c tests whether this was the training data or the
reward.

## The only change: the training set

`src/balance_train.py` draws 1,000 training cases from a fresh pool of 8,000 OpenExempt cases
(`src/ae_data.py --seed 7`, no overlap with the earlier training or test cases): 400 where only the
first state is right, 400 where only the day-730 state is right, all 37 where neither is right and
163 where both are right. Each of the two shortcuts is right on 56% of training cases; no other
heuristic in `src/shortcut_analysis.py` is right on more than 43%. All 1,000 labels and solver
derivations were checked with `src/proof_steps_v2.py` (0 mismatches, every derivation scores 1.0).

Everything else is H2b: same 300 test cases and twins (byte-identical `test.jsonl`), same step
instruction and step score, SFT warm start on 400 of the training cases (100 steps), then 120 GRPO
steps on the other 600, seeds 1 and 2, `colab/pilot_h2b.py --name h2c --data pilot_data_h2c.zip`
on Modal L4 (`modal_h2c.py`). The untrained model is not re-run: evaluation is greedy with the same
prompt and test set, so `h2b-base` is the reference.

| Run | Training | Reward |
|---|---|---|
| `h2c-sft` | SFT warm start on the balanced set | (imitation of solver derivations) |
| `h2c-outcome-s1`, `-s2` | warm start + GRPO, 120 steps | 1 if the final answer is exactly right, else 0 |
| `h2c-process-s1`, `-s2` | warm start + GRPO, 120 steps | 0.5 × outcome + 0.5 × step score |

## Questions, metrics and decision

**Q1, reward (primary, H2b rule unchanged):** process vs outcome per seed, CF on jointly correct
items with the paired bootstrap, and twin-consistent accuracy. **Promising** if, averaged over the two
seeds, process minus outcome is at least +5 CF points with both seeds above 0, twin-consistent
accuracy is at least 3 points higher for process, and process accuracy is at most 5 points below
outcome accuracy. **Not promising** if the averaged CF difference is below +2 points. Otherwise
**unclear**.

**Q2, data (secondary):** each H2c GRPO run against its H2b counterpart (same reward and seed).
Balanced data **helps** if twin-consistent accuracy is higher in all four pairs and by at least 5
points on average. **Does not help** if the average difference is below +2 points. Otherwise
**unclear**.

**Descriptive:** accuracy and fully correct derivations on the 42 test cases where the first-state
shortcut fails; agreement with each shortcut; derivations scored on the twins
(`src/twin_derivations.py`); accuracy on all 300 cases. Test accuracy can fall, because the test set
still rewards the shortcut on 86% of cases.

**Reading fixed in advance:** if no H2c run gets a fully correct derivation on more than 10% of the
42 shortcut-fail cases, the bottleneck is the 4B model's date arithmetic rather than the reward or
the data, and the next step is a larger model or a date tool, not another reward.

A promising result is reported as a two-seed pilot result. Records and analyses are added to the
repository whatever the outcome.
