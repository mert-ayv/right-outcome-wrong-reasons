# Pre-registration: H2e pilot (outcome vs proof-step reward, with the date tool)

**Written 2026-10-19, committed before any H2e run started** (see this file's git history).

## Why

The H2d check passed (`results/h2d/VERDICT.md`): with a date tool, the SFT warm start's derivations
follow the deciding date (CF on jointly correct items 0.93, twin-consistent accuracy 72%). For the
first time in these pilots, RL starts from a model that applies the rule. H2e asks the thesis
questions from there.

## Runs (Modal L4, `modal_h2e.py`, `colab/pilot_h2b.py --name h2d --data pilot_data_h2c.zip --tool`)

All four runs start from the H2d warm start (`h2d-sft`, checkpoint 100) and use the date tool,
the balanced training set (GRPO on the 600 cases not used for SFT) and the H2b settings: 120 GRPO
steps, LoRA r 16, 4 prompts × 8 samples, temperature 1, learning rate 2e-5, no KL.

| Run | Reward |
|---|---|
| `h2d-outcome-s1`, `-s2` | 1 if the final answer is exactly right, else 0 |
| `h2d-process-s1`, `-s2` | 0.5 × outcome + 0.5 × step score (`src/proof_steps_v2.py`) |

Offline check before launch: on this training set, a derivation template built on either shortcut
earns at most 0.56 outcome reward and 0.53 process reward; the warm start answers 82% of test cases
correctly.

## Questions, metrics and decision

**Q1, reward (primary, decision rule of `PREREGISTRATION_H2B.md` unchanged):** process vs outcome
per seed, CF on jointly correct items with the paired bootstrap, and twin-consistent accuracy.
**Promising** if, averaged over the two seeds, process minus outcome is at least +5 CF points with
both seeds above 0, twin-consistent accuracy is at least 3 points higher for process, and process
accuracy is at most 5 points below outcome accuracy. **Not promising** if the averaged CF difference
is below +2 points. Otherwise **unclear**.

**Q2, erosion (secondary):** each GRPO run against the warm start, CF on jointly correct items.
Outcome RL **erodes** faithfulness if CF falls by at least 5 points in both outcome seeds; it **does
not** if the average change is above −2 points. Otherwise **unclear**. Reported the same way for the
process runs.

**Descriptive:** accuracy, twin-consistent accuracy, derivations on the twins
(`src/twin_derivations.py`), the 42 shortcut-fail cases, agreement with the shortcuts, early
answering.

A promising result is reported as a two-seed pilot result. Records and analyses are added to the
repository whatever the outcome.
