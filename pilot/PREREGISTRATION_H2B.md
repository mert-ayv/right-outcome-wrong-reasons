# Pre-registration: H2b pilot (finer, gated proof-step reward after an SFT warm start)

**Written 2026-10-19, committed before any H2b run started** (see this file's git history).

## Why a second H2 pilot

The H2 pilot (`PREREGISTRATION_H2.md`, `results/h2/VERDICT.md`) was not promising: the model learned to
cite the easy steps for partial credit, not the 180-day computation that decides most cases. H2b
changes the setup in the five points identified afterwards; the question is the same.

| H2 | H2b |
|---|---|
| The deciding computation was one tag, `[180: State]` | The model also cites the days per state in the 180-day window, `[180-DAYS: Oregon 120, Arizona 60]`, checked by the solver with a tolerance of ±2 days |
| Easy steps earned credit on their own | Steps: `[730: ...]`, `[180-DAYS: ...]`, `[DECISIVE: State]`, `[OPT-OUT State: yes/no]`. Step score = 0.5 × DECISIVE correct + 0.3 × share of correct day counts + 0.2 × easy steps, and the easy steps count only if DECISIVE is correct (single-state cases: 0.8 × DECISIVE + 0.2 × easy) |
| RL from the untrained model | Both arms start from the same SFT warm start: 100 steps on solver-written derivations for 400 training cases |
| 60 steps, 1 seed | 120 GRPO steps on the other 600 training cases, seeds 1 and 2 per arm |

`src/proof_steps_v2.py` reproduces the solver's label on all 1,300 cases, and every solver-written
derivation scores 1.0 (both checked before this file was committed).

## Runs (Modal, one NVIDIA L4 each, `modal_h2b.py` running `colab/pilot_h2b.py`)

| Run | Training | Reward |
|---|---|---|
| `h2b-base` | none | |
| `h2b-sft` | SFT warm start, 100 steps, learning rate 1e-4 | (imitation of solver derivations) |
| `h2b-outcome-s1`, `-s2` | warm start + GRPO, 120 steps | 1 if the final answer is exactly right, else 0 |
| `h2b-process-s1`, `-s2` | warm start + GRPO, 120 steps | 0.5 × outcome + 0.5 × step score |

GRPO settings otherwise as in H2 (LoRA r 16, 4 prompts × 8 samples, temperature 1, learning rate
2e-5, no KL). All prompts, training and test, carry the same step instruction.

## Metrics and decision

- **Primary:** CF on jointly correct items (`src/analyze.py`), process vs outcome, computed per seed
  with the same seed's outcome run as reference, with the paired bootstrap.
- **Co-primary, strict:** *twin-consistent accuracy*, the share of the 300 test cases where the
  original and all of its twins are answered correctly. A rule-following model scores high; a
  shortcut model fails the operative twins.
- **Secondary:** accuracy; operative-changed and invariant-kept rates; step score and share of fully
  correct derivations; every run against `h2b-base` and against `h2b-sft`; early answering.
- **Promising** if, averaged over the two seeds, process minus outcome is at least +5 CF points with
  both seeds above 0, and twin-consistent accuracy is at least 3 points higher for process, and
  process accuracy is at most 5 points below outcome accuracy.
- **Not promising** if the averaged CF difference is below +2 points. Otherwise **unclear**.

A promising result is reported as a two-seed pilot result. Whatever the outcome, records and
analyses are added to the repository.
