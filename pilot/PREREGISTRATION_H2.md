# Pre-registration: H2 pilot (proof-step rewards)

**Written 2026-10-19, committed before any H2 run started** (see this file's git history). Small,
single-seed pilot to decide whether H2 is promising enough to report in the exposé.

## Question

H2: with everything else equal, does a reward for solver-verified proof steps make the model's
answers follow the legally operative facts more often than an outcome-only reward?

## Proof steps

`src/proof_steps.py` derives the gold steps of OpenExempt Task AE from each case: `[730: <State>]`
or `[730: MULTIPLE]`, then `[180: <State>]` if multiple, then `[OPT-OUT <State>: yes|no]`. The
derivation reproduces the solver's label on all 1,300 cases (0 mismatches, checked before this file
was committed). Every prompt in this pilot (training, test, twins) gets one added instruction to
cite rule applications in this form (`tag_prompt`); nothing else in the prompt changes.

## Runs (Modal, one NVIDIA L4, `modal_h2.py` running `colab/pilot_h2.py`)

| Run | Training | Reward |
|---|---|---|
| `h2-base` | none | (tagged prompt, untrained) |
| `h2-outcome-s1` | GRPO, 60 steps, seed 1 | 1 if the final answer is exactly right, else 0 |
| `h2-process-s1` | GRPO, 60 steps, seed 1 | 0.5 x outcome + 0.5 x F1 of the cited steps against the gold steps |

GRPO settings as in the H1 reruns (LoRA r 16, 4 prompts x 8 samples, temperature 1, learning rate
2e-5, no KL). Step F1 counts only steps that match a gold step, so citing extra or wrong steps lowers
the reward. Checkpoint 60 is evaluated with the H1 evaluation (same 300 test cases and twins) plus
the step F1 of each answer.

## Metrics and decision

- **Primary:** CF on jointly correct items (`src/analyze.py`), comparing `h2-process-s1-step60`
  directly with `h2-outcome-s1-step60` (`--base h2-outcome-s1-step60`), with the paired bootstrap.
- **Secondary:** accuracy; the operative-changed rate on jointly correct items; each run against
  `h2-base`; step F1; early answering with parse failures.
- **Promising** if process minus outcome is +5 CF points or more with a 95% interval above 0, and
  process accuracy is at most 5 points below outcome accuracy.
- **Not promising** if the CF difference is below +2 points. Otherwise **unclear**.

A promising result is reported as a single-seed pilot result; it is not a test of H2 for the thesis.
Whatever the outcome, the records and the analysis are added to the repository.
