# Pre-registration: H2d check (does a date tool make derivations follow the facts?)

**Written 2026-10-19, committed before any H2d run started** (see this file's git history).

## Why

In H2b and H2c the trained models cite the same day counts on an operative twin as on the original
in 88 to 100% of cases, although the deciding date moved by 20 to 240 days (`results/h2c/VERDICT.md`).
Their derivations are not computed from the dates. H2d tests whether the obstacle is calendar
arithmetic, which a lawyer would also hand to a calculator, rather than the legal rule.

## Setup

`src/date_tool.py` adds a date tool's output to every prompt (training, test and twins): each move
and the filing as a day number relative to the filing date. The model still has to take the 730-day
and 180-day windows from the statute, count the days per state in them and apply the rule. The SFT
targets are the solver's derivations written in these day numbers (`tool_trace`, same step tags and
step score as H2b). Checked before this file was committed: on all 1,736 training, test and twin
cases the derivation reproduces the solver's label and scores 1.0.

Everything else is the H2c warm start: balanced training set (`colab/pilot_data_h2c.zip`), SFT on
400 cases for 100 steps, same 300 test cases and twins. This is a check before any RL, so only two
runs (Modal L4, `modal_h2d.py`, `colab/pilot_h2b.py --name h2d --data pilot_data_h2c.zip --tool`):

| Run | Training |
|---|---|
| `h2d-base` | none (untrained model with the tool output) |
| `h2d-sft` | SFT warm start on tool derivations |

## Gate

Reference: `h2c-sft`, the same warm start without the tool (`results/h2c/twin_derivations.md`).

1. **Twins:** of the operative twins of originals with a fully correct derivation, the twin's
   derivation is fully correct in at least 50% (h2c-sft: 3 of 60).
2. **Hard cases:** fully correct derivations on at least 21 of the 42 test cases where the
   first-state shortcut is wrong (h2c-sft: 15).

**Pass** (both met): calendar arithmetic was the obstacle. The H2 comparison (outcome vs process
reward, two seeds, decision rule of `PREREGISTRATION_H2B.md`) is then pre-registered and run with the
tool. **Fail** (either missed): the tool alone does not make the derivations follow the facts, and the
next suspect is the model's use of the rule itself.

Descriptive: accuracy, twin-consistent accuracy, agreement with the shortcuts, CF against `h2d-base`.
Records and analyses are added to the repository whatever the outcome.
