# Right Outcome, Wrong Reasons? Outcome vs. proof-step rewards for faithful statutory reasoning

Pilot study by Mert Ayvazoglu (M.Sc. Information Systems, TUM), October 2026.

**Question.** Does training on right answers teach a language model the legal rule behind them, and
do rewards for solver-checked reasoning steps do better?

## Summary

- **Problem:** reasoning models are trained with outcome-only rewards: a right answer earns full
  reward, whatever reasoning led to it. In law, the stated reasons must be the ones that decided.
- **Method:** Qwen3-4B learns a US bankruptcy rule (OpenExempt, the 730-day domicile rule) by outcome
  RL (GRPO) or by copying correct answers (SFT). Solver-checked twins test whether its answers follow
  the facts that decide the case. Metrics and decision rule are pre-registered.
- **Finding:** training on answers raises accuracy without teaching the rule. The SFT model reaches
  73% mostly by naming the first state in the text, and its answer never follows the deciding date.
  Outcome RL gains accuracy in one of two seeds and also reacts less to that date. H1 as
  pre-registered is refuted: its summary metric stays at chance.
- **Contribution:** solver-checked twins on real statutes, a pre-registered evaluation, and
  per-item records for every run.
- **Follow-up pilots:** a reward for solver-checked proof steps never beat the outcome reward (four
  pre-registered pilots). Without help on dates, models wrote derivations the solver accepts and
  copied them unchanged when the deciding date moved. With a date tool, the warm start follows the
  facts, and outcome RL raises accuracy to 88 to 96% without losing faithfulness.
- **Thesis:** find when outcome RL does erode faithfulness (training data where the shortcut pays,
  larger models, richer statutes) and why step rewards of this form hurt.

## How the pilot works

Does a model trained only on right answers learn the legal rule, or learn to guess right? Three
versions of the same model answer the same 300 test cases:

- **Untrained:** Qwen3-4B as released; the baseline for every comparison.
- **Outcome RL (GRPO):** the model writes reasoning and an answer 8 times per case and is rewarded
  for a correct answer only. Current reasoning models are trained this way.
- **Answer-only SFT (control):** the model copies the correct answers. Same information, no RL, so
  it shows whether an effect comes from RL or from training on answers at all.

All three use the same frozen 4-bit Qwen3-4B; training changes a LoRA adapter (under 1% of the weights).

```mermaid
flowchart LR
    D["1,000 training cases<br/>(domicile history<br/>+ question)"]
    subgraph U["Untrained"]
        direction TB
        U1["Qwen3-4B<br/>as released"]
    end
    subgraph R["Outcome RL (GRPO), 60 steps"]
        direction TB
        R1["4 cases per step,<br/>8 own answers each"]
        R2["Reward 1 if the final<br/>answer is right;<br/>reasoning unchecked"]
        R3["Answers above the<br/>group mean become<br/>more likely"]
        R1 --> R2 --> R3
    end
    subgraph S["Answer-only SFT, 60 steps"]
        direction TB
        S1["4 cases per step"]
        S2["Target: 'FINAL ANSWER:<br/>Federal, Oregon',<br/>no reasoning"]
        S3["Imitate<br/>the target"]
        S1 --> S2 --> S3
    end
    D --> U & R & S
    U & R & S --> E["Answer 300 test<br/>cases and their<br/>solver-checked twins"]
    E --> M1["Accuracy"]
    E --> M2["Counterfactual<br/>faithfulness"]
    E --> M3["Early answering:<br/>same answer without<br/>the reasoning?"]
    M1 & M2 & M3 --> H{"H1: accuracy up,<br/>faithfulness down?"}
```

- **Twins:** each test case has copies in which one date moves. If the solver's answer changes
  (operative twin), a model that applies the rule must change its answer; if not (invariant twin),
  it must keep it.
- **Early answering:** the model's own reasoning is cut to 0% or 50%. The same answer anyway means
  the reasoning was not needed.
- **Follow-up pilots (below the table):** the same GRPO loop with a reward that also checks each
  cited rule step against the solver, balanced training data, and a date tool.

## Contents

| Path | What |
|---|---|
| `pilot/` | code, data, pre-registrations and results ([pilot/README.md](pilot/README.md)) |

## Pilot result in one table

> **In one sentence:** the trained models can score higher without learning the rule. Answer-only
> SFT reaches 73% by naming the first state in the text and never reacts to the date that decides
> the case.

| | Untrained | Outcome RL, seed 1 | Outcome RL, seed 2 | Answer-only SFT |
|---|---|---|---|---|
| **Accuracy** | 33% | **55%** | 34% | **73%** |
| Answer = first state named¹ | 32% | 60% | 35% | **84%** |
| **Answer changes when the deciding date moves** (should be 100%)² | | 38% → **21%** | 21% → **11%** | 39% → **0%** |
| Answer stays when an irrelevant date moves (should be 100%)² | | 63% → 79% | 67% → 83% | 51% → 98% |
| Balanced CF, pre-registered (0.5 = chance)² | | 0.51 → 0.50 | 0.44 → 0.47 | 0.45 → 0.49 |
| H1 verdict (pre-registered) | | refuted | refuted | control |
| Same answer without the reasoning | 6% | 26% | 16% | 92%³ |

¹ Picks the first state in the text and ignores the dates; legally right on 86% of the test cases.
² On cases both the untrained and the trained model answer correctly, before → after; 19 to 29
operative pairs each, so descriptive. Seed 1 ran on a Colab T4, the others on a Modal L4, each
against an untrained run on the same machine.
³ SFT was trained to answer without reasoning.

**Findings**
1. **Accuracy hides a shortcut.** SFT reaches 73% by naming the first state in the text, which the
   OpenExempt generator makes right 86% of the time. Only the twins show that its answer never
   follows the deciding date.
2. **Outcome RL is unstable and does not learn the rule either.** One seed reaches 55%, the other
   34%. Both react less to the deciding date afterwards, even the seed whose accuracy barely moved.
3. **H1 as pre-registered is refuted.** Its metric, balanced CF, averages two rates; a model that
   never changes its answer keeps every invariant twin and lands at 0.5. The operative row carries
   the signal.
4. **The cause is training on answers.** SFT shows the effect more strongly than outcome RL.

## Follow-up pilots: proof-step rewards, balanced data, a date tool

Each step was pre-registered before it ran ([`pilot/PREREGISTRATION_H2*.md`](pilot/)); all use
Qwen3-4B, GRPO for 60 to 120 steps, and the same 300 test cases and twins.

| Pilot | What changed | Process vs outcome reward, CF points¹ | What it showed | Verdict |
|---|---|---|---|---|
| [H2](pilot/results/h2/VERDICT.md) | reward also checks the cited rule steps | −1.4 | the model learned the easy steps, not the 180-day computation | not promising |
| [H2b](pilot/results/h2b/VERDICT.md) | finer step reward, SFT warm start, two seeds | −0.7, −1.5 | both rewards end at the first-state shortcut; derivations the solver accepts are copied unchanged to the twins (0 of 61 to 66 correct) | not promising |
| [H2c](pilot/results/h2c/VERDICT.md) | training data where the shortcuts are right on 56%, not 80 to 90% | +1.5, −1.6 | derivations are still copied across twins (88 to 95%): they are not computed from the dates | not promising, data did not help |
| [H2d](pilot/results/h2d/VERDICT.md) | a date tool gives each date as a day number (warm start only) | | derivations follow the facts: twin-consistent accuracy² 44% → 72%, CF 0.93 | gate passed |
| [H2e](pilot/results/h2e/VERDICT.md) | GRPO from the H2d warm start, with the tool | −3.1, −4.0 | outcome RL: accuracy 88 and 96%, twin-consistent 80 and 87%, no loss of faithfulness; the step reward cost 21 to 26 points of accuracy | not promising; outcome RL does not erode |

¹ Process minus outcome on jointly correct items, per seed; +5 needed for "promising".
² Original and all of its twins answered correctly.

**What the follow-ups show**
1. **Derivations can pass the solver's check and still ignore the facts** (H2b): when the shortcut
   is right on most training cases, a step reward pays for a template as much as for the rule.
2. **The obstacle was calendar arithmetic, not the rule** (H2c, H2d): with day numbers given, the
   same warm start follows the deciding date.
3. **From that start, outcome-only RL improved accuracy without eroding faithfulness, and the step
   reward made things worse** (H2e). The remaining errors sit in one legal step: which states fall
   inside the 730-day window.

Records and analyses in [`pilot/results/`](pilot/results/); details in [pilot/README.md](pilot/README.md).

## License and credits

Code: MIT (see `LICENSE`). The pilot data in `pilot/data/`, `pilot/data_h2c/` and `pilot/colab/pilot_data*.zip`
is generated with the [OpenExempt](https://github.com/servantez/OpenExempt) framework (Servantez,
Lawsky, Jain, Linna, Hammond; Findings of ACL 2026), licensed CC BY 4.0, and remains under that
licence.
