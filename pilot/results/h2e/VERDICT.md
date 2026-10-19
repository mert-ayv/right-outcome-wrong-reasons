# H2e pilot: verdict (PREREGISTRATION_H2E.md)

Four GRPO runs (120 steps) from the H2d warm start, with the date tool, balanced training set,
Modal L4. Run tags are `h2d-{reward}-s{seed}-step120` because they share the H2d warm start;
`h2d-base` and `h2d-sft-step100` are copied into `records/`.
Analyses: `analysis_process_vs_outcome_s1.json` and `_s2.json` (Q1), `analysis_vs_sft.json` (Q2),
`analysis_vs_base.json`, `twin_derivations.md`, `shortcut_analysis.md`.

| | Untrained, with tool | Warm start | Outcome, seed 1 | Process, seed 1 | Outcome, seed 2 | Process, seed 2 |
|---|---|---|---|---|---|---|
| Accuracy | 44% | 82% | 88% | 66% | **96%** | 70% |
| Twin-consistent accuracy | 25% | 72% | 80% | 58% | **87%** | 61% |
| Answer changes when the deciding date moves (own correct items) | 14% | 80% | 83% | 78% | 83% | 78% |
| Answer stays when an irrelevant date moves (own correct items) | 86% | 98% | 99% | 99% | 99% | 98% |
| Wrong answers (of 300) | | 54 | 37 | 101 | 13 | 90 |
| Multi-state 730-day window read as one state (of 260) | | 50 | 28 | 96 | 0 | 85 |
| One-state window read as multi-state (of 40) | | 4 | 8 | 1 | 20 | 1 |

## Q1, reward (primary): process vs outcome

- Seed 1: CF on jointly correct items 0.930 → 0.899 (−3.1 points, 95% CI −5.6..−1.1), 192 items.
  Operative changed 0.87 → 0.81 (104 pairs). Twin-consistent accuracy −22 points. Accuracy −21.3
  (CI −26.7..−16.3).
- Seed 2: 0.965 → 0.924 (−4.0, CI −6.6..−1.9), 199 items. Operative changed 0.95 → 0.87 (99 pairs).
  Twin-consistent accuracy −26 points. Accuracy −25.7 (CI −31.7..−20.0).

**Verdict: not promising** (averaged CF difference −3.5 points, below +2). The step reward is worse
than the outcome reward on every pre-registered metric, in both seeds.

## Q2, erosion (secondary): each run against the warm start

- Outcome: CF +1.1 (CI −0.7..+3.1) and +0.7 (CI −1.4..+2.8); accuracy +5.7 and +13.7.
  **Verdict: outcome RL does not erode faithfulness** (average +0.9, above −2).
- Process: CF −2.3 (CI −4.3..−0.7) and −2.3 (CI −4.2..−0.7); accuracy −15.7 and −12.0.
  **Verdict: unclear** under the rule (average −2.3), with both CIs below zero.

## Descriptive, not pre-registered

- **With the tool, the errors sit in one legal step.** In every run, almost all wrong answers come
  from deciding which states fall inside the 730-day window, an interval-overlap reading of
  § 522(b)(3)(A). Counting the 180 days, which the tool supports, is no longer where the models fail.
- **Outcome RL fixed that step, the step reward made it worse.** The warm start reads 50 of 260
  multi-state windows as one state. Outcome RL brings this to 28 and 0. The process runs raise it to
  96 and 85, and then skip the 180-day computation. Their outputs are also shorter (186 and 190
  tokens vs 200). Why the step reward pushes in this direction is not established here; the step
  score does not pay for a wrong DECISIVE step.
- **Outcome seed 2 overcorrects.** It never reads a multi-state window as one state but reads 20 of
  the 40 one-state windows as multi-state. 12 of its 13 errors are among the 42 cases where the first
  state is wrong, and 11 of these answers name the first state. The derivations are consistent with
  that misreading (the 180-day counts follow from it), so this is a wrong reading of the window, not
  the first-state shortcut, although the answer is the same.
- **Outcome seed 1 drops a tag.** In 91 answers it writes "Oregon law applies" instead of
  `[DECISIVE: Oregon]`, so its share of fully correct derivations falls to 54% while its accuracy
  and twin metrics rise. This is format drift under a reward that does not check the steps.
- **What this means for the hypotheses.** Once the model can do the arithmetic and starts from a
  faithful warm start, outcome-only RL improves accuracy and keeps the answers tied to the deciding
  facts (H1 not supported here either). A solver-checked step reward of this form does not add
  faithfulness and costs accuracy (H2 not supported).
