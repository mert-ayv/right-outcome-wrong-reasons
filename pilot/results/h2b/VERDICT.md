# H2b pilot: verdict (PREREGISTRATION_H2B.md)

Modal, one L4 per run, SFT warm start (100 steps on solver-written derivations), then 120 GRPO steps,
seeds 1 and 2, same step-citing prompt for all runs.
Analyses: `analysis_process_vs_outcome_s1.json` and `_s2.json` (primary), `twin_derivations.md`
(co-primary and descriptive), `analysis_vs_base.json`, `analysis_vs_sft.json`, `shortcut_analysis.md`.

| | Untrained | SFT warm start | Outcome, seed 1 | Process, seed 1 | Outcome, seed 2 | Process, seed 2 |
|---|---|---|---|---|---|---|
| Accuracy | 38% | 79% | 86% | 86% | 83% | 84% |
| Twin-consistent accuracy | 18% | 50% | 55% | 54% | 53% | 54% |
| Fully correct derivations | 3% | 69% | 70% | 76% | 71% | 74% |
| Answer = first state named | 42% | 83% | 100% | 100% | 94% | 98% |

**Primary comparison, process vs outcome, CF on jointly correct items:**
- Seed 1: 0.505 → 0.498 (−0.7 points, 95% CI −2.0..0.0), 258 items. Operative changed 0.01 → 0.00
  (95 pairs), invariant kept 1.00 → 1.00. Accuracy +0.0.
- Seed 2: 0.515 → 0.500 (−1.5 points, CI −3.5..0.0), 244 items. Operative changed 0.03 → 0.00
  (86 pairs), invariant kept 1.00 → 1.00. Accuracy +1.0 (CI −1.3..+3.7).
- Twin-consistent accuracy, process minus outcome: −0.3 (seed 1), +1.0 (seed 2).

**Verdict: not promising** (averaged CF difference −1.1 points, below +2).

Descriptive, not pre-registered:
- **All four GRPO runs converged on the shortcut.** Their answers equal "first state named" in 94 to
  100% of test cases, and their accuracy (83 to 86%) is the accuracy of that shortcut (86%). On
  jointly correct items, the answer changes when the deciding date moves in 0 to 3% of operative
  twins, against 22 to 24% for the untrained model.
- **Correct derivations, unfaithful anyway.** The trained models write a derivation that the solver
  accepts as fully correct for 70 to 76% of test cases. On an operative twin, the process-reward
  models cite exactly the same steps as on the original in 100% of cases, including the same day
  counts, although the deciding date moved. Of the operative twins of originals with a fully correct
  derivation (61 to 66 per GRPO run), the twin's derivation is correct in none
  (`twin_derivations.md`). The derivation reproduces a template that the solver accepts on most
  cases. It is not computed from the facts.
- **Why the step reward pays for this.** On the training distribution, a derivation that gives the
  first state all 180 days earns full step credit wherever the first-state shortcut is right (86% of
  cases), so a solver-checked step reward cannot tell the template from the computation. Checking
  steps against the solver is not enough when the training cases do not separate the rule from the
  shortcut. Counterfactual twins in training, or step rewards computed on twins, are the next test.
- The SFT warm start already shows the pattern (operative changed 0.12 on jointly correct items,
  5 of 60 twin derivations correct). GRPO with either reward strengthens it: against the warm
  start, CF on jointly correct items falls by 2 to 4 points, and the process-reward CIs exclude zero.
