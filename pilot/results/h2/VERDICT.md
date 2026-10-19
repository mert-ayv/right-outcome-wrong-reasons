# H2 pilot: verdict (PREREGISTRATION_H2.md)

Modal, one L4 per run, seed 1, 60 GRPO steps, same tagged prompt for all runs.
Analyses: `analysis_process_vs_outcome.json` (primary), `analysis_vs_base.json`, `shortcut_analysis.md`.

| | Tagged base | Outcome GRPO | Process GRPO |
|---|---|---|---|
| Accuracy | 38.7% | 55.7% | 45.3% |
| Step F1 (cited vs gold steps) | 0.33 | 0.33 | 0.48 |
| All steps right | 12.7% | 10.7% | 15.0% |
| Answers equal "first state named" | 42% | 59% | 48% |

**Primary comparison, process vs outcome, CF on 101 jointly correct items:** 0.542 → 0.527
(−0.014, 95% CI −0.123..+0.097); operative changed 0.21 → 0.27 (33 pairs), invariant kept
0.87 → 0.78. Accuracy −0.103 (CI −0.167..−0.037).

**Verdict: not promising** (CF difference below +2 points; accuracy loss above the 5-point limit).

Descriptive, not pre-registered:
- The process reward raised step F1 but barely the share of fully correct derivations: the model
  earns partial credit on the easy steps (730-day window "MULTIPLE", opt-out) and does not learn
  the 180-day computation that decides most cases.
- Against the tagged base, both trained models react more to the deciding date (CF on jointly
  correct items: outcome 0.458 → 0.560, CI −0.010..+0.204; process 0.410 → 0.541, CI +0.030..+0.227).
  In the H1 runs without the step instruction, the operative rate fell. Asking for explicit steps
  may counter the stickiness; this needs its own pre-registered test.
