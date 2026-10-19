# H2d check: verdict (PREREGISTRATION_H2D.md)

Date tool output in every prompt, balanced training set, SFT warm start only (no RL). Modal L4.
Reference: `h2c-sft-step100`, the same warm start without the tool (copied into `records/`).
Analyses: `twin_derivations.md`, `analysis_vs_base.json` (against the untrained model with the
tool), `shortcut_analysis.md`.

| | Untrained, with tool | Warm start, no tool (H2c) | Warm start, with tool |
|---|---|---|---|
| Accuracy | 44% | 74% | 82% |
| Twin-consistent accuracy | 25% | 44% | **72%** |
| Fully correct derivation: original / operative twin | 5% / 1% | 62% / 20% | 78% / **77%** |
| Operative twin cites the same steps as its original | 43% | 79% | **22%** |
| Operative twins of a fully correct original: derivation correct | | 3 / 60 | **84 / 112** |
| 42 cases where the first state is wrong: answer / derivation right | 1 / 0 | 23 / 15 | **39 / 33** |
| CF on jointly correct items, against the untrained model with tool | | 0.465 → 0.534 | **0.478 → 0.933** |
| Answer = first state named | 52% | 72% | 69% |

**Gate 1 (twins, at least 50%): met, 75% (84 of 112). Gate 2 (hard cases, at least 21 of 42):
met, 33. Verdict: pass.** Calendar arithmetic was the obstacle: with day numbers given, the warm
start's derivations follow the deciding date (CF on jointly correct items +45.5 points, 95% CI
+36.5..+53.6, 101 items), and on operative twins its answer changes in 80% of cases. The H2
comparison with the tool is pre-registered in `PREREGISTRATION_H2E.md`.

The untrained model does not use the tool output by itself (5% correct derivations). What changes
the behaviour is training on derivations that use it.
