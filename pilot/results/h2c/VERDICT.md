# H2c pilot: verdict (PREREGISTRATION_H2C.md)

H2b on a balanced training set (each of the two strongest shortcuts right on 56% of training cases
instead of 80% and 90%). Modal, one L4 per run, SFT warm start (100 steps), then 120 GRPO steps,
seeds 1 and 2. The untrained model is `h2b-base` (copied into `records/`).
Analyses: `analysis_process_vs_outcome_s1.json` and `_s2.json` (Q1), `twin_derivations.md`,
`analysis_vs_base.json`, `analysis_vs_sft.json`, `shortcut_analysis.md`.

| | Untrained | SFT warm start | Outcome, seed 1 | Process, seed 1 | Outcome, seed 2 | Process, seed 2 |
|---|---|---|---|---|---|---|
| Accuracy | 38% | 74% | 74% | 85% | 84% | 83% |
| Twin-consistent accuracy | 18% | 44% | 44% | 54% | 53% | 50% |
| Answer = first state named | 42% | 72% | 74% | 92% | 87% | 85% |
| 42 cases where the first state is wrong: answer right | 3 | 23 | 21 | 11 | 17 | 19 |
| same 42 cases: fully correct derivation | 0 | 15 | 19 | 8 | 3 | 19 |
| H2b run with the same reward and seed: answer right / derivation right | | 13 / 8 | 0 / 0 | 0 / 0 | 5 / 0 | 1 / 0 |

## Q1, reward (primary): process vs outcome

- Seed 1: CF on jointly correct items 0.528 → 0.543 (+1.5 points, 95% CI −1.2..+4.4), 205 items.
  Operative changed 0.08 → 0.10 (80 pairs), invariant kept 0.98 → 0.99. Twin-consistent accuracy
  +10.7 (CI +7.0..+14.7). Accuracy +10.7 (CI +5.3..+16.0).
- Seed 2: 0.541 → 0.525 (−1.6, CI −3.9..+0.4), 245 items. Operative changed 0.12 → 0.09 (93 pairs),
  invariant kept 0.96 → 0.96. Twin-consistent accuracy −2.7 (CI −5.0..−0.3). Accuracy −1.0.

**Verdict: not promising** (averaged CF difference −0.05 points, below +2). The seed-1 gain in
twin-consistent accuracy comes from the outcome run staying at the warm start's accuracy (74%),
not from process following the rule more.

## Q2, data (secondary): H2c vs H2b, twin-consistent accuracy

Outcome s1 −11.0 (CI −15.0..−7.3), outcome s2 −0.3, process s1 0.0, process s2 −4.0 (CI −7.3..−0.7).
Average −3.8. **Verdict: does not help.**

## Descriptive, not pre-registered

- **Balanced data stops GRPO from erasing the computation, but does not make the model follow the
  facts.** On the 42 shortcut-fail cases, H2b's GRPO runs ended with no correct derivation. Three of
  the four H2c runs kept or exceeded the warm start (8 to 19 correct derivations). Yet on operative
  twins the models still cite the same steps as on the original in 88 to 95% of cases, and for
  originals with a fully correct derivation the twin's derivation is correct in 2 to 8 of 55 to 69
  (`twin_derivations.md`).
- **The derivations are not computed from the dates.** Moving the deciding date by 20 to 240 days
  rarely changes the cited day counts, so even correct derivations come from patterns in the case
  rather than from counting days. Neither the reward (H2b) nor the training data (H2c) changes this
  at 4B parameters and 120 steps.
- **Reading fixed in advance.** The capability reading required no run above 10% correct
  derivations on the 42 cases. Two runs reach 45%, so it does not apply as worded. Combined with the
  copied twin derivations, the most likely remaining bottleneck is still the model's date
  arithmetic. A larger model or a date tool can be checked with the warm start alone before any RL.
