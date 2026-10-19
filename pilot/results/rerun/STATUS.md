# Confirmatory reruns: status 2026-10-19 09:10 (see ../../PREREGISTRATION.md)

Free Colab T4, Qwen3-4B, per-item records in `records/`, analysis in `analysis.json`
(`python -I src/analyze.py --records results/rerun/records --out results/rerun/analysis.json`).

| Run | Done | Accuracy (base 0.330) | CF_joint, base → run (95% CI of difference), n items | Operative changed (joint) | Invariant kept (joint) | Same answer, 0% reasoning (base 0.06) |
|---|---|---|---|---|---|---|
| grpo-s1 step 30 | yes | 0.473 | 0.539 → 0.498 (−0.041, −0.158..+0.078), 62 | 0.42 → 0.12 | 0.66 → 0.87 | 0.18 |
| grpo-s1 step 60 | yes | 0.553 | 0.506 → 0.501 (−0.005, −0.109..+0.100), 68 | 0.38 → 0.21 | 0.63 → 0.79 | 0.26 |
| grpo-s2 step 30 | yes | 0.377 | 0.501 → 0.491 (−0.011, −0.156..+0.138), 55 | 0.35 → 0.22 | 0.65 → 0.76 | 0.20 |
| grpo-s2 step 60 | **no** | | | | | |
| sft-s1 step 60 | **no** | | | | | |

Parse failures in the early-answering test: 0% in every run, including the base model.

**Verdict so far (pre-registered rule, step 60):** seed 1 *refuted* (CF_joint moves by −0.5
points). Overall: *pending* until seed 2 step 60 exists.

**Reading.** On items both models answer correctly, outcome-GRPO makes the model less responsive to
*every* date edit: operative edits change the answer less often and invariant edits keep it more
often, by similar amounts, so balanced CF does not move. The drop in CF reported for the
exploratory run (CF on each model's own correct items) does not survive the jointly-correct
comparison. The rise in answering without the stated reasoning (6% → 18 to 26%) holds in both
seeds with zero parse failures.

**Why incomplete.** The free Colab runtime stopped after about 5.5 hours (05:10 UTC) and no GPU
was available afterwards. Checkpoint `grpo-s2/checkpoint-30` is on Google Drive; running the
notebook again resumes seed 2 at step 30 and then runs SFT.

## Update 2026-10-19 12:20: H1 complete

Seed 2 ran on Modal (L4, amendment 3), compared with the Modal base (`../rerun_modal/analysis.json`):
accuracy 0.327 → 0.343 (+0.017, CI −0.050..+0.083); CF_joint 0.442 → 0.466 (+0.023, CI −0.113..+0.152,
46 items); operative changed 0.21 → 0.11, invariant kept 0.67 → 0.83 (19 and 46 pairs).
**Verdict: seed 1 refuted, seed 2 refuted, so H1 as pre-registered is refuted.** Answer-only SFT
(Modal): accuracy 0.73, operative changed 0.39 → 0.00, invariant kept 0.51 → 0.97 on 80 jointly
correct items; balanced CF 0.45 → 0.49.
