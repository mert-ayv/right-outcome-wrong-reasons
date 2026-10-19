# Pre-registration: H1 pilot reruns

**Written 2026-10-19, committed before any of the runs below were started** (see this file's git
history). It covers the confirmatory reruns only. The exploratory run of 2026-10-18 (seed 1,
reported in `README.md`) happened before this file existed; its results are known and are not
evidence under this plan.

## Where the decision rule comes from

The rule was fixed before the exploratory run, in two places:

1. The research-design check of 2026-10-18 (private working notes, not in this repo), pilot plan,
   quoted verbatim:
   > H1 supported if outcome-GRPO raises accuracy by at least 5 points and lowers CF by at least 5
   > points vs base at matched accuracy in both seeds (bootstrap 95% CI excluding 0). Refuted if CF
   > rises or moves less than 2 points
2. Exposé v2 (thesis proposal draft of 2026-10-18; not part of the public repository), which specifies CF "on items that both
   compared models answer correctly" and adds: "A 2 to 5 point drop triggers a third seed."

This plan follows exposé v2's metric definition (jointly correct items) instead of "matched
accuracy", because checkpoints with equal accuracy do not exist in a 60-step run.

## Hypothesis

H1: relative to the base model, outcome-reward GRPO raises accuracy and lowers counterfactual
faithfulness (CF) on OpenExempt Task AE.

## Fixed setup

- Data: `data/train.jsonl`, `data/test.jsonl` as committed (300 test cases, 136 operative and
  300 invariant twins), generated from OpenExempt commit `ec6fa25`.
- Model: `unsloth/Qwen3-4B` (4-bit), non-thinking chat template, greedy evaluation, max 512 tokens.
- GRPO: identical to the exploratory run (`colab/pilot_t4b.py`): LoRA r 16, alpha 32, 4 prompts x 8
  samples per step, temperature 1.0, learning rate 2e-5, no KL term, 60 steps, max 400 completion
  tokens, outcome reward 1/0 on the exact jurisdiction set.
- Checkpoints evaluated: step 30 and step 60. The decision uses **step 60**.

## Runs

| Run | Purpose |
|---|---|
| `base` | untrained model, evaluated once (decoding is greedy; seed-independent) |
| `grpo-s1` | rerun of the exploratory seed with per-item logging |
| `grpo-s2` | second seed |
| `sft-s1` | answer-only SFT (same training pool, 240 examples = 60 steps x 4, target `FINAL ANSWER: <gold>`, LoRA and learning rate as GRPO), evaluated at step 60: the same-information baseline without RL |

## Metrics

**Primary: CF_joint.** On test items that *both* the base model and the compared checkpoint answer
correctly: for each operative twin, 1 if the checkpoint's answer changes, else 0; for each invariant
twin, 1 if it stays the same. CF_joint = mean of the operative rate and the invariant rate (0.5 is
chance). Base CF_joint is recomputed on the same item set for each comparison.

Secondary, reported for every run:
- accuracy on the 300 originals;
- CF_own (each model on its own correctly answered items; the metric of the exploratory run);
- operative and invariant rates separately;
- early-answering agreement at 0% and 50% of the model's own reasoning (first 100 parseable items),
  **with the parse-failure rate**, and agreement recomputed among parseable outputs only;
- mean completion tokens on right and wrong answers.

**Uncertainty:** paired item-level bootstrap (10,000 resamples of test items) for the step-60 minus
base differences in accuracy and CF_joint; 95% percentile intervals.

## Decision rule (per seed, step 60 vs base)

- **Supported:** accuracy +5 points or more, and CF_joint −5 points or more with the bootstrap 95%
  interval of the CF_joint difference excluding 0.
- **Refuted:** CF_joint rises or falls by less than 2 points.
- **Inconclusive:** anything else.

Overall: H1 is supported if both seeds are supported, refuted if both are refuted; otherwise
**pending**, which triggers a third seed. The SFT arm does not enter the H1 verdict; it decides
whether an effect is specific to RL: if `sft-s1` shows a CF_joint drop of similar size (within
2 points of the GRPO mean), the effect is reported as "training on labels", not "RL".

## Deviations from the original pilot plan (design check of 2026-10-18)

| Original plan | Here | Why |
|---|---|---|
| Qwen3-1.7B and 4B | 4B only | 1.7B scored 7.5% (floor) |
| 300 GRPO steps, LoRA r 32 | 60 steps, r 16 | free Colab T4 compute |
| Max 1,536 completion tokens | 512 (eval), 400 (training) | non-thinking mode; mean answer about 180 tokens |
| Tasks AE and EC, 4,000/400 items | Task AE, 1,000/300 | compute; EC labels come from annotations, not a derivation |
| Statute corpus as in OpenExempt | only § 522(b) and the state opt-out provisions | prompt length |
| CF at matched accuracy | CF on jointly correct items (exposé v2) | matched checkpoints do not exist at 60 steps |
| Early answering at 0/25/50/75/100%, mistake insertion on 200 items | 0% and 50% on 100 items, no mistake insertion | evaluation time |
| SARA check, oracle/constant/random mocks | not run | time; listed as open |

## Known limitations

One model size, one domain, a shortened statute excerpt, 60 steps. The base model's CF is close to
chance, so a drop is bounded. CF_joint uses fewer items than CF_own.

## Amendment 1 (2026-10-19, before the run started)

Colab refused a GPU on 2026-10-19 around 00:15 local time because the free usage limit was
reached, so the runs above cannot start tonight. An additional run is added; the plan above is
unchanged and will be run when Colab provides a GPU again.

**`mlx-grpo-s1`: replication on Apple Silicon (MLX)**, run by `run_rerun_mlx.sh`:
base evaluation and outcome-GRPO seed 1 with `src/eval_records.py` and `src/grpo.py`, same data,
prompts, reward, decoding, metrics and analysis (`src/analyze.py`). Differences from the Colab
setup: MLX 4-bit model (`mlx-community/Qwen3-4B-4bit`) instead of bitsandbytes 4-bit; LoRA rank 16
with scale 2.0 (equal to alpha 32 / r 16) on the **top 12 of 36 layers** only (all-layer LoRA does
not fit in 16 GB memory at usable speed); Adam instead of 8-bit AdamW; a different random number
generator. Because the stack differs, this run is reported separately as a replication and does
**not** enter the H1 verdict above. The same decision rule is applied to it and reported.

## Amendment 2 (2026-10-19, before the Kaggle run started)

Status before this amendment: on Colab, `base`, `grpo-s1` (steps 30 and 60) and `grpo-s2` step 30
finished (records in `results/rerun/records/`, summary in `results/rerun/STATUS.md`). The free
Colab runtime stopped during `grpo-s2`, and no Colab GPU was available afterwards on either account.
The `mlx-grpo-s1` run of amendment 1 was stopped before producing results (fan noise) and is dropped.

The remaining runs move to a free Kaggle notebook (T4, one GPU via `CUDA_VISIBLE_DEVICES=0`), with
the same script (`colab/pilot_t4c.py`, output folder set by `H1_OUT`), data, settings, metrics and
decision rule:

- `base` is evaluated again on Kaggle, so that Kaggle runs are compared with a base model evaluated
  on the same machine and software versions.
- `grpo-s2` runs **from scratch** (60 steps). The Colab checkpoint cannot be transferred (upload
  size), so the Colab `grpo-s2` step-30 records are kept as supplementary and do not enter the
  verdict.
- `sft-s1` runs as specified.

Verdict: seed 1 uses the Colab runs (Colab base vs Colab `grpo-s1`); seed 2 uses the Kaggle runs
(Kaggle base vs Kaggle `grpo-s2`). The overall rule is unchanged. The SFT comparison uses Kaggle base.

## Amendment 3 (2026-10-19, before the Modal run started)

The Kaggle run of amendment 2 stayed queued for over an hour (no free T4 in Kaggle's pool), and
no Colab GPU was available. The remaining runs therefore move to Modal (free starter credit):
`modal_app.py` runs `colab/pilot_t4c.py` unchanged on one **NVIDIA L4** (instead of a T4), in the
order `base` → `sft-s1` → `grpo-s2` (SFT before seed 2, so the short run cannot be lost to a
time limit). The L4 supports bfloat16, so Unsloth computes in bf16 instead of fp16; this and the
GPU type are the only intended differences. Comparisons and the verdict follow amendment 2, with
"Kaggle" replaced by "Modal": Modal runs are compared with the Modal base. The queued Kaggle version
is cancelled once the Modal run has started; if Kaggle finishes first instead, its results are
reported as an additional replication and do not change the verdict.

Note (2026-10-19, 10:40): the queued Kaggle version is **not** cancelled (author's decision). If
it runs, its results are reported as an additional replication in either order of completion and
do not change the verdict, which uses the Modal runs as stated in amendment 3.
