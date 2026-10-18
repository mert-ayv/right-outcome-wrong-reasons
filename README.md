# Right Outcome, Wrong Reasons? Outcome vs. proof-step rewards for faithful statutory reasoning

Research proposal and pilot by Mert Ayvazoglu (M.Sc. Information Systems, TUM), October 2026.

**Question.** Does outcome-only reinforcement learning lower the causal faithfulness of LLM
reasoning on statutes, and do rewards for solver-verified proof steps prevent this at an acceptable
accuracy cost?

## Contents

| Path | What |
|---|---|
| `Expose_RLVR_Faithfulness_Ayvazoglu_v2.pdf`, `expose/expose_v2.tex` | current one-page exposé (v2, generic thesis version) |
| `Expose_CodeX_Stanford_Ayvazoglu.pdf`, `expose/expose.tex` | first version, written for the CodeX Stanford scholarship call |
| `pilot/` | H1 pilot: code, data, results ([pilot/README.md](pilot/README.md)) |

## Pilot result in one table

Qwen3-4B, outcome-reward GRPO, 60 steps, one seed, OpenExempt Task AE, 300 test cases.

| | Base | After 60 steps |
|---|---|---|
| Accuracy | 33.3% | 55.3% |
| Answer changes when the legally decisive date changes | 46% | 25% |
| Same answer with the reasoning removed | 6% | 24% |
| Counterfactual faithfulness (balanced, 0.5 = chance) | 0.54 | 0.50 |

Accuracy rose while the model tracked the operative fact less and relied less on its stated
reasoning. Preliminary: one seed, and the balanced CF drop (4.4 points) is below the pre-registered
5-point threshold. Details and limitations in [pilot/README.md](pilot/README.md).

## License and credits

Code: MIT (see `LICENSE`). The pilot data in `pilot/data/` and `pilot/colab/pilot_data.zip` is
generated with the [OpenExempt](https://github.com/servantez/OpenExempt) framework (Servantez,
Lawsky, Jain, Linna, Hammond; Findings of ACL 2026), licensed CC BY 4.0, and remains under that
licence.
