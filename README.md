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

## How the pilot works

All arms train only a small LoRA adapter (rank 16, under 1% of the parameters) on a frozen 4-bit
Qwen3-4B. The arms differ in the training signal.

```mermaid
flowchart LR
    D[("1,000 training cases<br/>(domicile history + question)")]

    subgraph GRPO["Outcome RL (GRPO), 60 steps"]
        direction TB
        G1["Draw 4 cases"] --> G2["Model answers each case 8 times<br/>(reasoning + FINAL ANSWER, temperature 1)"]
        G2 --> G3["Reward per answer:<br/>1 if the jurisdiction set is exact, else 0<br/><b>reasoning is not checked</b>"]
        G3 --> G4["Compare within each group of 8:<br/>above group mean = positive advantage"]
        G4 --> G5["Update the LoRA adapter:<br/>make the better answers more likely"]
        G5 -->|next step| G1
    end

    subgraph SFT["Answer-only SFT, 60 steps"]
        direction TB
        S1["Draw 4 cases"] --> S2["Target text: 'FINAL ANSWER: Federal, Oregon'<br/>(no reasoning)"]
        S2 --> S3["Update the LoRA adapter:<br/>imitate the target"]
        S3 -->|next step| S1
    end

    D --> GRPO
    D --> SFT
    GRPO --> C[("Checkpoints<br/>step 30, step 60")]
    SFT --> C
    C --> E["Evaluation on 300 test cases<br/>(see next figure)"]
```

The thesis variant (H2, not part of this pilot) changes only the reward box of the GRPO loop: it
credits each cited rule application that the solver re-executes and that belongs to the gold proof
tree, instead of the final answer alone.

```mermaid
flowchart TB
    T["Test case<br/>(e.g. Oregon 2018, Arizona 03/2023, Pennsylvania 10/2024)"]
    T --> O["Original: model answers"]
    T --> OP["Operative twin:<br/>one date shifted so the solver's answer changes"]
    T --> IN["Invariant twin:<br/>one date shifted, solver's answer stays"]
    O --> A["Accuracy"]
    O & OP --> CF1["Does the answer change?<br/>(should: yes)"]
    O & IN --> CF2["Does the answer stay?<br/>(should: yes)"]
    CF1 & CF2 --> CF["Counterfactual faithfulness (CF)<br/>mean of both rates, 0.5 = chance"]
    O --> TR["Early answering: cut the model's own reasoning<br/>to 0% / 50%, ask for the answer again"]
    TR --> TA["Same answer without the reasoning?<br/>(high = reasoning not needed)"]
```

## Pilot result in one table

Qwen3-4B, outcome-reward GRPO, 60 steps, one seed, OpenExempt Task AE, 300 test cases.

| | Base | After 60 steps |
|---|---|---|
| Accuracy | 33.3% | 55.3% |
| Answer changes when the legally decisive date changes | 46% | 25% |
| Same answer with the reasoning removed | 6% | 24% |
| Counterfactual faithfulness (balanced, 0.5 = chance) | 0.54 | 0.50 |

Exploratory run, one seed. Accuracy rose while the model tracked the operative fact less and
relied less on its stated reasoning. The CF figures are computed on each model's own correct items,
not on jointly correct items as planned, and the 4.4-point drop falls in the range where the rule
written before the run asks for another seed: **H1 is pending.** Confirmatory reruns are specified
in [pilot/PREREGISTRATION.md](pilot/PREREGISTRATION.md). Details in [pilot/README.md](pilot/README.md).

## License and credits

Code: MIT (see `LICENSE`). The pilot data in `pilot/data/` and `pilot/colab/pilot_data.zip` is
generated with the [OpenExempt](https://github.com/servantez/OpenExempt) framework (Servantez,
Lawsky, Jain, Linna, Hammond; Findings of ACL 2026), licensed CC BY 4.0, and remains under that
licence.
