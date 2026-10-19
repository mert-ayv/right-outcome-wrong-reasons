"""Write colab/h1_rerun_kaggle.ipynb: the remaining pre-registered runs (amendment 2) on Kaggle.

Settings needed in the Kaggle editor: Accelerator "GPU T4 x2", Internet on. Start with
"Save Version" > "Save & Run All (Commit)": the run continues without the browser, and everything
written to /kaggle/working is kept as the notebook's output. Run: python colab/make_notebook_kaggle.py
"""
import json
from pathlib import Path

RAW = "https://raw.githubusercontent.com/mert-ayv/right-outcome-wrong-reasons/main/pilot"
ENV = ("CUDA_VISIBLE_DEVICES=0 H1_OUT=/kaggle/working/h1_rerun "
       "UNSLOTH_VLLM_NO_FLASHINFER=1 VLLM_USE_FLASHINFER_SAMPLER=0")
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n")})


md("""
# H1 pilot: remaining pre-registered runs on Kaggle (amendment 2)

Untrained model, outcome-GRPO seed 2, answer-only SFT seed 1, then the pre-registered analysis.
See [PREREGISTRATION.md](https://github.com/mert-ayv/right-outcome-wrong-reasons/blob/main/pilot/PREREGISTRATION.md).
Settings: Accelerator GPU T4 x2 (one GPU is used), Internet on. Outputs: /kaggle/working/h1_rerun/.
""")

code(f"""
!pip install -q unsloth vllm vllm-bnb-plugin 2>&1 | tail -2
!wget -q -N {RAW}/colab/pilot_t4c.py {RAW}/colab/pilot_data.zip {RAW}/src/analyze.py && ls -la
!nvidia-smi --query-gpu=name,memory.total --format=csv
""")

for job in ["--job base", "--job grpo --seed 2", "--job sft --seed 1"]:
    code(f"!{ENV} python -u pilot_t4c.py {job} 2>&1 | grep -E 'RESULT|TRAIN DONE|SKIP|RESUME|Error|rror:' | grep -v WARNING")

code("""
!python analyze.py --records /kaggle/working/h1_rerun/records --out /kaggle/working/h1_rerun/analysis_kaggle.json
!rm -rf /kaggle/working/h1_rerun/grpo-s2 /kaggle/working/h1_rerun/sft-s1
!cd /kaggle/working && zip -qr h1_rerun_kaggle.zip h1_rerun && ls -la /kaggle/working
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}},
      "nbformat": 4, "nbformat_minor": 4}
out = Path(__file__).with_name("h1_rerun_kaggle.ipynb")
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
