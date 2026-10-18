"""Write colab/h1_rerun_t4.ipynb: the pre-registered H1 reruns (../PREREGISTRATION.md) on a free T4.

The notebook fetches pilot_t4c.py and pilot_data.zip from the public repo, runs each job in its
own process and offers the per-item records as a zip download. Run: python colab/make_notebook_rerun.py
"""
import json
from pathlib import Path

RAW = "https://raw.githubusercontent.com/mert-ayv/right-outcome-wrong-reasons/main/pilot/colab"
ENV = "UNSLOTH_VLLM_NO_FLASHINFER=1 VLLM_USE_FLASHINFER_SAMPLER=0"
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n")})


md("""
# H1 pilot: pre-registered reruns (free Colab T4)

Runs the jobs specified in
[PREREGISTRATION.md](https://github.com/mert-ayv/right-outcome-wrong-reasons/blob/main/pilot/PREREGISTRATION.md):
the untrained model, outcome-GRPO with seeds 1 and 2, and answer-only SFT. Every evaluation writes
per-item records (raw outputs, parsed answers, early-answering outputs with parse status) to
`/content/h1_rerun/records/` and refreshes `/content/h1_rerun.zip`.

**Runtime:** Runtime > Change runtime type > T4 GPU. Approximate times on a T4: base 20 min,
each GRPO run about 2.5 h including two evaluations, SFT about 40 min. A free session may not last
for all jobs: run them one at a time and download the zip after each.

Analyse locally: `python -I src/analyze.py --records <unzipped>/records --out analysis.json`.
""")

md("""
## 1. Save everything to Google Drive

Mounts Drive (Google asks once for access) and points `/content/h1_rerun` to
`MyDrive/h1_rerun`, so records, summaries and checkpoints survive a dropped session. After a
disconnect, reconnect and run all cells again: finished jobs and evaluations are skipped and
training resumes from the last checkpoint.
""")

code("""
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/h1_rerun && rm -rf /content/h1_rerun && ln -sfn /content/drive/MyDrive/h1_rerun /content/h1_rerun && ls -la /content/h1_rerun
""")

md("## 2. Setup")

code(f"""
!pip install -q unsloth vllm vllm-bnb-plugin 2>&1 | tail -2
!wget -q -N {RAW}/pilot_t4c.py {RAW}/pilot_data.zip && ls -la pilot_t4c.py pilot_data.zip
!nvidia-smi --query-gpu=name,memory.total --format=csv
""")

md("## 3. Jobs (one cell each; each runs in a fresh process so GPU memory is released)")

code(f"!{ENV} python -u pilot_t4c.py --job base 2>&1 | grep -E 'RESULT|Error|rror:' | grep -v WARNING")
code(f"!{ENV} python -u pilot_t4c.py --job grpo --seed 1 2>&1 | grep -E 'RESULT|TRAIN DONE|Error|rror:' | grep -v WARNING")
code(f"!{ENV} python -u pilot_t4c.py --job grpo --seed 2 2>&1 | grep -E 'RESULT|TRAIN DONE|Error|rror:' | grep -v WARNING")
code(f"!{ENV} python -u pilot_t4c.py --job sft --seed 1 2>&1 | grep -E 'RESULT|TRAIN DONE|Error|rror:' | grep -v WARNING")

md("## 4. Download the records (optional; everything is also on Drive)")

code("""
!cat /content/h1_rerun/summaries.jsonl
from google.colab import files
files.download("/content/h1_rerun.zip")
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 0}
out = Path(__file__).with_name("h1_rerun_t4.ipynb")
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
