"""H2e pilot on Modal (PREREGISTRATION_H2E.md): four GRPO runs (outcome and process reward, seeds 1 and 2)
from the H2d warm start, with the date tool, each on its own L4.

  modal run --detach modal_h2e.py
  modal volume get h1-rerun h2d/records results/h2d/      # fetch records afterwards
"""
import os
import subprocess

import modal

RAW = "https://raw.githubusercontent.com/mert-ayv/right-outcome-wrong-reasons/main/pilot"
FIRST = []  # the warm start is h2d-sft from modal_h2d.py
GRPO = ["--job grpo --reward process --seed 1", "--job grpo --reward outcome --seed 1",
        "--job grpo --reward process --seed 2", "--job grpo --reward outcome --seed 2"]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wget", "zip", "build-essential", "git")
    .pip_install("unsloth", "vllm", "vllm-bnb-plugin")
    .run_commands("pip uninstall -y torchaudio || true")
)
volume = modal.Volume.from_name("h1-rerun", create_if_missing=True)
app = modal.App("h2e-pilot", image=image)


@app.function(gpu="L4", volumes={"/out": volume}, timeout=6 * 3600)
def run_job(job: str) -> str:
    os.chdir("/root")
    subprocess.run(f"wget -q -N {RAW}/colab/pilot_h2b.py {RAW}/colab/pilot_data_h2c.zip {RAW}/src/proof_steps_v2.py "
                   f"{RAW}/src/date_tool.py",
                   shell=True, check=True)
    env = dict(os.environ, H1_OUT="/out/h2d", UNSLOTH_VLLM_NO_FLASHINFER="1", VLLM_USE_FLASHINFER_SAMPLER="0")
    proc = subprocess.run(f"python -u pilot_h2b.py {job} --name h2d --data pilot_data_h2c.zip --tool",
                          shell=True, env=env, capture_output=True, text=True)
    volume.commit()
    keep = [l for l in (proc.stdout + proc.stderr).splitlines()
            if any(k in l for k in ("RESULT", "TRAIN DONE", "SKIP", "RESUME", "Error", "rror:")) and "WARNING" not in l]
    print("\n".join(keep[-20:]), flush=True)
    return f"{job}: exit {proc.returncode}"


@app.function(timeout=12 * 3600)
def pipeline() -> list:
    """Four GRPO runs in parallel."""
    first = [run_job.spawn(job) for job in FIRST]
    out = [c.get() for c in first]
    print(out, flush=True)
    calls = [run_job.spawn(job) for job in GRPO]
    for c in calls:
        out.append(c.get())
        print(out[-1], flush=True)
    return out


@app.local_entrypoint()
def main():
    print(pipeline.remote())
