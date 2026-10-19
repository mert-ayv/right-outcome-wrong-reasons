"""Remaining pre-registered runs on Modal (amendment 3 in PREREGISTRATION.md).

Runs colab/pilot_t4c.py unchanged on one L4 GPU, one job per container, in the order
base -> sft (seed 1) -> grpo (seed 2). Records go to the Modal volume "h1-rerun" and are committed
after each job, so a failure later loses at most the running job.

  modal run --detach modal_app.py                     # start (keeps running if the laptop sleeps)
  modal volume get h1-rerun h1_rerun/records results/rerun_modal/   # fetch records afterwards
"""
import os
import subprocess

import modal

RAW = "https://raw.githubusercontent.com/mert-ayv/right-outcome-wrong-reasons/main/pilot"
JOBS = ["--job base", "--job sft --seed 1", "--job grpo --seed 2"]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wget", "zip", "build-essential", "git")
    .pip_install("unsloth", "vllm", "vllm-bnb-plugin")
    .run_commands("pip uninstall -y torchaudio || true")
)
volume = modal.Volume.from_name("h1-rerun", create_if_missing=True)
app = modal.App("h1-rerun", image=image)


@app.function(gpu="L4", volumes={"/out": volume}, timeout=6 * 3600)
def run_job(job: str) -> str:
    os.chdir("/root")
    subprocess.run(f"wget -q -N {RAW}/colab/pilot_t4c.py {RAW}/colab/pilot_data.zip", shell=True, check=True)
    env = dict(os.environ, H1_OUT="/out/h1_rerun", UNSLOTH_VLLM_NO_FLASHINFER="1", VLLM_USE_FLASHINFER_SAMPLER="0")
    proc = subprocess.run(f"python -u pilot_t4c.py {job}", shell=True, env=env, capture_output=True, text=True)
    volume.commit()
    keep = [l for l in (proc.stdout + proc.stderr).splitlines()
            if any(k in l for k in ("RESULT", "TRAIN DONE", "SKIP", "RESUME", "Error", "rror:")) and "WARNING" not in l]
    print("\n".join(keep[-20:]), flush=True)
    return f"{job}: exit {proc.returncode}"


@app.function(timeout=12 * 3600)
def pipeline() -> list:
    """Runs the jobs one after another in the cloud, so the sequence survives a sleeping laptop."""
    out = []
    for job in JOBS:
        out.append(run_job.remote(job))
        print(out[-1], flush=True)
    return out


@app.local_entrypoint()
def main():
    print(pipeline.remote())
