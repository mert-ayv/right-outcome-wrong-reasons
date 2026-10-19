"""H2 pilot on Modal (PREREGISTRATION_H2.md): tagged-prompt base, outcome-GRPO and process-GRPO.

  modal run --detach modal_h2.py
  modal volume get h1-rerun h2/records results/h2/      # fetch records afterwards
"""
import os
import subprocess

import modal

RAW = "https://raw.githubusercontent.com/mert-ayv/right-outcome-wrong-reasons/main/pilot"
JOBS = ["--job base", "--job grpo --reward process --seed 1", "--job grpo --reward outcome --seed 1"]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wget", "zip", "build-essential", "git")
    .pip_install("unsloth", "vllm", "vllm-bnb-plugin")
    .run_commands("pip uninstall -y torchaudio || true")
)
volume = modal.Volume.from_name("h1-rerun", create_if_missing=True)
app = modal.App("h2-pilot", image=image)


@app.function(gpu="L4", volumes={"/out": volume}, timeout=6 * 3600)
def run_job(job: str) -> str:
    os.chdir("/root")
    subprocess.run(f"wget -q -N {RAW}/colab/pilot_h2.py {RAW}/colab/pilot_data.zip {RAW}/src/proof_steps.py",
                   shell=True, check=True)
    env = dict(os.environ, H1_OUT="/out/h2", UNSLOTH_VLLM_NO_FLASHINFER="1", VLLM_USE_FLASHINFER_SAMPLER="0")
    proc = subprocess.run(f"python -u pilot_h2.py {job}", shell=True, env=env, capture_output=True, text=True)
    volume.commit()
    keep = [l for l in (proc.stdout + proc.stderr).splitlines()
            if any(k in l for k in ("RESULT", "TRAIN DONE", "SKIP", "RESUME", "Error", "rror:")) and "WARNING" not in l]
    print("\n".join(keep[-20:]), flush=True)
    return f"{job}: exit {proc.returncode}"


@app.function(timeout=12 * 3600)
def pipeline() -> list:
    out = []
    for job in JOBS:
        out.append(run_job.remote(job))
        print(out[-1], flush=True)
    return out


@app.local_entrypoint()
def main():
    print(pipeline.remote())
