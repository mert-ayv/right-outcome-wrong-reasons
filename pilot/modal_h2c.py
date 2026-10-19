"""H2c pilot on Modal (PREREGISTRATION_H2C.md): SFT warm start on the balanced training set, then four
GRPO runs in parallel (outcome and process reward, seeds 1 and 2), each on its own L4. The untrained
model is not re-run: h2b-base is the reference (greedy evaluation, same prompt and test set).

  modal run --detach modal_h2c.py
  modal volume get h1-rerun h2c/records results/h2c/      # fetch records afterwards
"""
import os
import subprocess

import modal

RAW = "https://raw.githubusercontent.com/mert-ayv/right-outcome-wrong-reasons/main/pilot"
FIRST = ["--job sft"]
GRPO = ["--job grpo --reward process --seed 1", "--job grpo --reward outcome --seed 1",
        "--job grpo --reward process --seed 2", "--job grpo --reward outcome --seed 2"]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wget", "zip", "build-essential", "git")
    .pip_install("unsloth", "vllm", "vllm-bnb-plugin")
    .run_commands("pip uninstall -y torchaudio || true")
)
volume = modal.Volume.from_name("h1-rerun", create_if_missing=True)
app = modal.App("h2c-pilot", image=image)


@app.function(gpu="L4", volumes={"/out": volume}, timeout=6 * 3600)
def run_job(job: str) -> str:
    os.chdir("/root")
    subprocess.run(f"wget -q -N {RAW}/colab/pilot_h2b.py {RAW}/colab/pilot_data_h2c.zip {RAW}/src/proof_steps_v2.py",
                   shell=True, check=True)
    env = dict(os.environ, H1_OUT="/out/h2c", UNSLOTH_VLLM_NO_FLASHINFER="1", VLLM_USE_FLASHINFER_SAMPLER="0")
    proc = subprocess.run(f"python -u pilot_h2b.py {job} --name h2c --data pilot_data_h2c.zip",
                          shell=True, env=env, capture_output=True, text=True)
    volume.commit()
    keep = [l for l in (proc.stdout + proc.stderr).splitlines()
            if any(k in l for k in ("RESULT", "TRAIN DONE", "SKIP", "RESUME", "Error", "rror:")) and "WARNING" not in l]
    print("\n".join(keep[-20:]), flush=True)
    return f"{job}: exit {proc.returncode}"


@app.function(timeout=12 * 3600)
def pipeline() -> list:
    """SFT first; the GRPO runs start from its adapter, so they wait for it."""
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
