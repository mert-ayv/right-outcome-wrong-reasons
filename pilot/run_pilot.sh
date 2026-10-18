#!/bin/bash
# H1 pilot on Apple Silicon: base evaluation, outcome-GRPO (seed 0), checkpoint evaluations.
# Run from the pilot directory: caffeinate -i ./run_pilot.sh > logs/pilot.log 2>&1
set -e
PY=".venv/bin/python -I"
MODEL=mlx-community/Qwen3-4B-4bit
RUN=runs/grpo4b-s0
mkdir -p results logs

[ -f results/base4b.json ] || $PY src/evaluate.py --model $MODEL --test data/test.jsonl --out results/base4b.json --batch_size 16

$PY src/grpo.py --model $MODEL --train data/train.jsonl --out $RUN --steps 150 \
    --prompts_per_step 4 --group_size 8 --max_tokens 400 --micro_batch 1 --num_layers 12 \
    --save_every 25 --seed 0

for STEP in 75 150; do
  $PY src/evaluate.py --model $MODEL --adapter $RUN/step_$STEP --test data/test.jsonl \
      --out results/grpo4b-s0-step$STEP.json --batch_size 16
done
echo "PILOT DONE"
