#!/bin/bash
# Amendment 1 run (see PREREGISTRATION.md): base evaluation and outcome-GRPO seed 1 on Apple Silicon (MLX),
# per-item records in the same schema as colab/pilot_t4c.py. Run from the pilot directory:
#   nohup caffeinate -i ./run_rerun_mlx.sh > logs/rerun_mlx.log 2>&1 &
set -e
PY=".venv/bin/python -I"
MODEL=mlx-community/Qwen3-4B-4bit
RUN=runs/mlx-grpo-s1
REC=results/rerun_mlx/records
mkdir -p $REC logs
[ -f $REC/base.jsonl ] || $PY src/eval_records.py --model $MODEL --test data/test.jsonl --out $REC/base.jsonl --batch_size 8
[ -d $RUN/step_60 ] || $PY src/grpo.py --model $MODEL --train data/train.jsonl --out $RUN --steps 60 \
    --prompts_per_step 4 --group_size 8 --max_tokens 400 --micro_batch 1 --num_layers 12 \
    --rank 16 --lora_scale 2.0 --lr 2e-5 --save_every 30 --seed 1
for STEP in 30 60; do
  [ -f $REC/mlx-grpo-s1-step$STEP.jsonl ] || $PY src/eval_records.py --model $MODEL --adapter $RUN/step_$STEP \
      --test data/test.jsonl --out $REC/mlx-grpo-s1-step$STEP.jsonl --batch_size 8
done
$PY src/analyze.py --records $REC --out results/rerun_mlx/analysis.json
echo "RERUN MLX DONE"
