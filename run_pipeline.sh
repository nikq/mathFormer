#!/bin/bash
set -e

# Configuration
MODEL_SIZE="medium"  # tiny for speed, change to small/medium/large as needed
BATCH_SIZE=32
STEPS=200         # Baseline training steps
RL_STEPS=100      # RL training steps
NUM_EXPERTS=4     # MoE configuration
ACTIVE_EXPERTS=2

echo "========================================================"
echo "Starting Baseline Training Pipeline"
echo "Model Size: $MODEL_SIZE"
echo "========================================================"

# 1. Run Baseline Training
echo "[1/2] Running Supervised Baseline Training..."
# Note: output checkpoint path is printed by train.py
# We will use a fixed directory for simplicity or grab the latest
uv run python -m src.train \
    --modelsize $MODEL_SIZE \
    --steps $STEPS \
    --batch_size $BATCH_SIZE \
    --max_digits 3 \
    --max_depth 3 \
    --num_experts $NUM_EXPERTS \
    --active_experts $ACTIVE_EXPERTS \
    --log_csv "baseline_log.csv" \
    --step_eval

# Find the latest checkpoint
LATEST_CKPT=$(ls -t checkpoints/model*.pt | head -n 1)

if [ -z "$LATEST_CKPT" ]; then
    echo "Error: No checkpoint found after baseline training!"
    exit 1
fi

echo "Found baseline checkpoint: $LATEST_CKPT"

# 2. Run RL Training (using the baseline checkpoint)
echo "========================================================"
echo "[2/2] Running RL Training (GRPO) with MoE..."
echo "Checkpoint: $LATEST_CKPT"
echo "========================================================"

uv run python -m src.train_rl \
    --checkpoint "$LATEST_CKPT" \
    --modelsize $MODEL_SIZE \
    --steps $RL_STEPS \
    --batch_size $BATCH_SIZE \
    --group_size 4 \
    --top_k 50 \
    --num_experts $NUM_EXPERTS \
    --active_experts $ACTIVE_EXPERTS \
    --log_csv "rl_log.csv" \
    --save_interval 50 \
    --eval_interval 20

echo "========================================================"
echo "Pipeline Completed Successfully!"
echo "Final RL Model saved in checkpoints/rl_model_final.pt"
echo "========================================================"
