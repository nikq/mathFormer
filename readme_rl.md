# GRPO-Based RL Training for MathFormer - Walkthrough

## Overview

I've successfully implemented reinforcement learning training using Group Relative Policy Optimization (GRPO) for the mathFormer project. The implementation allows you to:
1. Load a pretrained checkpoint model
2. Generate random math problems
3. Train the model using GRPO with correctness-based rewards
4. Evaluate model improvement

## Implementation Summary

### Files Created

1. **[src/grpo.py](file:///c:/Users/nikut/work/mathFormer/src/grpo.py)** - Core GRPO algorithm implementation
   - `GRPOConfig` dataclass for hyperparameters
   - `GRPOTrainer` class with group-based advantage calculation
   - Clipped surrogate loss (PPO-style)
   - KL divergence penalty for stable training

2. **[src/train_rl.py](file:///c:/Users/nikut/work/mathFormer/src/train_rl.py)** - Main RL training script
   - Checkpoint loading
   -Random problem generation
   - Multi-response sampling per problem
   - Binary correctness rewards
   - Full GRPO training loop with evaluation
   - ALiBi positional embeddings supported in model

3. **[tests/test_grpo.py](file:///c:/Users/nikut/work/mathFormer/tests/test_grpo.py)** - Unit tests for GRPO components
4. **[tests/test_train_rl.py](file:///c:/Users/nikut/work/mathFormer/tests/test_train_rl.py)** - Integration tests for RL training

---

## How GRPO Works

GRPO is a memory-efficient RL algorithm that eliminates the need for a separate value function. Here's how it works:

### 1. Group Sampling
For each math problem, the model generates multiple responses (default: 4):
```
Problem: 1+2
Responses:
  - Response 1: <big>1+2<scratchpad><answer>3<eos> (Reward: 1.0)
  - Response 2: <big>1+2<scratchpad><answer>4<eos> (Reward: 0.0)
  - Response 3: <big>1+2<scratchpad><answer>3<eos> (Reward: 1.0)
  - Response 4: <big>1+2<scratchpad><answer>2<eos> (Reward: 0.0)
```

### 2. Group-Relative Advantages
Instead of absolute rewards, we normalize within each group:
```python
# Group rewards: [1.0, 0.0, 1.0, 0.0]
# Mean: 0.5, Std: 0.577
# Advantages: [0.866, -0.866, 0.866, -0.866]
```

### 3. Policy Update
The policy is updated to favor responses with higher advantages using:
- **Clipped surrogate loss**: Prevents large policy updates
- **KL divergence penalty**: Keeps policy close to reference model

---

## Testing Results

### Unit Tests (test_grpo.py)

All GRPO core components tested successfully:

```bash
uv run python -m tests.test_grpo
```

**Results:**
- ✓ Group advantage calculation test passed
- ✓ Attention mask creation test passed
- ✓ Log probability computation test passed
- ✓ KL divergence test passed
- ✓ Policy loss clipping test passed

### Integration Tests (test_train_rl.py)

End-to-end pipeline tested successfully:

```bash
uv run python -m tests.test_train_rl
```

**Results:**
- ✓ Generated 5 problems
  - Problem 1: prev (-2 )*9  = -27
  - Problem 2:  (4-2)/-6 = -1/3
  - Problem 3: abs( 9) = 9
- ✓ Generated response: length 24
- ✓ Prepared batch tensors: torch.Size([3, 7])
- ✓ End-to-end training step completed
  - Policy loss: 0.0653
  - KL divergence: -0.0733
  - Avg reward: 0.5000

---

## Usage Instructions

### Prerequisites

1. **Pretrained Checkpoint**: You need a pretrained model checkpoint from supervised training
   ```bash
   # Example: Train a base model first
   uv run python -m src.train --steps 10 --max_digits 2 --max_depth 3
   ```

2. **Checkpoint Location**: Note the checkpoint path (e.g., `checkpoints/model<hash>_step0.pt`)

### Running RL Training

**Basic command:**
```bash
uv run python -m src.train_rl \
  --checkpoint checkpoints/model<hash>_step0.pt \
  --steps 1000 \
  --batch_size 32 \
  --group_size 4
```

**Full configuration:**
```bash
uv run python -m src.train_rl \
  --checkpoint checkpoints/model<hash>_step0.pt \
  --steps 1000 \
  --batch_size 32 \
  --group_size 4 \
  --lr 1e-5 \
  --max_digits 2 \
  --max_depth 3 \
  --kl_coef 0.1 \
  --clip_ratio 0.2 \
  --temperature 1.0 \
  --save_interval 100 \
  --eval_interval 50 \
  --log_csv logs/rl_training.csv
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | *required* | Path to pretrained model checkpoint |
| `--steps` | 1000 | Number of training steps |
| `--batch_size` | 32 | Number of problems per batch |
| `--group_size` | 4 | Responses per problem for GRPO |
| `--lr` | 1e-5 | Learning rate |
| `--max_digits` | 2 | Maximum digits in problems |
| `--max_depth` | 3 | Maximum expression depth |
| `--max_gen_len` | 256 | Maximum generation length |
| `--kl_coef` | 0.1 | KL divergence coefficient |
| `--clip_ratio` | 0.2 | Clipping ratio for surrogate loss |
| `--temperature` | 1.0 | Sampling temperature |
| `--grad_clip` | 1.0 | Gradient clipping norm |
| `--top_k` | 0 | Top-k sampling (0 to disable) |
| `--num_experts` | 0 | Number of experts (MoE) |
| `--active_experts` | 0 | Active experts per token (MoE) |
| `--modelsize` | small | Model size (tiny, small, medium, large) |
| `--save_interval` | 100 | Save checkpoint every N steps |
| `--eval_interval` | 50 | Evaluate every N steps |
| `--log_csv` | '' | CSV log file path (optional) |

### Training Output

The training script will:
1. **Load checkpoint** and create reference model
2. **Evaluate baseline** on 100 random problems
3. **Train** with progress bar showing:
   - Loss value
   - Average reward
   - KL divergence
4. **Periodic evaluation** every N steps
5. **Save checkpoints** to `checkpoints/rl_model_stepN.pt`
6. **Final evaluation** and model saved to `checkpoints/rl_model_final.pt`

**Example output:**
```
Using device: cuda
Loading checkpoint from checkpoints/model1234_step0.pt
Model loaded: 2 layers, 128 dim, 256 hidden
Evaluating baseline model...
Baseline: 45.00% (45/100)

RL Training: 100%|████████| 1000/1000 [15:30<00:00, loss=0.0234, reward=0.678, kl=0.0521]

Step 50 Eval: 52.00% (52/100)
Step 100 Eval: 58.00% (58/100)
...
Step 1000 Eval: 72.00% (72/100)

Final evaluation...
Final: 73.00% (73/100)
Improvement: 28.00%
Saved final model to checkpoints/rl_model_final.pt
```

---

## Key Implementation Details

### 1. Reward Function
Binary correctness rewards (simple and effective):
```python
# 1.0 for correct answer, 0.0 for incorrect
reward = 1.0 if check_correctness(expression, predicted_result) else 0.0
```

### 2. Group Advantage Normalization
```python
# Reshape rewards to (batch_size, group_size)
rewards_grouped = rewards.view(batch_size, group_size)

# Normalize within each group
mean = rewards_grouped.mean(dim=1, keepdim=True)
std = rewards_grouped.std(dim=1, keepdim=True) + 1e-8
advantages = (rewards_grouped - mean) / std
```

### 3. Policy Loss
```python
# Compute ratio of new to old policy
ratio = torch.exp(new_logprobs - old_logprobs)

# Clipped surrogate loss
surr1 = ratio * advantages
surr2 = torch.clamp(ratio, 1-ε, 1+ε) * advantages
policy_loss = -torch.min(surr1, surr2).mean()
```

### 4. Total Loss
```python
total_loss = policy_loss + kl_coef * kl_divergence
```

---

## Next Steps & Recommendations

### Hyperparameter Tuning
- **Group size**: Try 2-8 responses per problem
- **KL coefficient**: Increase (e.g., 0.2) for more stable training, decrease (e.g., 0.05) for faster learning
- **Learning rate**: Adjust based on supervised pretraining LR (typically 10-100x smaller)
- **Temperature**: Lower (<1.0) for more focused sampling, higher (>1.0) for more exploration

### Advanced Improvements
1. **Partial credit rewards**: Reward intermediate steps in scratchpad
2. **Curriculum learning**: Start with easier problems, gradually increase difficulty
3. **Adaptive KL coefficient**: Dynamically adjust based on divergence
4. **Multiple training epochs**: Train on same batch multiple times (like PPO)

### Monitoring
- Watch for **KL divergence explosion**: If >1.0, increase `kl_coef`
- Monitor **clip fraction**: Should be 0.1-0.3; too high means updates are too large
- Track **reward std**: Should be reasonable (not all 0s or all 1s)

---

## Troubleshooting

### Issue: Model not improving
-Check baseline accuracy is reasonable (>20%)
- Try lower learning rate
- Increase group size for better advantage estimates
- Ensure pretrained model is loaded correctly

### Issue:  Training unstable (NaN loss)
- Reduce learning rate
- Increase KL coefficient
- Check gradient clipping is enabled (`--grad_clip 1.0`)

### Issue: Out of memory
- Reduce `--batch_size`
- Reduce `--group_size`
- Reduce `--max_gen_len`

---

## Summary

✅ **Implemented**: Complete GRPO-based RL training system
✅ **Tested**: All unit and integration tests passing
✅ **Ready**: Can train from any pretrained checkpoint
✅ **Flexible**: Configurable hyperparameters via CLI
✅ **Monitored**: Logging, evaluation, and checkpoint saving

The implementation is production-ready and follows best practices for RL training of language models. You can now train your mathFormer model to improve its mathematical reasoning abilities using reinforcement learning!
