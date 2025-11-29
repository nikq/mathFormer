#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reinforcement Learning Training Script using GRPO

This script implements RL training for the MathFormer model using Group Relative
Policy Optimization (GRPO). It loads a pretrained checkpoint, generates random
math problems, samples multiple solutions per problem, and trains using GRPO.
"""

import torch
import torch.nn as nn
import argparse
import csv
import os
import copy
from typing import List, Tuple
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, *a, **k):
        return x

from src.model import AutoRegressiveTransformerModel
from src.prepare_data import build_vocab
from src.generate_data import GenConfig, stream_samples, GeneratorResult
from src.evaluate import check_correctness, split_scratchpad_and_result, _infer_model_hparams
from src.grpo import GRPOConfig, GRPOTrainer, create_attention_mask
from src.modelparam import NInp, NHead, NHid, NLayers, Dropout

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def generate_batch_problems(
    gen_config: GenConfig,
    batch_size: int
) -> List[GeneratorResult]:
    """Generate a batch of random math problems.
    
    Args:
        gen_config: Generation configuration
        batch_size: Number of problems to generate
        
    Returns:
        List of GeneratorResult objects
    """
    stream = stream_samples(gen_config)
    problems = []
    for _ in range(batch_size):
        problems.append(next(stream))
    return problems


def sample_response(
    model: nn.Module,
    prompt_tokens: torch.Tensor,
    max_new_tokens: int,
    eos_token: int,
    temperature: float = 1.0
) -> torch.Tensor:
    """Generate a single response using nucleus/top-p sampling.
    
    Args:
        model: The model to use for generation
        prompt_tokens: Prompt token tensor of shape (T,)
        max_new_tokens: Maximum number of tokens to generate
        eos_token: EOS token ID
        temperature: Sampling temperature (1.0 = normal, <1.0 = sharper, >1.0 = more random)
        
    Returns:
        Generated sequence including prompt (T_total,)
    """
    model.eval()
    
    if prompt_tokens.dim() == 1:
        prompt_tokens = prompt_tokens.unsqueeze(0)  # (1, T)
    
    batch_size = prompt_tokens.shape[0]
    current_seq = prompt_tokens.clone()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get logits for next token
            logits = model(current_seq)  # (B, T, V)
            next_token_logits = logits[:, -1, :] / temperature  # (B, V)
            
            # Sample from distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            
            # Append to sequence
            current_seq = torch.cat([current_seq, next_token], dim=1)
            
            # Check for EOS
            if (next_token == eos_token).all():
                break
    
    return current_seq.squeeze(0)  # Return (T_total,)


def generate_batch_responses(
    model: nn.Module,
    problems: List[GeneratorResult],
    vocab: dict,
    num_samples: int,
    max_len: int = 256,
    temperature: float = 1.0
) -> Tuple[List[str], List[torch.Tensor], List[str]]:
    """Generate multiple responses for each problem.
    
    Args:
        model: Model to use for generation
        problems: List of math problems
        vocab: Vocabulary dictionary
        num_samples: Number of responses per problem
        max_len: Maximum generation length
        temperature: Sampling temperature
        
    Returns:
        Tuple of (problem_exprs, response_tensors, response_strs):
            - problem_exprs: List of problem expressions (repeated num_samples times)
            - response_tensors: List of full response token tensors
            - response_strs: List of decoded response strings
    """
    inv_vocab = {i: char for char, i in vocab.items()}
    
    problem_exprs = []
    response_tensors = []
    response_strs = []
    
    for problem in problems:
        # Create prompt for each problem (assume big endian for consistency)
        expression = problem.exprBigEndian
        prompt_str = f'{expression}'
        prompt = (
            [vocab['<sos>']] +
            [vocab['<big>']] +
            [vocab[c] for c in expression] +
            [vocab['<scratchpad>']]
        )
        prompt_tensor = torch.tensor(prompt, dtype=torch.long).to(device)
        
        # Generate num_samples responses for this problem
        for _ in range(num_samples):
            generated = sample_response(
                model,
                prompt_tensor,
                max_new_tokens=max_len,
                eos_token=vocab['<eos>'],
                temperature=temperature
            )
            
            # Decode the full sequence
            decoded = [inv_vocab[int(i)] for i in generated[1:]]  # skip <sos>
            full_seq = "".join(decoded)
            
            problem_exprs.append(expression)
            response_tensors.append(generated)
            response_strs.append(full_seq)
    
    return problem_exprs, response_tensors, response_strs


def compute_rewards(
    expressions: List[str],
    responses: List[str],
    vocab: dict
) -> torch.Tensor:
    """Compute binary correctness rewards for responses.
    
    Args:
        expressions: List of mathematical expressions
        responses: List of response strings
        vocab: Vocabulary dictionary
        
    Returns:
        Tensor of rewards (1.0 for correct, 0.0 for incorrect)
    """
    rewards = []
    
    for expr, response in zip(expressions, responses):
        # Split response into scratchpad and result
        _, result = split_scratchpad_and_result(response)
        
        # Check correctness
        correct = check_correctness(expr, result)
        reward = 1.0 if correct else 0.0
        rewards.append(reward)
    
    return torch.tensor(rewards, dtype=torch.float32, device=device)


def prepare_batch_tensors(
    response_tensors: List[torch.Tensor],
    vocab: dict,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare padded batch tensors and attention masks.
    
    Args:
        response_tensors: List of variable-length response tensors
        vocab: Vocabulary dictionary
        device: Device to put tensors on
        
    Returns:
        Tuple of (padded_input_ids, attention_mask)
    """
    pad_token = vocab['<scratchpad>']  # Use scratchpad token as pad
    max_len = max(len(seq) for seq in response_tensors)
    batch_size = len(response_tensors)
    
    # Create padded tensor
    input_ids = torch.full((batch_size, max_len), pad_token, dtype=torch.long, device=device)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.float32, device=device)
    
    for i, seq in enumerate(response_tensors):
        seq_len = len(seq)
        input_ids[i, :seq_len] = seq
        attention_mask[i, :seq_len] = 1.0
    
    return input_ids, attention_mask


def evaluate_on_test_set(
    model: nn.Module,
    gen_config: GenConfig,
    vocab: dict,
    num_samples: int = 100
) -> Tuple[float, int, int]:
    """Evaluate model on a test set of random problems.
    
    Args:
        model: Model to evaluate
        gen_config: Generation configuration for test problems
        vocab: Vocabulary dictionary
        num_samples: Number of test problems
        
    Returns:
        Tuple of (accuracy, correct_count, total_count)
    """
    model.eval()
    inv_vocab = {i: char for char, i in vocab.items()}
    
    stream = stream_samples(gen_config)
    correct_count = 0
    total_count = 0
    
    with torch.no_grad():
        for _ in range(num_samples):
            problem = next(stream)
            expression = problem.exprBigEndian
            
            # Create prompt
            prompt = (
                [vocab['<sos>']] +
                [vocab['<big>']] +
                [vocab[c] for c in expression] +
                [vocab['<scratchpad>']]
            )
            prompt_tensor = torch.tensor(prompt, dtype=torch.long).to(device)
            
            # Generate response (greedy)
            generated = model.generate(
                prompt_tensor,
                max_new_tokens=256,
                eos_token=vocab['<eos>']
            )
            
            # Decode and check correctness
            decoded = [inv_vocab[int(i)] for i in generated[1:]]
            full_seq = "".join(decoded)
            _, result = split_scratchpad_and_result(full_seq)
            
            correct = check_correctness(expression, result)
            total_count += 1
            if correct:
                correct_count += 1
    
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    return accuracy, correct_count, total_count


def train_rl(args):
    """Main RL training loop using GRPO.
    
    Args:
        args: Command-line arguments
    """
    print(f"Using device: {device}")
    
    # Build vocabulary
    vocab = build_vocab()
    pad_token = vocab['<scratchpad>']
    
    # Load pretrained checkpoint
    if not args.checkpoint:
        raise ValueError("Must provide --checkpoint path to pretrained model")
    
    print(f"Loading checkpoint from {args.checkpoint}")
    raw = torch.load(args.checkpoint, map_location=device)
    
    # Handle different checkpoint formats
    if 'state_dict' in raw and isinstance(raw['state_dict'], dict):
        state_dict = raw['state_dict']
    else:
        state_dict = raw
    
    # Infer model hyperparameters
    ntoken, ninp_ckpt, nhid_ckpt, nlayers_ckpt = _infer_model_hparams(state_dict)
    
    # Create policy model and load weights
    model = AutoRegressiveTransformerModel(
        ntoken, ninp_ckpt, NHead, nhid_ckpt, nlayers_ckpt, Dropout
    ).to(device)
    model.load_state_dict(state_dict, strict=False)
    
    # Create reference model (frozen copy)
    ref_model = AutoRegressiveTransformerModel(
        ntoken, ninp_ckpt, NHead, nhid_ckpt, nlayers_ckpt, Dropout
    ).to(device)
    ref_model.load_state_dict(state_dict, strict=False)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    print(f"Model loaded: {nlayers_ckpt} layers, {ninp_ckpt} dim, {nhid_ckpt} hidden")
    
    # Create GRPO config and trainer
    grpo_config = GRPOConfig(
        group_size=args.group_size,
        kl_coef=args.kl_coef,
        clip_ratio=args.clip_ratio,
        temperature=args.temperature,
        max_grad_norm=args.grad_clip
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    trainer = GRPOTrainer(
        model=model,
        ref_model=ref_model,
        vocab=vocab,
        config=grpo_config,
        optimizer=optimizer,
        device=device
    )
    
    # Generation config for problems
    gen_config = GenConfig(
        max_depth_cap=args.max_depth,
        max_digits=args.max_digits,
        seed=None  # Random seed for variety
    )
    
    # Baseline evaluation
    print("Evaluating baseline model...")
    baseline_acc, baseline_correct, baseline_total = evaluate_on_test_set(
        model, gen_config, vocab, num_samples=100
    )
    print(f"Baseline: {baseline_acc:.2%} ({baseline_correct}/{baseline_total})")
    
    # Create checkpoint directory
    os.makedirs("checkpoints", exist_ok=True)
    
    # Training loop
    step_iter = tqdm(range(args.steps), desc="RL Training")
    
    for step in step_iter:
        # Generate batch of problems
        problems = generate_batch_problems(gen_config, args.batch_size)
        
        # Generate multiple responses per problem
        expressions, response_tensors, response_strs = generate_batch_responses(
            model, problems, vocab, args.group_size, args.max_gen_len, args.temperature
        )
        
        # Compute rewards
        rewards = compute_rewards(expressions, response_strs, vocab)
        
        # Prepare batch tensors
        input_ids, attention_mask = prepare_batch_tensors(response_tensors, vocab, device)
        
        # Compute old log probabilities (before update)
        with torch.no_grad():
            old_logprobs, _ = trainer.compute_logprobs(model, input_ids, attention_mask)
        
        # GRPO training step
        metrics = trainer.train_step(input_ids, attention_mask, rewards, old_logprobs)
        
        # Update progress bar
        step_iter.set_postfix(
            loss=f"{metrics['policy_loss']:.4f}",
            reward=f"{metrics['avg_reward']:.3f}",
            kl=f"{metrics['kl_divergence']:.4f}"
        )
        
        # Periodic evaluation
        if (step + 1) % args.eval_interval == 0:
            eval_acc, eval_correct, eval_total = evaluate_on_test_set(
                model, gen_config, vocab, num_samples=100
            )
            print(f"\nStep {step+1} Eval: {eval_acc:.2%} ({eval_correct}/{eval_total})")
            metrics['eval_acc'] = eval_acc
            metrics['eval_correct'] = eval_correct
            metrics['eval_total'] = eval_total
        
        # Log metrics
        if args.log_csv:
            write_rl_log(args.log_csv, step, metrics, args)
        
        # Save checkpoint
        if (step + 1) % args.save_interval == 0:
            checkpoint_path = f"checkpoints/rl_model_step{step+1}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"\nSaved checkpoint to {checkpoint_path}")
    
    # Final evaluation
    print("\nFinal evaluation...")
    final_acc, final_correct, final_total = evaluate_on_test_set(
        model, gen_config, vocab, num_samples=100
    )
    print(f"Final: {final_acc:.2%} ({final_correct}/{final_total})")
    print(f"Improvement: {final_acc - baseline_acc:.2%}")
    
    # Save final model
    final_path = "checkpoints/rl_model_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Saved final model to {final_path}")


def write_rl_log(path: str, step: int, metrics: dict, args):
    """Write RL training logs to CSV.
    
    Args:
        path: Path to CSV file
        step: Current training step
        metrics: Dictionary of metrics
        args: Command-line arguments
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    
    header = [
        'phase', 'step', 'total_loss', 'policy_loss', 'kl_divergence',
        'avg_reward', 'std_reward', 'clip_fraction', 'grad_norm', 'entropy',
        'eval_acc', 'eval_correct', 'eval_total', 'lr', 'group_size'
    ]
    
    file_exists = os.path.isfile(path)
    
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        
        row = {
            'phase': 'rl_train',
            'step': step,
            'total_loss': metrics.get('total_loss', ''),
            'policy_loss': metrics.get('policy_loss', ''),
            'kl_divergence': metrics.get('kl_divergence', ''),
            'avg_reward': metrics.get('avg_reward', ''),
            'std_reward': metrics.get('std_reward', ''),
            'clip_fraction': metrics.get('clip_fraction', ''),
            'grad_norm': metrics.get('grad_norm', ''),
            'entropy': metrics.get('entropy', ''),
            'eval_acc': metrics.get('eval_acc', ''),
            'eval_correct': metrics.get('eval_correct', ''),
            'eval_total': metrics.get('eval_total', ''),
            'lr': args.lr,
            'group_size': args.group_size
        }
        writer.writerow(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL training with GRPO for MathFormer')
    
    # Model and checkpoint
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to pretrained model checkpoint')
    
    # Training parameters
    parser.add_argument('--steps', type=int, default=1000,
                        help='Number of training steps')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Number of problems per batch')
    parser.add_argument('--group_size', type=int, default=4,
                        help='Number of responses per problem')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate')
    
    # GRPO parameters
    parser.add_argument('--kl_coef', type=float, default=0.1,
                        help='KL divergence coefficient')
    parser.add_argument('--clip_ratio', type=float, default=0.2,
                        help='Clipping ratio for surrogate loss')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping norm')
    
    # Problem generation
    parser.add_argument('--max_digits', type=int, default=2,
                        help='Maximum digits in generated problems')
    parser.add_argument('--max_depth', type=int, default=3,
                        help='Maximum expression depth')
    parser.add_argument('--max_gen_len', type=int, default=256,
                        help='Maximum generation length')
    
    # Logging and checkpointing
    parser.add_argument('--save_interval', type=int, default=100,
                        help='Save checkpoint every N steps')
    parser.add_argument('--eval_interval', type=int, default=50,
                        help='Evaluate every N steps')
    parser.add_argument('--log_csv', type=str, default='',
                        help='CSV log file path')
    
    args = parser.parse_args()
    train_rl(args)
