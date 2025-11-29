#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration tests for RL training with GRPO
"""

import torch
import pytest
import os
from src.train_rl import (
    generate_batch_problems,
    sample_response,
    prepare_batch_tensors
)
from src.generate_data import GenConfig
from src.prepare_data import build_vocab
from src.model import AutoRegressiveTransformerModel


def test_generate_batch_problems():
    """Test random problem generation."""
    gen_config = GenConfig(max_depth_cap=2, max_digits=1, seed=42)
    batch_size = 5
    
    problems = generate_batch_problems(gen_config, batch_size)
    
    assert len(problems) == batch_size
    assert all(hasattr(p, 'exprBigEndian') for p in problems)
    assert all(hasattr(p, 'result') for p in problems)
    
    print(f"✓ Generated {batch_size} problems")
    for i, p in enumerate(problems[:3]):
        print(f"  Problem {i+1}: {p.exprBigEndian} = {p.result}")


def test_sample_response():
    """Test response sampling from model."""
    vocab = build_vocab()
    device = torch.device('cpu')
    
    # Create small model for testing
    model = AutoRegressiveTransformerModel(
        ntoken=len(vocab),
        ninp=64,
        nhead=2,
        nhid=128,
        nlayers=2,
        dropout=0.1
    ).to(device)
    
    # Create a simple prompt
    prompt = torch.tensor([vocab['<sos>'], vocab['1'], vocab['+'], vocab['2']], dtype=torch.long).to(device)
    
    # Sample response
    response = sample_response(
        model,
        prompt,
        max_new_tokens=20,
        eos_token=vocab['<eos>'],
        temperature=1.0
    )
    
    # Check that response includes prompt
    assert len(response) >= len(prompt)
    assert (response[:len(prompt)] == prompt).all()
    
    print(f"✓ Sampled response: length {len(response)}")


def test_prepare_batch_tensors():
    """Test preparation of padded batch tensors."""
    vocab = build_vocab()
    device = torch.device('cpu')
    
    # Create variable-length sequences
    seq1 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long).to(device)
    seq2 = torch.tensor([1, 2, 3], dtype=torch.long).to(device)
    seq3 = torch.tensor([1, 2, 3, 4, 5, 6, 7], dtype=torch.long).to(device)
    
    response_tensors = [seq1, seq2, seq3]
    
    input_ids, attention_mask = prepare_batch_tensors(response_tensors, vocab, device)
    
    # Check shapes
    assert input_ids.shape == (3, 7)  # batch_size=3, max_len=7
    assert attention_mask.shape == (3, 7)
    
    # Check padding
    assert attention_mask[0, :5].sum() == 5  # First sequence has 5 tokens
    assert attention_mask[1, :3].sum() == 3  # Second sequence has 3 tokens
    assert attention_mask[2, :7].sum() == 7  # Third sequence has 7 tokens
    
    print(f"✓ Prepared batch tensors: {input_ids.shape}")


def test_end_to_end_single_step():
    """Test a single step of RL training end-to-end."""
    from src.grpo import GRPOConfig, GRPOTrainer
    
    vocab = build_vocab()
    device = torch.device('cpu')
    
    # Create small model
    model = AutoRegressiveTransformerModel(
        ntoken=len(vocab),
        ninp=64,
        nhead=2,
        nhid=128,
        nlayers=2,
        dropout=0.1
    ).to(device)
    
    # Create reference model (copy)
    ref_model = AutoRegressiveTransformerModel(
        ntoken=len(vocab),
        ninp=64,
        nhead=2,
        nhid=128,
        nlayers=2,
        dropout=0.1
    ).to(device)
    ref_model.load_state_dict(model.state_dict())
    ref_model.eval()
    
    # Create trainer
    config = GRPOConfig(group_size=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    trainer = GRPOTrainer(model, ref_model, vocab, config, optimizer, device)
    
    # Generate a batch of problems
    gen_config = GenConfig(max_depth_cap=2, max_digits=1, seed=123)
    problems = generate_batch_problems(gen_config, batch_size=2)
    
    # Create dummy sequences (in practice, these would come from generation)
    # For testing, just create random sequences
    seq_len = 20
    input_ids = torch.randint(0, len(vocab), (4, seq_len), dtype=torch.long).to(device)
    attention_mask = torch.ones(4, seq_len, dtype=torch.float32).to(device)
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32).to(device)
    
    # Compute old logprobs
    with torch.no_grad():
        old_logprobs, _ = trainer.compute_logprobs(model, input_ids, attention_mask)
    
    # Training step
    metrics = trainer.train_step(input_ids, attention_mask, rewards, old_logprobs)
    
    # Check that metrics are returned
    assert 'policy_loss' in metrics
    assert 'kl_divergence' in metrics
    assert 'avg_reward' in metrics
    assert 'clip_fraction' in metrics
    
    print(f"✓ End-to-end training step completed")
    print(f"  Policy loss: {metrics['policy_loss']:.4f}")
    print(f"  KL divergence: {metrics['kl_divergence']:.4f}")
    print(f"  Avg reward: {metrics['avg_reward']:.4f}")


if __name__ == '__main__':
    print("Running RL training integration tests...\n")
    
    test_generate_batch_problems()
    print()
    
    test_sample_response()
    print()
    
    test_prepare_batch_tensors()
    print()
    
    test_end_to_end_single_step()
    print()
    
    print("✓ All integration tests passed!")
