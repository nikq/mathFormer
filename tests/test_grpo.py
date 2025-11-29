#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for GRPO components
"""

import torch
import pytest
from src.grpo import GRPOConfig, GRPOTrainer, create_attention_mask


class DummyModel(torch.nn.Module):
    """Dummy model for testing."""
    
    def __init__(self, vocab_size=50, hidden_size=64):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.linear = torch.nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_ids):
        # Model expects input_ids of shape (B, T) and returns logits of shape (B, T, V)
        x = self.embedding(input_ids)
        return self.linear(x)


def test_group_advantage_calculation():
    """Test group-based advantage normalization."""
    device = torch.device('cpu')
    config = GRPOConfig(group_size=4)
    
    # Create dummy models
    model = DummyModel()
    ref_model = DummyModel()
    vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
    optimizer = torch.optim.Adam(model.parameters())
    
    trainer = GRPOTrainer(model, ref_model, vocab, config, optimizer, device)
    
    # Test with 2 groups of 4 responses each
    # Group 1: [0.5, 0.6, 0.7, 0.8]
    # Group 2: [0.1, 0.2, 0.3, 0.4]
    rewards = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.1, 0.2, 0.3, 0.4])
    
    advantages = trainer.compute_group_advantages(rewards, group_size=4)
    
    # Each group should have mean ~0 and std ~1
    advantages_grouped = advantages.view(2, 4)
    
    # Check group 1 (mean should be ~0)
    assert torch.allclose(advantages_grouped[0].mean(), torch.tensor(0.0), atol=1e-5)
    # The std normalization in GRPO uses unbiased=True by default, 
    # so we check that advantages produce reasonable values
    group1_std = advantages_grouped[0].std()
    assert 0.5 < group1_std < 1.5, f"Group 1 std {group1_std} outside expected range"
    
    # Check group 2
    assert torch.allclose(advantages_grouped[1].mean(), torch.tensor(0.0), atol=1e-5)
    group2_std = advantages_grouped[1].std()
    assert 0.5 < group2_std < 1.5, f"Group 2 std {group2_std} outside expected range"
    
    print("✓ Group advantage calculation test passed")


def test_attention_mask_creation():
    """Test attention mask creation for variable-length sequences."""
    seq1 = torch.tensor([1, 2, 3, 4, 5])
    seq2 = torch.tensor([1, 2, 3])
    seq3 = torch.tensor([1, 2, 3, 4, 5, 6, 7])
    
    sequences = [seq1, seq2, seq3]
    pad_token = 0
    
    mask = create_attention_mask(sequences, pad_token)
    
    assert mask.shape == (3, 7)  # max_len = 7
    
    # Check mask values
    assert mask[0, :5].sum() == 5  # First sequence has 5 tokens
    assert mask[0, 5:].sum() == 0
    
    assert mask[1, :3].sum() == 3  # Second sequence has 3 tokens
    assert mask[1, 3:].sum() == 0
    
    assert mask[2, :7].sum() == 7  # Third sequence has 7 tokens
    
    print("✓ Attention mask creation test passed")


def test_logprob_computation():
    """Test log probability computation."""
    device = torch.device('cpu')
    vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
    config = GRPOConfig()
    
    model = DummyModel(vocab_size=50)
    ref_model = DummyModel(vocab_size=50)
    optimizer = torch.optim.Adam(model.parameters())
    
    trainer = GRPOTrainer(model, ref_model, vocab, config, optimizer, device)
    
    # Create sample input
    batch_size = 4
    seq_len = 10
    input_ids = torch.randint(0, 50, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Compute log probabilities
    logprobs, entropy = trainer.compute_logprobs(model, input_ids, attention_mask)
    
    # Check shapes
    assert logprobs.shape == (batch_size, seq_len - 1)
    assert entropy.shape == (batch_size, seq_len - 1)
    
    # Check that logprobs are negative (since they're log probabilities)
    assert (logprobs <= 0).all()
    
    # Check that entropy is non-negative
    assert (entropy >= 0).all()
    
    print("✓ Log probability computation test passed")


def test_kl_divergence():
    """Test KL divergence computation."""
    device = torch.device('cpu')
    vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
    config = GRPOConfig()
    
    model = DummyModel(vocab_size=50)
    ref_model = DummyModel(vocab_size=50)
    optimizer = torch.optim.Adam(model.parameters())
    
    trainer = GRPOTrainer(model, ref_model, vocab, config, optimizer, device)
    
    # Same distribution should have KL ~0
    batch_size = 4
    seq_len = 10
    logprobs = torch.randn(batch_size, seq_len - 1)
    ref_logprobs = logprobs.clone()
    attention_mask = torch.ones(batch_size, seq_len)
    
    kl = trainer.compute_kl_divergence(logprobs, ref_logprobs, attention_mask)
    
    # KL should be close to 0 for identical distributions
    assert torch.allclose(kl, torch.tensor(0.0), atol=1e-5)
    
    print("✓ KL divergence test passed")


def test_policy_loss_clipping():
    """Test that policy loss properly clips ratios."""
    device = torch.device('cpu')
    vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
    config = GRPOConfig(clip_ratio=0.2)
    
    model = DummyModel(vocab_size=50)
    ref_model = DummyModel(vocab_size=50)
    optimizer = torch.optim.Adam(model.parameters())
    
    trainer = GRPOTrainer(model, ref_model, vocab, config, optimizer, device)
    
    batch_size = 8
    seq_len = 10
    
    # Create log probabilities with large difference (should trigger clipping)
    old_logprobs = torch.ones(batch_size, seq_len - 1) * -2.0
    new_logprobs = torch.ones(batch_size, seq_len - 1) * -0.5  # Much higher (ratio > 1+ε)
    
    advantages = torch.randn(batch_size)
    attention_mask = torch.ones(batch_size, seq_len)
    
    loss, clip_frac = trainer.compute_policy_loss(
        new_logprobs, old_logprobs, advantages, attention_mask
    )
    
    # With large differences, clip fraction should be high
    assert clip_frac > 0.5, f"Expected high clip fraction, got {clip_frac}"
    
    # Loss should be a scalar
    assert loss.dim() == 0
    
    print("✓ Policy loss clipping test passed")


if __name__ == '__main__':
    print("Running GRPO unit tests...\n")
    
    test_group_advantage_calculation()
    test_attention_mask_creation()
    test_logprob_computation()
    test_kl_divergence()
    test_policy_loss_clipping()
    
    print("\n✓ All GRPO tests passed!")
