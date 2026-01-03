#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRPO (Group Relative Policy Optimization) Implementation

This module implements the GRPO algorithm for reinforcement learning training
of language models. GRPO eliminates the need for a separate value function by
comparing multiple generated responses for each input and using group-relative
advantages for policy updates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Tuple, Optional
import math


@dataclass
class GRPOConfig:
    """Configuration for GRPO training.
    
    Attributes:
        group_size: Number of responses to generate per problem
        kl_coef: Coefficient for KL divergence penalty
        clip_ratio: Clipping parameter for surrogate loss (ε)
        temperature: Sampling temperature for generation
        max_grad_norm: Maximum gradient norm for clipping
    """
    group_size: int = 4
    kl_coef: float = 0.1
    clip_ratio: float = 0.2
    temperature: float = 1.0
    max_grad_norm: float = 1.0
    top_k: Optional[int] = None


class GRPOTrainer:
    """GRPO trainer for policy optimization using group-relative advantages.
    
    The trainer manages the training loop for GRPO, which includes:
    1. Generating multiple responses per problem (group sampling)
    2. Computing rewards and group-relative advantages
    3. Updating the policy using clipped surrogate loss with KL divergence penalty
    """
    
    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        vocab: dict,
        config: GRPOConfig,
        optimizer: torch.optim.Optimizer,
        device: torch.device
    ):
        """Initialize GRPO trainer.
        
        Args:
            model: The policy model to train
            ref_model: Reference model (frozen) for KL divergence
            vocab: Vocabulary dictionary mapping tokens to indices
            config: GRPO configuration
            optimizer: Optimizer for policy updates
            device: Device to run training on
        """
        self.model = model
        self.ref_model = ref_model
        self.vocab = vocab
        self.inv_vocab = {i: char for char, i in vocab.items()}
        self.config = config
        self.optimizer = optimizer
        self.device = device
        
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()
    
    def compute_group_advantages(
        self,
        rewards: torch.Tensor,
        group_size: int
    ) -> torch.Tensor:
        """Compute advantages using group-wise normalization.
        
        For each group of responses to the same problem, normalize rewards
        to have mean 0 and std 1 within the group.
        
        Args:
            rewards: Tensor of shape (batch_size * group_size,) containing rewards
            group_size: Number of responses per problem
            
        Returns:
            Advantages tensor of same shape as rewards
        """
        # Reshape to (batch_size, group_size)
        batch_size = rewards.shape[0] // group_size
        rewards_grouped = rewards.view(batch_size, group_size)
        
        # Compute group-wise mean and std
        mean = rewards_grouped.mean(dim=1, keepdim=True)  # (batch_size, 1)
        std = rewards_grouped.std(dim=1, keepdim=True) + 1e-8  # (batch_size, 1)
        
        # Normalize within each group
        advantages = (rewards_grouped - mean) / std
        
        # Flatten back to (batch_size * group_size,)
        return advantages.view(-1)
    
    def compute_logprobs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute log probabilities for given sequences.
        
        Args:
            model: Model to use for computing logprobs
            input_ids: Token IDs of shape (B, T)
            attention_mask: Attention mask of shape (B, T)
            
        Returns:
            Tuple of (logprobs, entropy):
                - logprobs: Log probabilities of shape (B, T-1)
                - entropy: Entropy of the distribution of shape (B, T-1)
        """
        # Get logits from model
        logits = model(input_ids[:, :-1])  # (B, T-1, V)
        
        # Get log probabilities
        log_probs = F.log_softmax(logits, dim=-1)  # (B, T-1, V)
        
        # Gather log probs of actual tokens
        target_ids = input_ids[:, 1:]  # (B, T-1)
        token_logprobs = torch.gather(
            log_probs,
            dim=-1,
            index=target_ids.unsqueeze(-1)
        ).squeeze(-1)  # (B, T-1)
        
        # Compute entropy
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)  # (B, T-1)
        
        # Apply attention mask
        token_logprobs = token_logprobs * attention_mask[:, 1:]
        entropy = entropy * attention_mask[:, 1:]
        
        return token_logprobs, entropy
    
    def compute_policy_loss(
        self,
        logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute clipped surrogate loss (PPO-style).
        
        Args:
            logprobs: Current log probabilities (B, T-1)
            old_logprobs: Old log probabilities (B, T-1)
            advantages: Advantages for each sequence (B,)
            attention_mask: Attention mask (B, T)
            
        Returns:
            Tuple of (policy_loss, clip_fraction):
                - policy_loss: Scalar loss value
                - clip_fraction: Fraction of updates that were clipped
        """
        # Sum log probs over sequence length (weighted by mask)
        mask = attention_mask[:, 1:]  # (B, T-1)
        seq_logprobs = (logprobs * mask).sum(dim=1)  # (B,)
        old_seq_logprobs = (old_logprobs * mask).sum(dim=1)  # (B,)
        
        # Compute ratio
        ratio = torch.exp(seq_logprobs - old_seq_logprobs)  # (B,)
        
        # Expand advantages to match
        advantages = advantages.view(-1)  # (B,)
        
        # Compute surrogate losses
        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio,
            1.0 - self.config.clip_ratio,
            1.0 + self.config.clip_ratio
        ) * advantages
        
        # Take minimum (for loss maximization, we negate later)
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Compute clip fraction for monitoring
        clip_fraction = ((ratio - 1.0).abs() > self.config.clip_ratio).float().mean()
        
        return policy_loss, clip_fraction
    
    def compute_kl_divergence(
        self,
        logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute KL divergence between current and reference policy.
        
        Args:
            logprobs: Current policy log probabilities (B, T-1)
            ref_logprobs: Reference policy log probabilities (B, T-1)
            attention_mask: Attention mask (B, T)
            
        Returns:
            Mean KL divergence across the batch
        """
        # KL(ref || current) using log probabilities
        # This version uses sequence-level KL
        mask = attention_mask[:, 1:]  # (B, T-1)
        
        # Sum over sequence length
        seq_logprobs = (logprobs * mask).sum(dim=1)  # (B,)
        ref_seq_logprobs = (ref_logprobs * mask).sum(dim=1)  # (B,)
        
        # Approximate KL at sequence level
        kl = ref_seq_logprobs - seq_logprobs  # (B,)
        
        return kl.mean()
    
    def train_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        rewards: torch.Tensor,
        old_logprobs: torch.Tensor
    ) -> dict:
        """Execute one GRPO training step.
        
        Args:
            input_ids: Token IDs of shape (B, T)
            attention_mask: Attention mask of shape (B, T)
            rewards: Rewards of shape (B,)
            old_logprobs: Old log probabilities of shape (B, T-1)
            
        Returns:
            Dictionary containing loss metrics
        """
        self.model.train()
        
        # Compute advantages using group normalization
        advantages = self.compute_group_advantages(rewards, self.config.group_size)
        
        # Compute current log probabilities
        logprobs, entropy = self.compute_logprobs(self.model, input_ids, attention_mask)
        
        # Compute reference log probabilities (no gradients)
        with torch.no_grad():
            ref_logprobs, _ = self.compute_logprobs(self.ref_model, input_ids, attention_mask)
        
        # Compute policy loss
        policy_loss, clip_fraction = self.compute_policy_loss(
            logprobs, old_logprobs, advantages, attention_mask
        )
        
        # Compute KL divergence penalty
        kl_div = self.compute_kl_divergence(logprobs, ref_logprobs, attention_mask)
        
        # Total loss
        total_loss = policy_loss + self.config.kl_coef * kl_div
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm
        )
        
        self.optimizer.step()
        
        # Return metrics
        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'kl_divergence': kl_div.item(),
            'clip_fraction': clip_fraction.item(),
            'grad_norm': grad_norm.item(),
            'entropy': entropy.mean().item(),
            'avg_reward': rewards.mean().item(),
            'std_reward': rewards.std().item(),
        }


def create_attention_mask(sequences: List[torch.Tensor], pad_token: int) -> torch.Tensor:
    """Create attention mask from variable-length sequences.
    
    Args:
        sequences: List of token tensors
        pad_token: Token ID used for padding
        
    Returns:
        Attention mask tensor of shape (B, max_len)
    """
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    
    mask = torch.zeros(batch_size, max_len, dtype=torch.float32)
    
    for i, seq in enumerate(sequences):
        mask[i, :len(seq)] = 1.0
    
    return mask
