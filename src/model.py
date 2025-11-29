import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).
    
    Applies rotary position embeddings to query and key tensors.
    Based on "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    https://arxiv.org/abs/2104.09864
    """
    def __init__(self, dim, max_len=4096, base=10000.0):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.base = base
        
        # Precompute frequency tensor
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache for cos/sin values
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
    
    def _update_cos_sin_cache(self, seq_len, device):
        """Update cached cos/sin values if sequence length changes."""
        if seq_len > self._seq_len_cached or self._cos_cached is None:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim/2)
            emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()
    
    def forward(self, q, k):
        """Apply rotary embeddings to query and key tensors.
        
        Args:
            q: Query tensor of shape (B, num_heads, T, head_dim)
            k: Key tensor of shape (B, num_heads, T, head_dim)
            
        Returns:
            Tuple of (rotated_q, rotated_k)
        """
        seq_len = q.shape[2]
        self._update_cos_sin_cache(seq_len, q.device)
        
        # Get cos/sin for current sequence length
        cos = self._cos_cached[:seq_len, :self.dim]  # (T, dim)
        sin = self._sin_cached[:seq_len, :self.dim]  # (T, dim)
        
        # Reshape for broadcasting: (1, 1, T, dim)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        
        # Apply rotation
        q_rot = self._apply_rotary_emb(q, cos, sin)
        k_rot = self._apply_rotary_emb(k, cos, sin)
        
        return q_rot, k_rot
    
    def _apply_rotary_emb(self, x, cos, sin):
        """Apply rotary embedding to input tensor."""
        # Split into two halves
        x1, x2 = x[..., :self.dim//2], x[..., self.dim//2:self.dim]
        
        # Apply rotation
        # Rotation formula: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
        cos_half = cos[..., :self.dim//2]
        sin_half = sin[..., :self.dim//2]
        
        rotated = torch.cat([
            x1 * cos_half - x2 * sin_half,
            x1 * sin_half + x2 * cos_half
        ], dim=-1)
        
        # If head_dim > dim, keep the rest unchanged
        if x.shape[-1] > self.dim:
            rotated = torch.cat([rotated, x[..., self.dim:]], dim=-1)
        
        return rotated


class TransformerBlockWithRoPE(nn.Module):
    """Transformer block with RoPE-enabled multi-head attention.
    
    This is a custom implementation that integrates RoPE into the attention mechanism.
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, max_len=4096):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        # Multi-head attention components
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # RoPE
        self.rope = RotaryPositionalEmbedding(self.head_dim, max_len=max_len)
        
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, attn_mask=None, key_padding_mask=None):
        """
        Args:
            x: Input tensor of shape (B, T, d_model)
            attn_mask: Attention mask of shape (T, T) for causal masking
            key_padding_mask: Padding mask of shape (B, T)
            
        Returns:
            Output tensor of shape (B, T, d_model)
        """
        # Self-attention with RoPE
        x = x + self._sa_block(self.norm1(x), attn_mask, key_padding_mask)
        
        # Feedforward
        x = x + self._ff_block(self.norm2(x))
        
        return x
    
    def _sa_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block with RoPE."""
        B, T, C = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)  # (B, T, d_model)
        k = self.k_proj(x)  # (B, T, d_model)
        v = self.v_proj(x)  # (B, T, d_model)
        
        # Reshape for multi-head attention: (B, num_heads, T, head_dim)
        q = q.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        q, k = self.rope(q, k)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, num_heads, T, T)
        
        # Apply attention mask (causal mask)
        if attn_mask is not None:
            scores = scores + attn_mask.unsqueeze(0).unsqueeze(0)  # Broadcast over batch and heads
        
        # Apply key padding mask
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),  # (B, 1, 1, T)
                float('-inf')
            )
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # (B, num_heads, T, head_dim)
        
        # Reshape back: (B, T, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        
        # Final projection
        output = self.out_proj(attn_output)
        output = self.dropout1(output)
        
        return output
    
    def _ff_block(self, x):
        """Feedforward block."""
        x = self.linear2(F.relu(self.linear1(x)))
        x = self.dropout2(x)
        return x


class AutoRegressiveTransformerModel(nn.Module):
    """Decoder-only causal Transformer (GPT-style) with RoPE for single sequence modeling.

    Uses RoPE (Rotary Position Embeddings) instead of standard positional encoding.
    Expects input shape (B, T) of token indices. Applies causal mask internally
    if not provided. Returns logits of shape (B, T, ntoken).
    """
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.1, max_len=2048):
        super().__init__()
        self.model_type = 'DecoderOnlyTransformer'
        self.ninp = ninp
        self.ntoken = ntoken
        
        # Token embedding
        self.tok_emb = nn.Embedding(ntoken, ninp)
        
        # Transformer blocks with RoPE
        self.blocks = nn.ModuleList([
            TransformerBlockWithRoPE(ninp, nhead, nhid, dropout, max_len=max_len)
            for _ in range(nlayers)
        ])
        
        # Output projection
        self.out_proj = nn.Linear(ninp, ntoken)
        self.dropout = nn.Dropout(dropout)
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.02
        nn.init.uniform_(self.tok_emb.weight, -initrange, initrange)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.uniform_(self.out_proj.weight, -initrange, initrange)

    def generate_square_subsequent_mask(self, sz, device):
        # returns (T,T) with -inf where future positions should be masked
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        float_mask = torch.zeros(sz, sz, device=device)
        float_mask.masked_fill_(mask, float('-inf'))
        return float_mask

    def forward(self, tokens: torch.Tensor, attn_mask: torch.Tensor | None = None, key_padding_mask: torch.Tensor | None = None):
        # tokens: (B, T)
        B, T = tokens.shape
        device = tokens.device
        if attn_mask is None:
            attn_mask = self.generate_square_subsequent_mask(T, device)
        
        # Token embedding with scaling (no positional encoding added here!)
        x = self.tok_emb(tokens) * math.sqrt(self.ninp)  # (B,T,C)
        x = self.dropout(x)
        
        # Apply transformer blocks (RoPE is applied internally)
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        
        # Project to vocabulary
        logits = self.out_proj(x)  # (B,T,ntoken)
        return logits

    @torch.no_grad()
    def generate(self, prompt_tokens: torch.Tensor, max_new_tokens: int, eos_token: int | None = None, include_eos: bool = False):
        """Autoregressive greedy generation.

        prompt_tokens: (T,) or (B,T) of token indices. If (T,), assumes batch size 1.
        Returns 1D tensor (T_total,) when batch_size==1.
        """
        if prompt_tokens.dim() == 1:
            prompt_tokens = prompt_tokens.unsqueeze(0)  # (1,T)
        elif prompt_tokens.dim() == 2 and prompt_tokens.size(0) != 1:
            raise ValueError("generate currently supports batch_size==1 only.")
        generated = prompt_tokens  # (1,T)
        for _ in range(max_new_tokens):
            T = generated.size(1)
            attn_mask = self.generate_square_subsequent_mask(T, generated.device)
            logits = self.forward(generated, attn_mask=attn_mask)  # (1,T,ntoken)
            next_token_logits = logits[0, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)  # ()
            # Ensure next_token is (1,1) for cat
            next_token = next_token.view(1, 1)
            generated = torch.cat([generated, next_token], dim=1)  # (1,T+1)
            if eos_token is not None and next_token.item() == eos_token:
                if include_eos:
                    return generated.squeeze(0)
                else:
                    return generated.squeeze(0)[:-1]
        return generated.squeeze(0)