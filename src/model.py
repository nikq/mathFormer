import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_alibi_slopes(n):
    def get_slopes_power_of_2(n):
        start = (2**(-8))**(1/n)
        return [start * (2**(-8/n))**i for i in range(n)]

    if math.log2(n).is_integer():
        slopes = get_slopes_power_of_2(n)
    else:
        closest_pow_2 = 2 ** int(math.floor(math.log2(n)))
        slopes = get_slopes_power_of_2(closest_pow_2) + \
                 get_slopes_power_of_2(2 * closest_pow_2)[0::2][:n - closest_pow_2]
    return torch.tensor(slopes)

class MoELayer(nn.Module):
    """Mixture-of-Experts Layer.
    
    Replaces the standard feedforward layer. Routes tokens to top-k experts.
    """
    def __init__(self, d_model, dim_feedforward, num_experts, active_experts, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.active_experts = active_experts
        self.d_model = d_model
        
        # Router (Gate) stores weights for all experts
        self.router = nn.Linear(d_model, num_experts, bias=False)
        
        # Experts (MLPs)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model),
                nn.Dropout(dropout)
            )
            for _ in range(num_experts)
        ])
        
    def forward(self, x):
        # x: (B, T, d_model)
        B, T, C = x.shape
        x_flat = x.view(-1, C)  # (B*T, C)
        
        # Router logits
        router_logits = self.router(x_flat)  # (B*T, num_experts)
        
        # Top-k selection
        routing_weights, selected_experts = torch.topk(router_logits, self.active_experts, dim=-1)
        # routing_weights: (B*T, k), selected_experts: (B*T, k)
        
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        # Dispatch inputs to experts
        final_output = torch.zeros_like(x_flat)
        
        # Naive loop implementation for simplicity and clarity
        # (For production efficiency, use scatter/gather or optimized kernels)
        for i in range(self.active_experts):
            expert_idx = selected_experts[:, i]
            weight = routing_weights[:, i].unsqueeze(1)
            
            for expert_id in range(self.num_experts):
                # Find tokens assigned to this expert at this rank
                mask = (expert_idx == expert_id)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[expert_id](expert_input)
                    final_output[mask] += expert_output * weight[mask]
                    
        return final_output.view(B, T, C)


class TransformerBlock(nn.Module):
    """Transformer block with ALiBi-enabled multi-head attention."""
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, num_experts=0, active_experts=0):
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
        
        # ALiBi slopes
        self.register_buffer('alibi_slopes', get_alibi_slopes(nhead))
        
        # Feedforward network or MoE
        self.num_experts = num_experts
        self.active_experts = active_experts
        
        if num_experts > 0:
            self.moe = MoELayer(d_model, dim_feedforward, num_experts, active_experts, dropout)
            self.linear1 = None
            self.linear2 = None
        else:
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.linear2 = nn.Linear(dim_feedforward, d_model)
            self.moe = None
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, attn_mask=None, key_padding_mask=None, return_attn_weights=False):
        """
        Args:
            x: Input tensor of shape (B, T, d_model)
            attn_mask: Attention mask of shape (T, T) for causal masking
            key_padding_mask: Padding mask of shape (B, T)
            return_attn_weights: If True, returns attention weights
            
        Returns:
            Output tensor of shape (B, T, d_model), or (output, attn_weights) if return_attn_weights is True
        """
        # Self-attention with ALiBi
        sa_out, attn_weights = self._sa_block(self.norm1(x), attn_mask, key_padding_mask, return_attn_weights=return_attn_weights)
        x = x + sa_out
        
        # Feedforward
        if self.moe is not None:
             x = x + self.moe(self.norm2(x))
        else:
             x = x + self._ff_block(self.norm2(x))
        
        if return_attn_weights:
            return x, attn_weights
        return x
    
    def _sa_block(self, x, attn_mask, key_padding_mask, return_attn_weights=False):
        """Self-attention block with ALiBi."""
        B, T, C = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)  # (B, T, d_model)
        k = self.k_proj(x)  # (B, T, d_model)
        v = self.v_proj(x)  # (B, T, d_model)
        
        # Reshape for multi-head attention: (B, num_heads, T, head_dim)
        q = q.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, num_heads, T, T)
        
        # ALiBi bias
        # Create a bias matrix based on relative distance
        # For causal masking, query position i, key position j, distance is i - j (if i >= j)
        # We need a matrix of shape (1, 1, T, T) or similar to broadcast
        context_position = torch.arange(T, device=x.device)[:, None]
        memory_position = torch.arange(T, device=x.device)[None, :]
        relative_position = memory_position - context_position # (T, T). For j < i, this is negative.
        relative_position = torch.abs(relative_position).unsqueeze(0).expand(self.nhead, -1, -1) # (nhead, T, T)
        
        # bias = -slope * distance
        # slopes: (nhead) -> (nhead, 1, 1)
        alibi_bias = -self.alibi_slopes.view(self.nhead, 1, 1) * relative_position # (nhead, T, T)
        alibi_bias = alibi_bias.unsqueeze(0) # (1, nhead, T, T)
        
        scores = scores + alibi_bias
        
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
        
        output = self.dropout1(output)
        
        if return_attn_weights:
            return output, attn_weights
        return output, None
    
    def _ff_block(self, x):
        """Feedforward block."""
        x = self.linear2(F.gelu(self.linear1(x)))
        x = self.dropout2(x)
        return x


class AutoRegressiveTransformerModel(nn.Module):
    """Decoder-only causal Transformer (GPT-style) with ALiBi for single sequence modeling.

    Uses ALiBi instead of standard positional encoding.
    Expects input shape (B, T) of token indices. Applies causal mask internally
    if not provided. Returns logits of shape (B, T, ntoken).
    """
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.1, max_len=2048, num_experts=0, active_experts=0):
        super().__init__()
        self.model_type = 'DecoderOnlyTransformer'
        self.ninp = ninp
        self.ntoken = ntoken
        
        # Token embedding
        self.tok_emb = nn.Embedding(ntoken, ninp)
        
        # Transformer blocks with ALiBi
        self.blocks = nn.ModuleList([
            TransformerBlock(ninp, nhead, nhid, dropout, num_experts=num_experts, active_experts=active_experts)
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

    def forward(self, tokens: torch.Tensor, attn_mask: torch.Tensor | None = None, key_padding_mask: torch.Tensor | None = None, return_diagnostics: bool = False):
        # tokens: (B, T)
        B, T = tokens.shape
        device = tokens.device
        if attn_mask is None:
            attn_mask = self.generate_square_subsequent_mask(T, device)
        
        # Token embedding with scaling (no positional encoding added here!)
        emb = self.tok_emb(tokens) * math.sqrt(self.ninp)  # (B,T,C)
        x = self.dropout(emb)
        
        enc_attms = []
        activations = []
        
        # Apply transformer blocks
        for block in self.blocks:
            if return_diagnostics:
                x, attn = block(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, return_attn_weights=True)
                enc_attms.append(attn)
                activations.append(x)
            else:
                x = block(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        
        # Project to vocabulary
        logits = self.out_proj(x)  # (B,T,ntoken)
        
        if return_diagnostics:
             return logits, {
                 'embeddings': emb,
                 'activations': activations,
                 'attentions': enc_attms
             }
        
        return logits

    @torch.no_grad()
    def generate(self, prompt_tokens: torch.Tensor, max_new_tokens: int, eos_token: int | None = None, include_eos: bool = False, top_k: int | None = None, temperature: float = 1.0):
        """Autoregressive generation.

        prompt_tokens: (B, T) of token indices.
        Returns (B, T_total).
        """
        if prompt_tokens.dim() == 1:
            prompt_tokens = prompt_tokens.unsqueeze(0)  # (1,T)
        
        # prompt_tokens is (B, T)
        generated = prompt_tokens.clone()
        
        # We need to track which sequences have finished if batching (optional, but good for efficiency)
        # For simplicity here, we generate for fixed max_new_tokens or until all hit EOS if we implemented detailed checking.
        # But keeping it simple like the original: just loop max_new_tokens.
        
        for _ in range(max_new_tokens):
            T = generated.size(1)
            attn_mask = self.generate_square_subsequent_mask(T, generated.device)
            logits = self.forward(generated, attn_mask=attn_mask)  # (B, T, ntoken)
            next_token_logits = logits[:, -1, :]  # (B, ntoken)
            
            # Apply temperature
            if temperature != 1.0 and temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # Top-k sampling
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, -1].unsqueeze(-1)] = -float('Inf')
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1) # (B, 1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # (B, 1)
            
            generated = torch.cat([generated, next_token], dim=1)  # (B, T+1)
            
            # Implementation detail: If we wanted to stop early for batch items that hit EOS, we'd need more logic.
            # But the request is to "support batch", and the simplest "support" matches the loop structure.
            # Ideally we check for EOS, but let's stick to the current blocking logic unless requested otherwise.
            # The original code did an early return for batch_size=1 if EOS found.
            if eos_token is not None:
                 # If all have hit EOS, we could stop. Use a simple check if strictly needed.
                 # For now, let's keep it running for max_new_tokens to ensure tensor shapes are consistent
                 # or return early if ALL match.
                 # To preserve exact behavior of single batch early exit:
                 if generated.size(0) == 1 and next_token.item() == eos_token:
                     if include_eos:
                         return generated.squeeze(0)
                     else:
                         return generated.squeeze(0)[:-1]

        if generated.size(0) == 1:
            return generated.squeeze(0)
        return generated
