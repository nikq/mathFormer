import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for batch_first tensors.

    Expects input of shape (B, T, C) and adds positional embeddings along T.
    """
    def __init__(self, d_model, dropout=0.1, max_len=4096):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # store as (1, max_len, d_model) for broadcasting over batch
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        T = x.size(1)
        x = x + self.pe[:, :T, :]
        return self.dropout(x)

class AutoRegressiveTransformerModel(nn.Module):
    """Decoder-only causal Transformer (GPT-style) for single sequence modeling.

    Uses batch_first=True for better inference performance & nested tensor optimization.
    Expects input shape (B, T) of token indices. Applies causal mask internally
    if not provided. Returns logits of shape (B, T, ntoken).
    """
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.1, max_len=2048):
        super().__init__()
        self.model_type = 'DecoderOnlyTransformer'
        self.ninp = ninp
        self.ntoken = ntoken
        self.pos_encoder = PositionalEncoding(ninp, dropout, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=ninp, nhead=nhead, dim_feedforward=nhid, dropout=dropout, batch_first=True)
        self.block_stack = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.tok_emb = nn.Embedding(ntoken, ninp)
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
        x = self.tok_emb(tokens) * math.sqrt(self.ninp)  # (B,T,C)
        x = self.pos_encoder(x)
        x = self.block_stack(x, mask=attn_mask, src_key_padding_mask=key_padding_mask)  # (B,T,C)
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