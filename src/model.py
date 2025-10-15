import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.ninp = ninp
        self.encoder = nn.Embedding(ntoken, ninp)
        self.transformer = nn.Transformer(ninp, nhead, nlayers, nlayers, nhid, dropout)
        self.decoder = nn.Linear(ninp, ntoken)
        self.init_weights()

    def generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, tgt, src_mask, tgt_mask, src_key_padding_mask):
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        tgt = self.encoder(tgt) * math.sqrt(self.ninp)
        tgt = self.pos_encoder(tgt)
        output = self.transformer(src, tgt, src_mask, tgt_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(output)
        return output


class AutoRegressiveTransformerModel(nn.Module):
    """Decoder-only causal Transformer (GPT-style) for single sequence modeling.

    Expects input shape (T, B) of token indices. Applies causal mask internally
    if not provided. Returns logits of shape (T, B, ntoken).
    """
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.1, max_len=2048):
        super().__init__()
        self.model_type = 'DecoderOnlyTransformer'
        self.ninp = ninp
        self.ntoken = ntoken
        self.pos_encoder = PositionalEncoding(ninp, dropout, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=ninp, nhead=nhead, dim_feedforward=nhid, dropout=dropout, batch_first=False)
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
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        # True where we want to mask (upper triangle), convert to float -inf/0
        float_mask = torch.zeros(sz, sz, device=device)
        float_mask.masked_fill_(mask, float('-inf'))
        return float_mask

    def forward(self, tokens: torch.Tensor, attn_mask: torch.Tensor | None = None, key_padding_mask: torch.Tensor | None = None):
        # tokens: (T, B)
        T, B = tokens.shape
        device = tokens.device
        if attn_mask is None:
            attn_mask = self.generate_square_subsequent_mask(T, device)
        x = self.tok_emb(tokens) * math.sqrt(self.ninp)
        x = self.pos_encoder(x)
        x = self.block_stack(x, mask=attn_mask, src_key_padding_mask=key_padding_mask)
        logits = self.out_proj(x)
        return logits

    @torch.no_grad()
    def generate(self, prompt_tokens: torch.Tensor, max_new_tokens: int, eos_token: int | None = None, include_eos: bool = False):
        # prompt_tokens: (T, B) or (T,) -> ensure (T,B)
        if prompt_tokens.dim() == 1:
            prompt_tokens = prompt_tokens.unsqueeze(1)
        generated = prompt_tokens
        for _ in range(max_new_tokens):
            attn_mask = self.generate_square_subsequent_mask(generated.size(0), generated.device)
            logits = self.forward(generated, attn_mask=attn_mask)
            next_token_logits = logits[-1, 0, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token.view(1,1)], dim=0)
            if eos_token is not None and next_token.item() == eos_token:
                if include_eos:
                    return generated.squeeze(1)
                else:
                    return generated[:-1].squeeze(1)  # exclude eos
        return generated.squeeze(1)  # (T')