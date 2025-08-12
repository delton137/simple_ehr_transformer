#!/usr/bin/env python3
"""
Smaller ETHOS Transformer (~30M params) with 6 layers and weight tying.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from config import model_config

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.pos_embedding = nn.Parameter(torch.randn(max_len, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.max_len:
            self.pos_embedding = nn.Parameter(torch.randn(seq_len, self.d_model, device=x.device))
            self.max_len = seq_len
        return x + self.pos_embedding[:seq_len, :].unsqueeze(0)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            if mask.dtype is not torch.bool:
                mask = mask.bool()
            neg_inf = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask == 0, neg_inf)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        return torch.matmul(attn, V)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, t, _ = x.size()
        Q = self.w_q(x).view(b, t, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(b, t, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(b, t, self.n_heads, self.d_k).transpose(1, 2)
        out = self.scaled_dot_product_attention(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(b, t, self.d_model)
        return self.w_o(out)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.attn(x, mask)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x

class SmallETHOSTransformer(nn.Module):
    """
    Smaller ETHOS-like transformer
    Defaults: d_model=480, n_heads=8, n_layers=6, d_ff=1920, max_seq_len from config
    Weight tying between token embedding and output projection
    """
    def __init__(self, vocab_size: int, d_model: int = 480, n_heads: int = 8, n_layers: int = 6, d_ff: int = 1920, max_seq_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        # Output projection with weight tying
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        self.output_projection.weight = self.token_embedding.weight
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def create_causal_mask(self, seq_len: int, device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, 0)
        mask = mask.masked_fill(mask == 0, 1)
        return mask.bool()

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, t = input_ids.size()
        causal_mask = self.create_causal_mask(t, input_ids.device)
        if attention_mask is not None:
            causal_mask = causal_mask & attention_mask.unsqueeze(1)
        x = self.token_embedding(input_ids)
        x = self.dropout(x)
        x = self.pos_encoding(x)
        for blk in self.blocks:
            x = blk(x, causal_mask)
        logits = self.output_projection(x)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_small_ethos_model(vocab_size: int) -> SmallETHOSTransformer:
    model = SmallETHOSTransformer(
        vocab_size=vocab_size,
        d_model=480,
        n_heads=8,
        n_layers=6,
        d_ff=1920,
        max_seq_len=model_config.max_seq_len,
        dropout=model_config.dropout,
    )
    return model
