"""
Transformer model for EHR data based on ETHOS architecture
Implements a decoder-only transformer for Patient Health Timeline modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import logging

from config import model_config

logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    """Learnable positional encoding for the transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Learnable positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(max_len, d_model))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        if seq_len > self.max_len:
            # Extend positional embeddings if needed
            self.pos_embedding = nn.Parameter(
                torch.randn(seq_len, self.d_model, device=x.device)
            )
            self.max_len = seq_len
        
        return x + self.pos_embedding[:seq_len, :].unsqueeze(0)

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    
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
        
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, 
                                   V: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute scaled dot-product attention"""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape [batch_size, seq_len, d_model]
            mask: Attention mask, shape [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.size()
        
        # Linear transformations
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Reshape and apply output projection
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        output = self.w_o(attention_output)
        
        return output

class FeedForward(nn.Module):
    """Feed-forward network with residual connection"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
    """Single transformer block with self-attention and feed-forward"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class ETHOSTransformer(nn.Module):
    """
    ETHOS-like transformer model for EHR data
    Decoder-only architecture for sequence modeling
    """
    
    def __init__(self, vocab_size: int, d_model: int = 768, n_heads: int = 12, 
                 n_layers: int = 12, d_ff: int = 3072, max_seq_len: int = 2048, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Create causal mask for autoregressive generation"""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, 0)
        mask = mask.masked_fill(mask == 0, 1)
        return mask.bool()
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the transformer
        
        Args:
            input_ids: Input token IDs, shape [batch_size, seq_len]
            attention_mask: Attention mask, shape [batch_size, seq_len]
            
        Returns:
            Logits for next token prediction, shape [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.size()
        
        # Create causal mask
        causal_mask = self.create_causal_mask(seq_len).to(input_ids.device)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Combine causal mask with attention mask
            causal_mask = causal_mask & attention_mask.unsqueeze(1)
        
        # Token embeddings
        x = self.token_embedding(input_ids)
        x = self.dropout(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, causal_mask)
        
        # Output projection
        logits = self.output_projection(x)
        
        return logits
    
    def generate(self, input_ids: torch.Tensor, max_length: int = 100, 
                temperature: float = 1.0, top_k: int = 50, 
                top_p: float = 0.9, do_sample: bool = True) -> torch.Tensor:
        """
        Generate sequence continuation using the trained model
        
        Args:
            input_ids: Starting sequence, shape [batch_size, seq_len]
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            do_sample: Whether to use sampling or greedy decoding
            
        Returns:
            Generated sequence, shape [batch_size, generated_length]
        """
        batch_size = input_ids.size(0)
        current_ids = input_ids.clone()
        
        for _ in range(max_length):
            # Get model predictions
            with torch.no_grad():
                logits = self.forward(current_ids)
                next_token_logits = logits[:, -1, :] / temperature
            
            if do_sample:
                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, -float('inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Apply top-p (nucleus) sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append next token to sequence
            current_ids = torch.cat([current_ids, next_token], dim=1)
            
            # Check for end of sequence token
            if (next_token == 2).any():  # Assuming 2 is EOS token
                break
        
        return current_ids
    
    def get_sequence_probability(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Get probability of the input sequence
        
        Args:
            input_ids: Input sequence, shape [batch_size, seq_len]
            
        Returns:
            Sequence probabilities, shape [batch_size]
        """
        logits = self.forward(input_ids)
        
        # Get probabilities for each token
        probs = F.softmax(logits, dim=-1)
        
        # Get the probability of the actual next token
        batch_size, seq_len = input_ids.size()
        token_probs = torch.gather(
            probs[:, :seq_len-1, :], 
            dim=2, 
            index=input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        
        # Calculate sequence probability (product of token probabilities)
        sequence_probs = torch.prod(token_probs, dim=1)
        
        return sequence_probs
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def create_ethos_model(vocab_size: int, config: Optional[dict] = None) -> ETHOSTransformer:
    """
    Create ETHOS transformer model with specified configuration
    
    Args:
        vocab_size: Size of the vocabulary
        config: Model configuration dictionary
        
    Returns:
        Initialized ETHOS transformer model
    """
    if config is None:
        config = {
            'd_model': model_config.d_model,
            'n_heads': model_config.n_heads,
            'n_layers': model_config.n_layers,
            'd_ff': model_config.d_ff,
            'max_seq_len': model_config.max_seq_len,
            'dropout': model_config.dropout
        }
    
    model = ETHOSTransformer(
        vocab_size=vocab_size,
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout']
    )
    
    logger.info(f"Created ETHOS model with {model.count_parameters():,} parameters")
    return model

if __name__ == "__main__":
    # Test the model
    vocab_size = 10000
    model = create_ethos_model(vocab_size)
    
    # Test forward pass
    batch_size, seq_len = 4, 128
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        output = model(input_ids)
        print(f"Input shape: {input_ids.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Model parameters: {model.count_parameters():,}")
