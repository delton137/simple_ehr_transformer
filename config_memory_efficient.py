"""
Custom configuration file for memory-efficient training
This shows how to override the default config.py settings
"""

from config import ModelConfig

# Create a custom config that inherits from the base ModelConfig
class MemoryEfficientConfig(ModelConfig):
    """Memory-efficient configuration for V100 with 16GB RAM"""
    
    # Model architecture - smaller model
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 6
    d_ff: int = 1536
    max_seq_len: int = 512
    
    # Training parameters - smaller batches
    batch_size: int = 4
    grad_accum_steps: int = 1  # No gradient accumulation to save memory
    
    # Memory optimization
    use_amp: bool = True
    
    # Data loading - fewer workers
    num_workers: int = 4
    
    # Logging - less frequent to save memory
    log_every: int = 100
    validate_every_steps: int = 2000
    checkpoint_every_steps: int = 10000

# Export the config instance
model_config = MemoryEfficientConfig()
