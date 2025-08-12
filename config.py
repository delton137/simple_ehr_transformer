"""
Configuration file for ETHOS-like transformer model for EHR data
Optimized for large OMOP datasets
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class ModelConfig:
    """Model configuration parameters"""
    # Model architecture
    vocab_size: int = 50000  # Increased to allow for more concept tokens
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    d_ff: int = 3072
    max_seq_len: int = 2048
    dropout: float = 0.1
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 3e-4
    warmup_steps: int = 4000
    max_epochs: int = 100
    gradient_clip: float = 1.0
    
    # Tokenization parameters
    max_age_tokens: int = 20  # Age intervals (0-5, 5-10, etc.)
    max_time_interval_tokens: int = 13  # Time intervals between events
    max_quantile_tokens: int = 10  # Quantiles for numerical values
    
    # Data processing
    min_time_interval: float = 5.0  # minutes
    max_time_interval: float = 365.25 * 24 * 60  # 1 year in minutes

@dataclass
class DataConfig:
    """Data configuration parameters optimized for large OMOP datasets"""
    # Input paths
    omop_data_dir: str = "omop_data"
    output_dir: str = "processed_data"
    
    # OMOP tables to process (in order of importance)
    omop_tables: List[str] = None
    
    def __post_init__(self):
        if self.omop_tables is None:
            self.omop_tables = [
                "person", "observation_period", "visit_occurrence",
                "condition_occurrence", "drug_exposure", "procedure_occurrence",
                "measurement", "observation", "death"
            ]
    
    # Large dataset optimizations
    chunk_size: int = 10000  # Process data in chunks to manage memory
    max_patients_per_chunk: int = 5000  # Maximum patients to process in memory at once
    use_parallel_processing: bool = True  # Enable parallel processing for large datasets
    memory_limit_gb: float = 8.0  # Memory limit for processing

@dataclass
class TokenConfig:
    """Token configuration for different data types"""
    # Special tokens
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"
    eos_token: str = "<EOS>"
    sos_token: str = "<SOS>"
    
    # Event type tokens
    event_tokens: Dict[str, str] = None
    
    def __post_init__(self):
        if self.event_tokens is None:
            self.event_tokens = {
                "admission": "ADM",
                "discharge": "DIS",
                "condition": "COND",
                "medication": "MED",
                "procedure": "PROC",
                "measurement": "MEAS",
                "observation": "OBS",
                "death": "DEATH",
                "age": "AGE",
                "time_interval": "TIME",
                "quantile": "Q",
                "static": "STATIC"
            }

# Global configuration instances
model_config = ModelConfig()
data_config = DataConfig()
token_config = TokenConfig()

# Create output directories
os.makedirs(data_config.output_dir, exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("temp", exist_ok=True)  # Temporary directory for chunked processing
