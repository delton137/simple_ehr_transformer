"""
Data loader for tokenized Patient Health Timelines (PHTs)
Handles batching, sequence preparation, and data augmentation for training
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from tqdm import tqdm
import random

from config import model_config

logger = logging.getLogger(__name__)

class PHTDataset(Dataset):
    """Dataset for Patient Health Timelines"""
    
    def __init__(self, tokenized_timelines: Dict[int, List[int]], 
                 max_seq_len: int = 2048, pad_token_id: int = 0):
        self.tokenized_timelines = tokenized_timelines
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        
        # Convert to list of (patient_id, tokens) pairs
        self.data = list(tokenized_timelines.items())
        
        # Filter out very short sequences
        self.data = [(pid, tokens) for pid, tokens in self.data if len(tokens) > 10]
        
        logger.info(f"Created dataset with {len(self.data)} patient timelines")
        logger.info(f"Max sequence length: {max_seq_len}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single training example"""
        patient_id, tokens = self.data[idx]
        
        # Create input and target sequences
        if len(tokens) <= self.max_seq_len:
            # Sequence fits within max length
            input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
            target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        else:
            # Sequence is too long, take a random slice
            start_idx = random.randint(0, len(tokens) - self.max_seq_len - 1)
            end_idx = start_idx + self.max_seq_len
            input_ids = torch.tensor(tokens[start_idx:end_idx-1], dtype=torch.long)
            target_ids = torch.tensor(tokens[start_idx+1:end_idx], dtype=torch.long)
        
        return input_ids, target_ids

class PHTDataLoader:
    """Data loader for PHT training with batching and padding"""
    
    def __init__(self, dataset: PHTDataset, batch_size: int = 32, 
                 shuffle: bool = True, num_workers: int = 4):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        
        # Create data loader
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True
        )
    
    def _collate_fn(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Custom collate function to handle variable length sequences"""
        input_ids, target_ids = zip(*batch)
        
        # Pad sequences to the maximum length in the batch
        max_len = max(len(ids) for ids in input_ids)
        
        # Pad input sequences
        padded_input_ids = []
        for ids in input_ids:
            if len(ids) < max_len:
                padding = torch.full((max_len - len(ids),), self.dataset.pad_token_id, dtype=torch.long)
                padded_ids = torch.cat([ids, padding])
            else:
                padded_ids = ids
            padded_input_ids.append(padded_ids)
        
        # Pad target sequences
        padded_target_ids = []
        for ids in target_ids:
            if len(ids) < max_len:
                padding = torch.full((max_len - len(ids),), self.dataset.pad_token_id, dtype=torch.long)
                padded_ids = torch.cat([ids, padding])
            else:
                padded_ids = ids
            padded_target_ids.append(padded_ids)
        
        # Stack into batches
        batch_input_ids = torch.stack(padded_input_ids)
        batch_target_ids = torch.stack(padded_target_ids)
        
        return batch_input_ids, batch_target_ids
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)

class PHTDataProcessor:
    """Data processor for training data preparation"""
    
    def __init__(self, tokenized_timelines: Dict[int, List[int]], 
                 vocab_size: int, train_split: float = 0.9):
        self.tokenized_timelines = tokenized_timelines
        self.vocab_size = vocab_size
        self.train_split = train_split
        
        # Split data into train/validation sets
        self.train_data, self.val_data = self._split_data()
        
        logger.info(f"Split data: {len(self.train_data)} train, {len(self.val_data)} validation")
    
    def _split_data(self) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
        """Split data into training and validation sets"""
        patient_ids = list(self.tokenized_timelines.keys())
        random.shuffle(patient_ids)
        
        split_idx = int(len(patient_ids) * self.train_split)
        train_ids = patient_ids[:split_idx]
        val_ids = patient_ids[split_idx:]
        
        train_data = {pid: self.tokenized_timelines[pid] for pid in train_ids}
        val_data = {pid: self.tokenized_timelines[pid] for pid in val_ids}
        
        return train_data, val_data
    
    def create_datasets(self, max_seq_len: int = 2048) -> Tuple[PHTDataset, PHTDataset]:
        """Create training and validation datasets"""
        train_dataset = PHTDataset(self.train_data, max_seq_len)
        val_dataset = PHTDataset(self.val_data, max_seq_len)
        
        return train_dataset, val_dataset
    
    def create_dataloaders(self, batch_size: int = 32, 
                          num_workers: int = 4) -> Tuple[PHTDataLoader, PHTDataLoader]:
        """Create training and validation data loaders"""
        train_dataset, val_dataset = self.create_datasets()
        
        train_loader = PHTDataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers
        )
        
        val_loader = PHTDataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers
        )
        
        return train_loader, val_loader

def create_continuous_sequence(tokenized_timelines: Dict[int, List[int]], 
                             max_seq_len: int = 2048) -> List[int]:
    """
    Create a continuous sequence from all patient timelines for training
    This approach concatenates all timelines with separator tokens
    """
    continuous_sequence = []
    
    for patient_id, tokens in tokenized_timelines.items():
        # Add start of sequence token
        continuous_sequence.append(3)  # SOS token
        
        # Add patient timeline tokens
        continuous_sequence.extend(tokens)
        
        # Add end of sequence token
        continuous_sequence.append(2)  # EOS token
    
    return continuous_sequence

class ContinuousPHTDataset(Dataset):
    """Dataset for continuous sequence training (alternative approach)"""
    
    def __init__(self, continuous_sequence: List[int], max_seq_len: int = 2048):
        self.continuous_sequence = continuous_sequence
        self.max_seq_len = max_seq_len
        
        # Create training examples by sliding window
        self.examples = []
        for i in range(0, len(continuous_sequence) - max_seq_len):
            self.examples.append(i)
        
        logger.info(f"Created continuous dataset with {len(self.examples)} examples")
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single training example"""
        start_idx = self.examples[idx]
        end_idx = start_idx + self.max_seq_len
        
        # Get sequence
        sequence = self.continuous_sequence[start_idx:end_idx]
        
        # Split into input and target
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        target_ids = torch.tensor(sequence[1:], dtype=torch.long)
        
        return input_ids, target_ids

def create_continuous_dataloader(tokenized_timelines: Dict[int, List[int]], 
                               batch_size: int = 32, max_seq_len: int = 2048,
                               num_workers: int = 4) -> PHTDataLoader:
    """Create a data loader for continuous sequence training"""
    continuous_sequence = create_continuous_sequence(tokenized_timelines, max_seq_len)
    dataset = ContinuousPHTDataset(continuous_sequence, max_seq_len)
    
    return PHTDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

def analyze_data_distribution(tokenized_timelines: Dict[int, List[int]]) -> Dict:
    """Analyze the distribution of sequence lengths and tokens"""
    sequence_lengths = [len(tokens) for tokens in tokenized_timelines.values()]
    all_tokens = []
    for tokens in tokenized_timelines.values():
        all_tokens.extend(tokens)
    
    analysis = {
        'num_patients': len(tokenized_timelines),
        'total_tokens': len(all_tokens),
        'avg_sequence_length': np.mean(sequence_lengths),
        'median_sequence_length': np.median(sequence_lengths),
        'min_sequence_length': np.min(sequence_lengths),
        'max_sequence_length': np.max(sequence_lengths),
        'std_sequence_length': np.std(sequence_lengths),
        'unique_tokens': len(set(all_tokens)),
        'token_frequency': {}
    }
    
    # Count token frequencies
    from collections import Counter
    token_counts = Counter(all_tokens)
    analysis['token_frequency'] = dict(token_counts.most_common(20))
    
    return analysis

if __name__ == "__main__":
    # Example usage
    import pickle
    
    # Load processed data
    with open('processed_data/tokenized_timelines.pkl', 'rb') as f:
        tokenized_timelines = pickle.load(f)
    
    with open('processed_data/vocabulary.pkl', 'rb') as f:
        vocab = pickle.load(f)
    
    # Analyze data
    analysis = analyze_data_distribution(tokenized_timelines)
    print("Data Analysis:")
    for key, value in analysis.items():
        if key != 'token_frequency':
            print(f"  {key}: {value}")
    
    print("\nTop 20 most frequent tokens:")
    for token, count in analysis['token_frequency'].items():
        print(f"  Token {token}: {count}")
    
    # Create data processor
    processor = PHTDataProcessor(tokenized_timelines, len(vocab))
    
    # Create dataloaders
    train_loader, val_loader = processor.create_dataloaders(batch_size=16)
    
    print(f"\nCreated dataloaders:")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    
    # Test a batch
    for batch_idx, (input_ids, target_ids) in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Input shape: {input_ids.shape}")
        print(f"  Target shape: {target_ids.shape}")
        print(f"  Sample input: {input_ids[0, :10].tolist()}")
        print(f"  Sample target: {target_ids[0, :10].tolist()}")
        break
