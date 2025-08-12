#!/usr/bin/env python3
"""
Training script for ETHOS transformer model on EHR data
Based on the ETHOS paper methodology for training transformer models on Patient Health Timelines
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging
import argparse
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from typing import Optional


from config import model_config, data_config
from data_loader import PHTDataProcessor, analyze_data_distribution, PHTDataset, PHTDataLoader
from transformer_model import create_ethos_model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ETHOSTrainer:
    """Trainer class for ETHOS transformer model"""
    
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.use_amp: bool = bool(config.get('use_amp', False))
        self.grad_accum_steps: int = int(config.get('grad_accum_steps', 1))
        
        # Training components
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding token
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=len(train_loader) * config['max_epochs'],
            eta_min=1e-6
        )
        
        # AMP scaler
        self.scaler: Optional[torch.cuda.amp.GradScaler]
        if self.use_amp and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        # Create output directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        os.makedirs('plots', exist_ok=True)
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}')
        
        for batch_idx, (input_ids, target_ids) in enumerate(progress_bar):
            # Move to device
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            # Forward pass
            if (batch_idx % self.grad_accum_steps) == 0:
                self.optimizer.zero_grad()
            
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = self.model(input_ids)
                    batch_size, seq_len, vocab_size = logits.size()
                    logits = logits.view(-1, vocab_size)
                    targets = target_ids.view(-1)
                    loss = self.criterion(logits, targets)
                    loss = loss / self.grad_accum_steps
                self.scaler.scale(loss).backward()
            else:
                logits = self.model(input_ids)
                # Reshape for loss calculation
                batch_size, seq_len, vocab_size = logits.size()
                logits = logits.view(-1, vocab_size)
                targets = target_ids.view(-1)
                # Calculate loss
                loss = self.criterion(logits, targets)
                loss = loss / self.grad_accum_steps
                # Backward pass
                loss.backward()
            
            # Step optimizer on accumulation boundary
            if ((batch_idx + 1) % self.grad_accum_steps) == 0 or ((batch_idx + 1) == len(self.train_loader)):
                # Gradient clipping
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip'])
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item() * self.grad_accum_steps
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
            
            # Log learning rate
            self.learning_rates.append(self.scheduler.get_last_lr()[0])
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate_epoch(self) -> float:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for input_ids, target_ids in tqdm(self.val_loader, desc='Validation'):
                # Move to device
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                
                # Forward pass
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        logits = self.model(input_ids)
                        batch_size, seq_len, vocab_size = logits.size()
                        logits = logits.view(-1, vocab_size)
                        targets = target_ids.view(-1)
                        loss = self.criterion(logits, targets)
                else:
                    logits = self.model(input_ids)
                    # Reshape for loss calculation
                    batch_size, seq_len, vocab_size = logits.size()
                    logits = logits.view(-1, vocab_size)
                    targets = target_ids.view(-1)
                    # Calculate loss
                    loss = self.criterion(logits, targets)
                
                # Update metrics
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'config': self.config
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, 'models/latest_checkpoint.pth')
        
        # Save best checkpoint if this is the best so far
        if is_best:
            torch.save(checkpoint, 'models/best_checkpoint.pth')
            logger.info(f"Saved best checkpoint with validation loss: {self.best_val_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.learning_rates = checkpoint['learning_rates']
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def plot_training_curves(self):
        """Plot training and validation curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        ax1.plot(self.train_losses, label='Training Loss', color='blue')
        ax1.plot(self.val_losses, label='Validation Loss', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Learning rate curve
        ax2.plot(self.learning_rates, label='Learning Rate', color='green')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.legend()
        ax2.grid(True)
        ax2.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('plots/training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def train(self, resume_from: str = None):
        """Main training loop"""
        logger.info("Starting training...")
        
        # Resume from checkpoint if specified
        if resume_from and os.path.exists(resume_from):
            self.load_checkpoint(resume_from)
            logger.info(f"Resumed training from epoch {self.current_epoch + 1}")
        
        # Training loop
        for epoch in range(self.current_epoch, self.config['max_epochs']):
            self.current_epoch = epoch
            
            # Train
            start_time = time.time()
            train_loss = self.train_epoch()
            train_time = time.time() - start_time
            
            # Validate
            val_loss = self.validate_epoch()
            
            # Update best validation loss
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            # Log metrics
            logger.info(
                f"Epoch {epoch + 1}/{self.config['max_epochs']} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f} - "
                f"Time: {train_time:.2f}s"
            )
            
            # Store losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Save checkpoint
            self.save_checkpoint(is_best=is_best)
            
            # Plot curves every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.plot_training_curves()
        
        # Final plotting
        self.plot_training_curves()
        logger.info("Training completed!")

def load_processed_data(data_dir: str):
    """Load tokenized timelines and vocabulary pickles from data_dir."""
    import pickle
    with open(os.path.join(data_dir, 'tokenized_timelines.pkl'), 'rb') as f:
        tokenized_timelines = pickle.load(f)
    with open(os.path.join(data_dir, 'vocabulary.pkl'), 'rb') as f:
        vocab = pickle.load(f)
    return tokenized_timelines, vocab

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train ETHOS Transformer Model')
    parser.add_argument('--data_dir', type=str, default='processed_data',
                       help='Directory containing processed data (default: processed_data/)')
    parser.add_argument('--tag', type=str, default=None,
                       help='Dataset tag to use (e.g., aou_2023, mimic_iv)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Training batch size (default: 32)')
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='Maximum training epochs (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                       help='Learning rate (default: 3e-4)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use (default: auto)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume training from checkpoint')
    parser.add_argument('--max_seq_len', type=int, default=model_config.max_seq_len,
                       help='Max sequence length per sample (default: from config)')
    parser.add_argument('--use_amp', action='store_true', help='Enable mixed precision (AMP)')
    parser.add_argument('--grad_accum_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    
    args = parser.parse_args()
    
    # Handle tag-based data directory
    if args.tag and not args.data_dir.startswith('processed_data_'):
        args.data_dir = f"processed_data_{args.tag}"
    
    # Validate data directory
    if not os.path.exists(args.data_dir):
        print(f"❌ Error: Data directory '{args.data_dir}' does not exist!")
        print(f"Available directories:")
        available_dirs = [d for d in os.listdir('.') if d.startswith('processed_data')]
        if available_dirs:
            for d in available_dirs:
                print(f"  - {d}")
        else:
            print("  - No processed_data directories found")
        print(f"\nTo process data with a tag, run:")
        print(f"  python data_processor.py --tag {args.tag or 'your_tag'}")
        return
    
    print(f"🚀 Starting ETHOS Transformer Training")
    print(f"📁 Data directory: {args.data_dir}")
    if args.tag:
        print(f"🏷️  Dataset tag: {args.tag}")
    print(f"⚙️  Batch size: {args.batch_size}")
    print(f"📈 Max epochs: {args.max_epochs}")
    print(f"📚 Learning rate: {args.learning_rate}")
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"💻 Device: {device}")
    # Enable TF32 for speed/memory on Ampere+ GPUs
    if device.type == 'cuda':
        try:
            torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
            torch.set_float32_matmul_precision('medium')  # type: ignore[attr-defined]
        except Exception:
            pass
    
    # Load data
    print("\n📊 Loading data...")
    try:
        tokenized_timelines, vocab = load_processed_data(args.data_dir)
        print(f"✅ Loaded {len(tokenized_timelines)} patient timelines")
        print(f"📚 Vocabulary size: {len(vocab)}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Create data loaders
    print("\n🔧 Creating data loaders...")
    try:
        data_processor = PHTDataProcessor(tokenized_timelines, len(vocab))
        # Respect max_seq_len from args
        train_dataset, val_dataset = data_processor.create_datasets(max_seq_len=args.max_seq_len)
        train_loader = PHTDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = PHTDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        print(f"✅ Training batches: {len(train_loader)}")
        print(f"✅ Validation batches: {len(val_loader)}")
    except Exception as e:
        print(f"❌ Error creating data loaders: {e}")
        return
    
    # Create model
    print("\n🏗️  Creating model...")
    try:
        model = create_ethos_model(len(vocab))
        model = model.to(device)
        print(f"✅ Model created with {model.count_parameters():,} parameters")
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return
    
    # Multi-GPU support (DataParallel for simplicity)
    if args.device in ('auto', 'cuda') and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"🧮 Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = torch.nn.DataParallel(model)

    # Setup training via ETHOSTrainer
    print("\n⚙️  Setting up training...")
    try:
        trainer_config = {
            'learning_rate': args.learning_rate,
            'max_epochs': args.max_epochs,
            'gradient_clip': model_config.gradient_clip,
            'use_amp': args.use_amp,
            'grad_accum_steps': args.grad_accum_steps,
        }
        trainer = ETHOSTrainer(model, train_loader, val_loader, device, trainer_config)
        trainer.train(resume_from=args.resume)
    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()
