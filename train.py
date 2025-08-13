#!/usr/bin/env python3
"""
Training script for ETHOS transformer model on EHR data
Based on the ETHOS paper methodology for training transformer models on Patient Health Timelines
"""

import os
import logging
import argparse
import time
import math
import csv
from typing import Optional
import torch
import torch.nn as nn
import torch.optim as optim
from config import model_config
from data_loader import PHTDataProcessor, analyze_data_distribution, PHTDataLoader
from model import create_ethos_model, ETHOSTransformer

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
    
    def __init__(self, model, train_loader, val_loader, device, config, model_dir: str = "models"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.use_amp: bool = bool(config.get('use_amp', False))
        self.grad_accum_steps: int = int(config.get('grad_accum_steps', 1))
        self.model_dir = model_dir
        
        # Training components
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding token
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=0.01
        )
        
        # Learning rate scheduler: Linear warmup -> Cosine decay
        self.warmup_steps: int = int(config.get('warmup_steps', 2000))
        self.steps_per_epoch: int = max(1, math.ceil(len(train_loader) / self.grad_accum_steps))
        self.total_steps: int = self.steps_per_epoch * int(config['max_epochs'])

        def lr_lambda(step: int):
            if step < self.warmup_steps:
                return float(step) / max(1, self.warmup_steps)
            progress = float(step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        from torch.optim.lr_scheduler import LambdaLR
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        
        if self.use_amp and torch.cuda.is_available():
            self.scaler = torch.amp.GradScaler()
        else:
            self.scaler = None
        
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.last_val_loss: Optional[float] = None
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.log_every: int = int(config.get('log_every', 200))
        # Step-based hooks
        self.validate_every_steps: int = int(config.get('validate_every_steps', 0))
        self.checkpoint_every_steps: int = int(config.get('checkpoint_every_steps', 0))
        self.global_step: int = 0  # counts optimizer update steps
        
        # CSV logging for training progress
        self.csv_log_path = os.path.join(self.model_dir, 'training_progress.csv')
        self._setup_csv_logging()
        
        # Create output directories
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        os.makedirs('plots', exist_ok=True)
    
    def _setup_csv_logging(self):
        """Setup CSV logging for training progress"""
        # Check if CSV file already exists (for resuming)
        csv_exists = os.path.exists(self.csv_log_path)
        
        if csv_exists:
            logger.info(f"CSV file exists, will append to: {self.csv_log_path}")
        else:
            # Create CSV file with headers
            with open(self.csv_log_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['iteration', 'epoch', 'train_loss', 'last_val_loss', 'learning_rate'])
            logger.info(f"Created new CSV file: {self.csv_log_path}")
        
        logger.info(f"CSV logging enabled: {self.csv_log_path}")
    
    def _log_to_csv(self, iteration: int, epoch: int, train_loss: float, last_val_loss: Optional[float], lr: float):
        """Log training progress to CSV (no scientific notation, no timestamp)."""
        with open(self.csv_log_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            last_val_str = "" if last_val_loss is None else f"{last_val_loss:.6f}"
            writer.writerow([iteration, epoch, f"{train_loss:.6f}", last_val_str, f"{lr:.6f}"])
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        running_loss = 0.0
        running_count = 0
        # Throughput timing for periodic logs
        last_log_time = time.time()
        last_log_step = 0
        # Accumulators for step-based CSV logging
        accum_step_loss: float = 0.0  # sum of losses across the current grad accumulation window
        loss_since_log: float = 0.0   # sum of optimizer-step losses since last CSV log
        steps_since_log: int = 0

        for batch_idx, (input_ids, target_ids) in enumerate(self.train_loader):
            # Move to device
            input_ids = input_ids.to(self.device, non_blocking=True)
            target_ids = target_ids.to(self.device, non_blocking=True)
            
            # Forward pass
            if (batch_idx % self.grad_accum_steps) == 0:
                self.optimizer.zero_grad(set_to_none=True)
                accum_step_loss = 0.0
            if self.scaler is not None:
                with torch.amp.autocast('cuda'):
                    logits = self.model(input_ids)
                    batch_size, seq_len, vocab_size = logits.size()
                    logits = logits.view(-1, vocab_size)
                    targets = target_ids.view(-1)
                    loss = self.criterion(logits, targets)
                    loss = loss / self.grad_accum_steps
                self.scaler.scale(loss).backward()
                accum_step_loss += float(loss.item())
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
                accum_step_loss += float(loss.item())
            
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
                self.global_step += 1

                # Record optimizer-step loss for CSV logging
                step_loss_value = accum_step_loss  # summed per-batch loss in this accumulation window
                loss_since_log += step_loss_value
                steps_since_log += 1

                if self.validate_every_steps > 0 and (self.global_step % self.validate_every_steps == 0):
                    _val_t0 = time.time()
                    val_loss = self.validate_epoch()
                    _val_time = time.time() - _val_t0

                    is_best = val_loss < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_loss
                    self.last_val_loss = val_loss
                    logger.info(
                        f"[Step {self.global_step}] Validation Loss: {val_loss:.4f} (best: {self.best_val_loss:.4f}) | "
                        f"Val Time: {_val_time:.2f}s"
                    )
                    # Optionally save best checkpoint at step
                    if is_best:
                        self.save_checkpoint(is_best=True)

                # Step-based checkpoint caching (snapshot)
                if self.checkpoint_every_steps > 0 and (self.global_step % self.checkpoint_every_steps == 0):
                    self.save_step_checkpoint(self.global_step)
            
            # Update metrics
            step_loss = loss.item() * self.grad_accum_steps
            total_loss += step_loss
            running_loss += step_loss
            running_count += 1
            num_batches += 1
            # Log learning rate
            self.learning_rates.append(self.scheduler.get_last_lr()[0])

            # Periodic console logging by batches (unchanged)
            if ((batch_idx + 1) % self.log_every == 0):
                now = time.time()
                steps_since = (batch_idx + 1) - last_log_step
                elapsed = max(1e-9, now - last_log_time)
                time_per_100 = (elapsed / max(1, steps_since)) * 100.0
                avg_running = running_loss / max(1, running_count)
                logger.info(
                    f"Epoch {self.current_epoch + 1} | Step {batch_idx + 1}/{len(self.train_loader)} | "
                    f"Loss: {avg_running:.4f} | LR: {self.scheduler.get_last_lr()[0]:.6f} | "
                    f"t/100 iters: {time_per_100:.2f}s"
                )
                running_loss = 0.0
                running_count = 0
                last_log_time = now
                last_log_step = batch_idx + 1

            # CSV logging every N optimizer steps
            if steps_since_log > 0 and (self.global_step % self.log_every == 0):
                avg_train_since = loss_since_log / float(steps_since_log)
                self._log_to_csv(
                    iteration=self.global_step,
                    epoch=self.current_epoch,
                    train_loss=avg_train_since,
                    last_val_loss=self.last_val_loss,
                    lr=self.scheduler.get_last_lr()[0]
                )
                loss_since_log = 0.0
                steps_since_log = 0
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        return avg_loss
    
    def validate_epoch(self) -> float:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for input_ids, target_ids in self.val_loader:
                # Move to device
                input_ids = input_ids.to(self.device, non_blocking=True)
                target_ids = target_ids.to(self.device, non_blocking=True)
                
                # Forward pass
                if self.scaler is not None and torch.cuda.is_available():
                    with torch.amp.autocast('cuda'):
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
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        return avg_loss
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
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
        torch.save(checkpoint, os.path.join(self.model_dir, 'latest_checkpoint.pth'))
        
        # Save best checkpoint if this is the best so far
        if is_best:
            torch.save(checkpoint, os.path.join(self.model_dir, 'best_checkpoint.pth'))
            logger.info(f"Saved best checkpoint with validation loss: {self.best_val_loss:.4f}")

    def save_step_checkpoint(self, step: int):
        """Save a snapshot checkpoint with step number to allow caching."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'config': self.config
        }
        filename = os.path.join(self.model_dir, f'checkpoint_step_{step:07d}.pth')
        torch.save(checkpoint, filename)
        logger.info(f"Saved step checkpoint: {filename}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = int(checkpoint.get('global_step', 0))
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.learning_rates = checkpoint['learning_rates']
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    
    def auto_resume_from_checkpoint(self) -> bool:
        """Automatically resume from the latest available checkpoint if it exists"""
        
        checkpoint_path = os.path.join(self.model_dir, 'latest_checkpoint.pth')
        if not(os.path.exists(checkpoint_path)):
            logger.info("No checkpoint found - starting training from scratch")
            self.current_epoch = 0
            self.global_step = 0
            self.best_val_loss = float('inf')
            self.train_losses = []
            self.val_losses = []
            self.learning_rates = []
        else:
            logger.info(f"Found checkpoint!!! ")
            self.load_checkpoint(checkpoint_path)
            
            # Check if we have a CSV file to determine the last logged iteration
            if os.path.exists(self.csv_log_path):
                try:
                    # Read the last line to get the last logged iteration
                    with open(self.csv_log_path, 'r') as csvfile:
                        lines = csvfile.readlines()
                        if len(lines) > 1:  # Has header + at least one data row
                            last_line = lines[-1].strip()
                            if last_line:
                                last_iteration = int(last_line.split(',')[0])
                                logger.info(f"Last logged iteration from CSV: {last_iteration}")
                                # Update global_step if CSV shows a later iteration
                                if last_iteration > self.global_step:
                                    logger.info(f"Updating global_step from {self.global_step} to {last_iteration}")
                                    self.global_step = last_iteration
                except Exception as e:
                    logger.warning(f"Could not read last iteration from CSV: {e}")
            
            logger.info(f"✓ Auto-resumed training from epoch {self.current_epoch + 1}, step {self.global_step}")
            return True
            

#-------------------------------------------------------------------------------------------
    def train(self, resume_from: str = None):
        """Main training loop with automatic checkpoint detection"""
        logger.info("Starting training...")
        # Auto-resume from latest checkpoint if available
        self.auto_resume_from_checkpoint()
        
        # Training loop
        for epoch in range(self.current_epoch, self.config['max_epochs']):
            self.current_epoch = epoch
            
            start_time = time.time()
            train_loss = self.train_epoch()
            train_time = time.time() - start_time
            
            _val_t0_epoch = time.time()
            val_loss = self.validate_epoch()
            val_time_epoch = time.time() - _val_t0_epoch
            
            # Update best validation loss
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            # Log metrics
            logger.info(
                f"Epoch {epoch + 1}/{self.config['max_epochs']} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f} - "
                f"Time: {train_time:.2f}s - Val Time: {val_time_epoch:.2f}s"
            )
            # Update last validation loss for reference
            self.last_val_loss = val_loss
            
            # Store losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Save checkpoint
            self.save_checkpoint(is_best=is_best)
            
        # Final message
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
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Training batch size per process (default: 8)')
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='Maximum training epochs (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                       help='Learning rate (default: 3e-4)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use (default: auto)')

    parser.add_argument('--max_seq_len', type=int, default=1024,
                       help='Max sequence length per sample (default: 1024)')
    parser.add_argument('--use_amp', action='store_true', help='Enable mixed precision (AMP). If omitted and CUDA is available, AMP will be auto-enabled.')
    parser.add_argument('--grad_accum_steps', type=int, default=4, help='Gradient accumulation steps (default: 4)')
    parser.add_argument('--warmup_steps', type=int, default=2000, help='LR warmup steps before cosine decay (default: 2000)')
    parser.add_argument('--log_every', type=int, default=200, help='Steps between loss logs (default: 200)')
    parser.add_argument('--num_workers', type=int, default=8, help='DataLoader workers (default: 8)')
    parser.add_argument('--validate_every_steps', type=int, default=4000, help='Run validation every N optimizer steps (default: 4000). Set 0 to disable step-based validation.')
    parser.add_argument('--checkpoint_every_steps', type=int, default=20000, help='Save a snapshot checkpoint every N optimizer steps (default: 20000). Set 0 to disable step snapshots.')
    parser.add_argument('--print_timeline_stats', action='store_true',
                       help='Print train/val patient counts and training timeline length histogram before training')
    parser.add_argument('--print_random_timeline', action='store_true',
                       help='Print a random patient timeline (decoded tokens) before training')

    args = parser.parse_args()

    # Favor less fragmentation on GPUs
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

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
    print(f"⚙️  Batch size (per process): {args.batch_size}")
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
    
    # Auto-enable AMP on CUDA if user didn't specify flag
    if device.type == 'cuda' and not args.use_amp:
        args.use_amp = True
    
    # Load data
    print("\n📊 Loading data...")
    try:
        tokenized_timelines, vocab = load_processed_data(args.data_dir)
        print(f"✅ Loaded {len(tokenized_timelines)} patient timelines")
        print(f"📚 Vocabulary size: {len(vocab)}")
        # Optionally print a random patient timeline
        if args.print_random_timeline and tokenized_timelines:
            import random
            id_to_token = {v: k for k, v in vocab.items()}
            # Prefer longer timelines (>100 tokens); fallback to the longest available
            candidates = [pid for pid, seq in tokenized_timelines.items() if len(seq) > 100]
            if candidates:
                rand_pid = random.choice(candidates)
            else:
                # Fallback: pick the longest timeline
                rand_pid = max(tokenized_timelines.keys(), key=lambda pid: len(tokenized_timelines[pid]))
            seq = tokenized_timelines[rand_pid]
            decoded = [id_to_token.get(t, f'UNKNOWN_{t}') for t in seq]
            print("\n🧪 Sample patient timeline (prefer >100 tokens):")
            print(f"  Patient ID: {rand_pid}")
            print(f"  Sequence length: {len(seq)}")
            preview = 200
            if len(decoded) > preview:
                print(f"  Tokens (first {preview}):")
                print("   ", decoded[:preview])
            else:
                print("  Tokens:")
                print("   ", decoded)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Create data loaders
    print("\n🔧 Creating data loaders...")
    try:
        data_processor = PHTDataProcessor(tokenized_timelines, len(vocab), train_split=0.9, seed=42)
        # Respect max_seq_len from args
        train_dataset, val_dataset = data_processor.create_datasets(max_seq_len=args.max_seq_len)
        # Optional stats before loader creation
        if args.print_timeline_stats:
            def _count_bucket(n: int) -> str:
                if n <= 10:
                    return '0-10'
                if n <= 20:
                    return '10-20'
                if n <= 100:
                    return '20-100'
                if n <= 200:
                    return '100-200'
                if n <= 800:
                    return '200-800'
                return '>800'
            train_lengths = [len(seq) for seq in data_processor.train_data.values()]
            val_patients = len(data_processor.val_data)
            train_patients = len(data_processor.train_data)
            buckets = {'0-10':0,'10-20':0,'20-100':0,'100-200':0,'200-800':0,'>800':0}
            for n in train_lengths:
                buckets[_count_bucket(n)] += 1
            print("\n📊 Timeline stats (pre-dataloader):")
            print(f"  Train patients: {train_patients}")
            print(f"  Val patients:   {val_patients}")
            print("  Training timeline length histogram:")
            for k in ['0-10','10-20','20-100','100-200','200-800','>800']:
                print(f"    {k}: {buckets[k]}")

        train_loader = PHTDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
        val_loader = PHTDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
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
        try:
            num_params = model.count_parameters()  # type: ignore[attr-defined]
            print(f"✅ Model created with {num_params:,} parameters")
        except Exception:
            print("✅ Model created")
    except RuntimeError as e:
        # Handle OOM at model init by retrying with a smaller config
        oom = ('out of memory' in str(e).lower())
        if oom:
            print("⚠️ OOM during model creation. Retrying with smaller configuration (d_model=384, n_heads=6, d_ff=1536)...")
            try:
                model = ETHOSTransformer(
                    vocab_size=len(vocab),
                    d_model=384,
                    n_heads=6,
                    n_layers=6,
                    d_ff=1536,
                    max_seq_len=args.max_seq_len,
                    dropout=model_config.dropout,
                ).to(device)
                try:
                    num_params = model.count_parameters()  # type: ignore[attr-defined]
                    print(f"✅ Model created with {num_params:,} parameters (fallback)")
                except Exception:
                    print("✅ Model created (fallback)")
            except Exception as ee:
                print(f"❌ Error creating fallback model: {ee}")
                return
        else:
            print(f"❌ Error creating model: {e}")
            return
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return
    
    # Setup training via ETHOSTrainer
    print("\n⚙️  Setting up training...")
    try:
        trainer_config = {
            'learning_rate': args.learning_rate,
            'max_epochs': args.max_epochs,
            'gradient_clip': model_config.gradient_clip,
            'use_amp': args.use_amp,
            'grad_accum_steps': args.grad_accum_steps,
            'warmup_steps': args.warmup_steps,
            'log_every': args.log_every,
            'validate_every_steps': args.validate_every_steps,
            'checkpoint_every_steps': args.checkpoint_every_steps,
        }
        # Determine model output directory (tag-aware)
        model_dir = os.path.join('models', args.tag) if args.tag else 'models'
        os.makedirs(model_dir, exist_ok=True)
        print(f"💾 Models will be saved to: {model_dir}/")
        trainer = ETHOSTrainer(model, train_loader, val_loader, device, trainer_config, model_dir=model_dir)
        trainer.train()
    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()
