"""
Hybrid CTC-Attention Training Script
Optimized for RTX 4060 (8GB VRAM)

Features:
- FP16 mixed precision training
- Gradient accumulation for effective larger batches
- Dynamic CTC/Attention loss weighting (λ decay)
- Label smoothing
- Cosine annealing with warmup
- Comprehensive checkpointing
- Early stopping with patience
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.amp import autocast
from tqdm import tqdm
import logging
import h5py

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from model_hybrid import HybridCTCAttentionModel


# ==================== CONFIGURATION ====================

class TrainConfig:
    """Training configuration for RTX 4060"""
    
    # Data paths
    data_dir = Path("E:/5thsem el/output_preprocessed")
    csv_path = Path("E:/5thsem el/kortex_5th_sem/data/iSign_v1.1.csv")
    vocab_path = Path("E:/5thsem el/kortex_5th_sem/sign_to_text/vocabulary.pkl")
    checkpoint_dir = Path("E:/5thsem el/kortex_5th_sem/sign_to_text/checkpoints_hybrid")
    
    # Model architecture
    input_dim = 414  # 138 position + 138 velocity + 138 acceleration
    hidden_dim = 384  # Balance accuracy/speed
    embedding_dim = 256
    encoder_layers = 3
    decoder_layers = 2
    num_heads = 4
    dropout = 0.3
    
    # Training - RTX 4060 optimized
    batch_size = 6  # Small due to 8GB VRAM
    gradient_accumulation = 4  # Effective batch = 24
    epochs = 80
    learning_rate = 1e-3
    min_lr = 1e-6
    weight_decay = 1e-5
    warmup_epochs = 5
    
    # CTC/Attention balance
    ctc_weight_start = 0.5  # Start with balanced
    ctc_weight_end = 0.2    # Decay to favor attention
    ctc_weight_decay_epochs = 30  # Epochs to decay over
    
    # Label smoothing
    label_smoothing = 0.1
    
    # Teacher forcing schedule
    tf_start = 1.0
    tf_end = 0.3
    tf_decay_epochs = 40
    
    # Early stopping
    patience = 15
    min_delta = 0.001
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = True  # Mixed precision
    
    # Data loading - use workers for parallel loading
    num_workers = 4  # Parallel data loading - big speedup!
    pin_memory = True
    prefetch_factor = 2  # Prefetch batches ahead
    persistent_workers = True  # Keep workers alive between epochs
    
    # Logging
    log_interval = 50  # batches - more frequent updates
    val_interval = 1  # epochs
    save_interval = 1  # save checkpoint every N epochs
    log_dir = Path("E:/5thsem el/kortex_5th_sem/sign_to_text/logs")
    
    # Sequence length limits
    max_src_len = 500  # Max frames (gestures)
    max_tgt_len = 100  # Max characters
    
    # Validation split
    val_ratio = 0.1
    seed = 42


# ==================== DATASET ====================

class SignLanguageDataset(Dataset):
    """
    Dataset for preprocessed landmark sequences
    """
    def __init__(self, data_dir, csv_path, char2idx, samples=None, 
                 max_src_len=500, max_tgt_len=100):
        self.data_dir = Path(data_dir)
        self.char2idx = char2idx
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        
        # Load CSV for labels
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        
        # Filter to only existing files (samples is set of uid strings)
        if samples is not None:
            self.df = self.df[self.df['uid'].isin(samples)]
        
        # Build sample list
        self.samples = []
        for _, row in self.df.iterrows():
            uid = str(row['uid'])
            # Support per-sample HDF5 (.h5) outputs, fall back to .npy
            h5_path = self.data_dir / f"{uid}.h5"
            npy_path = self.data_dir / f"{uid}.npy"
            if h5_path.exists():
                path = h5_path
            elif npy_path.exists():
                path = npy_path
            else:
                path = None

            if path is not None:
                self.samples.append({
                    'video_id': uid,
                    'path': path,
                    'label': str(row['text']).lower().strip()
                })
        
        print(f"Dataset: {len(self.samples)} samples loaded")
        
        # Special tokens
        self.sos_idx = char2idx.get('<sos>', 1)
        self.eos_idx = char2idx.get('<eos>', 2)
        self.blank_idx = char2idx.get('<blank>', 0)
        self.unk_idx = char2idx.get('<unk>', 4)
    
    def __len__(self):
        return len(self.samples)
    
    def encode_text(self, text):
        """Convert text to token indices with SOS/EOS"""
        tokens = [self.sos_idx]
        for char in text:
            idx = self.char2idx.get(char, self.unk_idx)
            tokens.append(idx)
        tokens.append(self.eos_idx)
        return tokens[:self.max_tgt_len]
    
    def encode_ctc(self, text):
        """Encode text for CTC (no SOS/EOS, just characters)"""
        tokens = []
        for char in text:
            idx = self.char2idx.get(char, self.unk_idx)
            tokens.append(idx)
        return tokens[:self.max_tgt_len - 2]  # Leave room
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load landmarks (support .h5 per-sample or .npy)
        path = Path(sample['path'])
        if path.suffix == '.h5':
            with h5py.File(path, 'r') as hf:
                # dataset key is 'features' written by the preprocessor
                landmarks = hf['features'][()]
        else:
            landmarks = np.load(path)
        
        # Truncate if too long
        if len(landmarks) > self.max_src_len:
            landmarks = landmarks[:self.max_src_len]
        
        # Encode text
        tgt_tokens = self.encode_text(sample['label'])
        ctc_tokens = self.encode_ctc(sample['label'])
        
        return {
            'video_id': sample['video_id'],
            'src': torch.FloatTensor(landmarks),
            'tgt': torch.LongTensor(tgt_tokens),
            'ctc_tgt': torch.LongTensor(ctc_tokens),
            'src_len': len(landmarks),
            'tgt_len': len(tgt_tokens),
            'ctc_len': len(ctc_tokens),
        }


def collate_fn(batch):
    """Collate with padding"""
    # Sort by source length (descending) for pack_padded_sequence
    batch = sorted(batch, key=lambda x: x['src_len'], reverse=True)
    
    # Get max lengths
    max_src_len = max(x['src_len'] for x in batch)
    max_tgt_len = max(x['tgt_len'] for x in batch)
    max_ctc_len = max(x['ctc_len'] for x in batch)
    
    batch_size = len(batch)
    
    # Initialize tensors
    src = torch.zeros(batch_size, max_src_len, batch[0]['src'].size(-1))
    tgt = torch.zeros(batch_size, max_tgt_len, dtype=torch.long)
    ctc_tgt = torch.zeros(batch_size, max_ctc_len, dtype=torch.long)
    
    src_lens = []
    tgt_lens = []
    ctc_lens = []
    video_ids = []
    
    for i, sample in enumerate(batch):
        src_len = sample['src_len']
        tgt_len = sample['tgt_len']
        ctc_len = sample['ctc_len']
        
        src[i, :src_len] = sample['src']
        tgt[i, :tgt_len] = sample['tgt']
        ctc_tgt[i, :ctc_len] = sample['ctc_tgt']
        
        src_lens.append(src_len)
        tgt_lens.append(tgt_len)
        ctc_lens.append(ctc_len)
        video_ids.append(sample['video_id'])
    
    return {
        'video_ids': video_ids,
        'src': src,
        'tgt': tgt,
        'ctc_tgt': ctc_tgt,
        'src_lens': torch.LongTensor(src_lens),
        'tgt_lens': torch.LongTensor(tgt_lens),
        'ctc_lens': torch.LongTensor(ctc_lens),
    }


# ==================== LOSS FUNCTIONS ====================

class LabelSmoothingLoss(nn.Module):
    """Cross entropy with label smoothing"""
    def __init__(self, vocab_size, smoothing=0.1, padding_idx=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        
    def forward(self, logits, targets):
        """
        Args:
            logits: (batch, seq_len, vocab_size)
            targets: (batch, seq_len)
        """
        logits = logits.contiguous().view(-1, self.vocab_size)
        targets = targets.contiguous().view(-1)
        
        # Create smoothed targets
        smooth_targets = torch.full_like(logits, self.smoothing / (self.vocab_size - 2))
        smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)
        smooth_targets[:, self.padding_idx] = 0
        
        # Mask padding
        mask = targets != self.padding_idx
        smooth_targets = smooth_targets[mask]
        logits = logits[mask]
        
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(smooth_targets * log_probs).sum(dim=-1).mean()
        
        return loss


class HybridLoss(nn.Module):
    """Combined CTC + Attention loss with dynamic weighting"""
    def __init__(self, vocab_size, blank_idx=0, smoothing=0.1, padding_idx=0):
        super().__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank_idx, reduction='mean', zero_infinity=True)
        self.attn_loss = LabelSmoothingLoss(vocab_size, smoothing, padding_idx)
        
    def forward(self, ctc_log_probs, attn_logits, ctc_targets, ctc_target_lens,
                attn_targets, src_lens, ctc_weight=0.3):
        """
        Args:
            ctc_log_probs: (batch, src_len, vocab)
            attn_logits: (batch, tgt_len, vocab)
            ctc_targets: (batch, ctc_len) CTC target tokens
            ctc_target_lens: (batch,) CTC target lengths
            attn_targets: (batch, tgt_len) Attention target tokens
            src_lens: (batch,) source sequence lengths
            ctc_weight: λ in loss = λ*CTC + (1-λ)*Attention
            
        Returns:
            total_loss, ctc_loss, attn_loss
        """
        # CTC Loss
        # CTC expects (time, batch, vocab)
        ctc_log_probs = ctc_log_probs.transpose(0, 1)
        
        # Flatten targets for CTC
        ctc_targets_flat = []
        for i in range(ctc_targets.size(0)):
            ctc_targets_flat.append(ctc_targets[i, :ctc_target_lens[i]])
        ctc_targets_flat = torch.cat(ctc_targets_flat)
        
        ctc_l = self.ctc_loss(ctc_log_probs, ctc_targets_flat, src_lens, ctc_target_lens)
        
        # Attention Loss (skip first SOS token in targets)
        attn_l = self.attn_loss(attn_logits[:, 1:], attn_targets[:, 1:])
        
        # Combined loss
        total = ctc_weight * ctc_l + (1 - ctc_weight) * attn_l
        
        return total, ctc_l, attn_l


# ==================== TRAINING ====================

class Trainer:
    """Training manager with all optimizations"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        
        # Create checkpoint directory
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Load vocabulary
        self.load_vocabulary()
        
        # Create model
        self.create_model()
        
        # Create datasets
        self.create_datasets()
        
        # Create optimizer
        self.create_optimizer()
        
        # Loss function
        self.criterion = HybridLoss(
            vocab_size=self.vocab_size,
            blank_idx=0,
            smoothing=config.label_smoothing,
            padding_idx=0
        ).to(self.device)
        
        # Mixed precision scaler
        self.scaler = torch.amp.GradScaler('cuda', enabled=config.use_amp)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # History
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_ctc_loss': [], 'val_ctc_loss': [],
            'train_attn_loss': [], 'val_attn_loss': [],
            'train_acc': [], 'val_acc': [],
            'ctc_weight': [], 'teacher_forcing': [],
            'learning_rate': []
        }
    
    def load_vocabulary(self):
        """Load or build vocabulary"""
        if self.config.vocab_path.exists():
            import json
            with open(self.config.vocab_path, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
            self.char2idx = vocab_data['char2idx']
            self.idx2char = {int(k): v for k, v in vocab_data['idx2char'].items()}
            self.vocab_size = len(self.char2idx)
            print(f"Loaded vocabulary: {self.vocab_size} tokens")
        else:
            # Build vocabulary if not exists
            print("Building vocabulary...")
            from vocabulary_builder import VocabularyBuilder
            builder = VocabularyBuilder()
            builder.build_from_csv(self.config.csv_path)
            builder.save(self.config.vocab_path)
            self.char2idx = builder.char2idx
            self.idx2char = builder.idx2char
            self.vocab_size = len(self.char2idx)
    
    def create_model(self):
        """Initialize model"""
        self.model = HybridCTCAttentionModel(
            input_dim=self.config.input_dim,
            hidden_dim=self.config.hidden_dim,
            vocab_size=self.vocab_size,
            embedding_dim=self.config.embedding_dim,
            encoder_layers=self.config.encoder_layers,
            decoder_layers=self.config.decoder_layers,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout
        ).to(self.device)
        
        num_params = self.model.get_num_params()
        model_size = self.model.get_model_size_mb()
        print(f"Model: {num_params:,} parameters, {model_size:.2f} MB")
    
    def create_datasets(self):
        """Create train/val datasets"""
        # Get all video IDs with preprocessed data
        all_files = list(self.config.data_dir.glob("*.npy"))
        all_ids = [f.stem for f in all_files]  # UIDs are strings
        
        # Split
        np.random.seed(self.config.seed)
        np.random.shuffle(all_ids)
        
        val_size = int(len(all_ids) * self.config.val_ratio)
        val_ids = set(all_ids[:val_size])
        train_ids = set(all_ids[val_size:])
        
        print(f"Train: {len(train_ids)}, Validation: {len(val_ids)}")
        
        # Create datasets - UIDs are strings, not ints
        self.train_dataset = SignLanguageDataset(
            self.config.data_dir, self.config.csv_path, self.char2idx,
            samples=train_ids,
            max_src_len=self.config.max_src_len,
            max_tgt_len=self.config.max_tgt_len
        )
        
        self.val_dataset = SignLanguageDataset(
            self.config.data_dir, self.config.csv_path, self.char2idx,
            samples=val_ids,
            max_src_len=self.config.max_src_len,
            max_tgt_len=self.config.max_tgt_len
        )
        
        # Create dataloaders with parallel loading
        worker_kwargs = {}
        if self.config.num_workers > 0:
            worker_kwargs = {
                'prefetch_factor': self.config.prefetch_factor,
                'persistent_workers': self.config.persistent_workers
            }
        
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True,
            **worker_kwargs
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size * 2,  # Larger for validation
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            **worker_kwargs
        )
    
    def create_optimizer(self):
        """Create optimizer and scheduler"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Cosine annealing with warm restarts
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # Restart every 10 epochs
            T_mult=2,  # Double period after each restart
            eta_min=self.config.min_lr
        )
    
    def get_ctc_weight(self, epoch):
        """Dynamic CTC weight with linear decay"""
        if epoch < self.config.ctc_weight_decay_epochs:
            decay = (self.config.ctc_weight_start - self.config.ctc_weight_end)
            progress = epoch / self.config.ctc_weight_decay_epochs
            return self.config.ctc_weight_start - decay * progress
        return self.config.ctc_weight_end
    
    def get_teacher_forcing(self, epoch):
        """Dynamic teacher forcing ratio with linear decay"""
        if epoch < self.config.tf_decay_epochs:
            decay = (self.config.tf_start - self.config.tf_end)
            progress = epoch / self.config.tf_decay_epochs
            return self.config.tf_start - decay * progress
        return self.config.tf_end
    
    def train_epoch(self, epoch):
        """Train for one epoch with tqdm progress bar"""
        self.model.train()
        
        total_loss = 0
        total_ctc_loss = 0
        total_attn_loss = 0
        correct = 0
        total_chars = 0
        
        ctc_weight = self.get_ctc_weight(epoch)
        tf_ratio = self.get_teacher_forcing(epoch)
        
        self.optimizer.zero_grad()
        
        batch_count = len(self.train_loader)
        
        # Create tqdm progress bar
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}",
            total=batch_count,
            ncols=120,
            leave=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            src = batch['src'].to(self.device)
            tgt = batch['tgt'].to(self.device)
            ctc_tgt = batch['ctc_tgt'].to(self.device)
            src_lens = batch['src_lens'].to(self.device)
            tgt_lens = batch['tgt_lens']
            ctc_lens = batch['ctc_lens']
            
            # Forward pass with mixed precision
            with autocast('cuda', enabled=self.config.use_amp):
                ctc_log_probs, attn_logits = self.model(
                    src, src_lens, tgt, tf_ratio
                )
                
                loss, ctc_l, attn_l = self.criterion(
                    ctc_log_probs, attn_logits, ctc_tgt, ctc_lens,
                    tgt, src_lens, ctc_weight
                )
                
                # Scale for gradient accumulation
                loss = loss / self.config.gradient_accumulation
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation == 0:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Accumulate metrics
            total_loss += loss.item() * self.config.gradient_accumulation
            total_ctc_loss += ctc_l.item()
            total_attn_loss += attn_l.item()
            
            # Character accuracy (attention output)
            with torch.no_grad():
                preds = attn_logits.argmax(dim=-1)
                mask = (tgt != 0)  # Ignore padding
                correct += ((preds == tgt) & mask).sum().item()
                total_chars += mask.sum().item()
            
            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            acc = correct / total_chars if total_chars > 0 else 0
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'ctc': f'{total_ctc_loss/(batch_idx+1):.3f}',
                'attn': f'{total_attn_loss/(batch_idx+1):.3f}',
                'acc': f'{acc*100:.1f}%'
            })
        
        pbar.close()
        
        # Epoch metrics
        avg_loss = total_loss / batch_count
        avg_ctc = total_ctc_loss / batch_count
        avg_attn = total_attn_loss / batch_count
        accuracy = correct / total_chars if total_chars > 0 else 0
        
        return avg_loss, avg_ctc, avg_attn, accuracy
    
    @torch.no_grad()
    def validate(self):
        """Run validation with progress bar"""
        self.model.eval()
        
        total_loss = 0
        total_ctc_loss = 0
        total_attn_loss = 0
        correct = 0
        total_chars = 0
        
        ctc_weight = self.get_ctc_weight(self.current_epoch)
        
        pbar = tqdm(
            self.val_loader,
            desc="Validating",
            total=len(self.val_loader),
            ncols=100,
            leave=False
        )
        
        for batch in pbar:
            src = batch['src'].to(self.device)
            tgt = batch['tgt'].to(self.device)
            ctc_tgt = batch['ctc_tgt'].to(self.device)
            src_lens = batch['src_lens'].to(self.device)
            ctc_lens = batch['ctc_lens']
            
            with autocast('cuda', enabled=self.config.use_amp):
                ctc_log_probs, attn_logits = self.model(
                    src, src_lens, tgt, teacher_forcing_ratio=0  # No teacher forcing in validation
                )
                
                loss, ctc_l, attn_l = self.criterion(
                    ctc_log_probs, attn_logits, ctc_tgt, ctc_lens,
                    tgt, src_lens, ctc_weight
                )
            
            total_loss += loss.item()
            total_ctc_loss += ctc_l.item()
            total_attn_loss += attn_l.item()
            
            # Character accuracy
            preds = attn_logits.argmax(dim=-1)
            mask = (tgt != 0)
            correct += ((preds == tgt) & mask).sum().item()
            total_chars += mask.sum().item()
            
            # Update progress bar
            acc = correct / total_chars if total_chars > 0 else 0
            pbar.set_postfix({'acc': f'{acc*100:.1f}%'})
        
        pbar.close()
        
        batch_count = len(self.val_loader)
        avg_loss = total_loss / batch_count
        avg_ctc = total_ctc_loss / batch_count
        avg_attn = total_attn_loss / batch_count
        accuracy = correct / total_chars if total_chars > 0 else 0
        
        return avg_loss, avg_ctc, avg_attn, accuracy
    
    def save_checkpoint(self, filename, is_best=False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': {
                'input_dim': self.config.input_dim,
                'hidden_dim': self.config.hidden_dim,
                'vocab_size': self.vocab_size,
                'encoder_layers': self.config.encoder_layers,
                'decoder_layers': self.config.decoder_layers,
            },
            'history': self.history,
            'char2idx': self.char2idx,
            'idx2char': self.idx2char,
        }
        
        path = self.config.checkpoint_dir / filename
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = self.config.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"  Saved best model: {best_path}")
    
    def load_checkpoint(self, path):
        """Load checkpoint to resume training"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        print(f"Resumed from epoch {self.current_epoch}")
    
    def train(self, resume_from=None):
        """Main training loop with proper logging"""
        if resume_from:
            self.load_checkpoint(resume_from)
        
        # Create log directory
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("="*60)
        logger.info(f"🚀 Starting training: {self.config.epochs} epochs")
        logger.info(f"💻 Device: {self.device}, AMP: {self.config.use_amp}")
        logger.info(f"📦 Batch size: {self.config.batch_size} x {self.config.gradient_accumulation} = "
              f"{self.config.batch_size * self.config.gradient_accumulation}")
        logger.info(f"🧠 Model: {self.model.get_num_params():,} params, {self.model.get_model_size_mb():.1f} MB")
        logger.info("="*60)
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            ctc_weight = self.get_ctc_weight(epoch)
            tf_ratio = self.get_teacher_forcing(epoch)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            logger.info(f"\n{'─'*60}")
            logger.info(f"📅 Epoch {epoch + 1}/{self.config.epochs} | LR: {current_lr:.2e} | "
                       f"CTC λ: {ctc_weight:.2f} | TF: {tf_ratio:.2f}")
            
            # Train
            train_loss, train_ctc, train_attn, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_ctc, val_attn, val_acc = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Log metrics
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_ctc_loss'].append(train_ctc)
            self.history['val_ctc_loss'].append(val_ctc)
            self.history['train_attn_loss'].append(train_attn)
            self.history['val_attn_loss'].append(val_attn)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['ctc_weight'].append(ctc_weight)
            self.history['teacher_forcing'].append(tf_ratio)
            self.history['learning_rate'].append(current_lr)
            
            epoch_time = time.time() - epoch_start
            
            # Pretty epoch summary with emojis
            logger.info(f"📊 Train | Loss: {train_loss:.4f} | CTC: {train_ctc:.4f} | "
                       f"Attn: {train_attn:.4f} | Acc: {train_acc*100:.2f}%")
            logger.info(f"📊 Valid | Loss: {val_loss:.4f} | CTC: {val_ctc:.4f} | "
                       f"Attn: {val_attn:.4f} | Acc: {val_acc*100:.2f}%")
            logger.info(f"⏱️  Epoch time: {epoch_time:.1f}s | ETA: {epoch_time * (self.config.epochs - epoch - 1) / 60:.1f} min")
            
            # Check for improvement
            is_best = val_loss < self.best_val_loss - self.config.min_delta
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                logger.info(f"🌟 New best model! Val loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1
                logger.info(f"⏳ No improvement. Patience: {self.patience_counter}/{self.config.patience}")
            
            # Save checkpoints every epoch for safety
            self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pth", is_best)
            
            # Also save latest for easy resume
            self.save_checkpoint("latest.pth", is_best=False)
            
            # Early stopping
            if self.patience_counter >= self.config.patience:
                logger.info(f"🛑 Early stopping at epoch {epoch + 1}")
                break
            
            # Save history after each epoch
            history_path = self.config.checkpoint_dir / "training_history.json"
            with open(history_path, 'w') as f:
                json.dump(self.history, f, indent=2)
        
        # Final save
        self.save_checkpoint("final_model.pth")
        
        total_time = time.time() - start_time
        logger.info("="*60)
        logger.info("🎉 Training complete!")
        logger.info(f"⏱️  Total time: {total_time/3600:.2f} hours")
        logger.info(f"📉 Best validation loss: {self.best_val_loss:.4f}")
        logger.info(f"🎯 Best validation accuracy: {max(self.history['val_acc'])*100:.2f}%")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Train Hybrid CTC-Attention Model")
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume')
    parser.add_argument('--epochs', type=int, default=None, help='Override number of epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size')
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate')
    args = parser.parse_args()
    
    config = TrainConfig()
    
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    
    trainer = Trainer(config)
    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()
