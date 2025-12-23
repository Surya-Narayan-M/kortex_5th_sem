"""
Training Script for Seq2Seq Sign Language Recognition
Encoder-Decoder with Attention for 90%+ accuracy on sentence-level prediction
"""

import os
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model_seq2seq import create_seq2seq_model
from vocabulary_builder import VocabularyBuilder


# ==================== CONFIGURATION ====================
class Config:
    # Paths
    csv_path = "../data/iSign_v1.1.csv"
    landmarks_dir = "E:/5thsem el/output"
    vocab_path = "vocabulary.pkl"
    output_dir = "checkpoints_seq2seq"
    
    # Model
    hidden_dim = 512  # Larger for 90%+ accuracy
    embedding_dim = 256
    encoder_layers = 3  # Deep encoder
    decoder_layers = 2
    dropout = 0.3
    
    # Training
    batch_size = 16
    num_epochs = 100  # More epochs for convergence
    learning_rate = 1e-3
    weight_decay = 1e-5
    teacher_forcing_ratio_start = 1.0  # Start with full teacher forcing
    teacher_forcing_ratio_end = 0.5  # Gradually reduce
    
    # Data
    max_frames = 150
    max_text_length = 100  # Maximum characters in output
    train_split = 0.8
    val_split = 0.1
    
    # Augmentation
    frame_dropout_prob = 0.05
    noise_std = 0.01
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 4
    
    # Checkpointing
    save_every = 5
    early_stopping_patience = 15


# ==================== DATASET ====================
class Seq2SeqSignLanguageDataset(Dataset):
    """Dataset for Seq2Seq sign language recognition"""
    
    def __init__(self, df, landmarks_dir, vocab, max_frames=150, max_text_length=100,
                 augment=False, frame_dropout=0.0, noise_std=0.0):
        self.df = df.reset_index(drop=True)
        self.landmarks_dir = Path(landmarks_dir)
        self.vocab = vocab
        self.max_frames = max_frames
        self.max_text_length = max_text_length
        self.augment = augment
        self.frame_dropout = frame_dropout
        self.noise_std = noise_std
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        uid = row['uid']
        text = row['text'].lower().strip()  # Lowercase and clean
        
        # Load landmarks
        landmark_file = self.landmarks_dir / f"{uid}.npy"
        if not landmark_file.exists():
            return self._get_dummy_sample()
        
        landmarks = np.load(landmark_file).astype(np.float32)
        
        # Apply augmentation
        if self.augment:
            landmarks = self._augment(landmarks)
        
        # Get actual length
        actual_length = min(len(landmarks), self.max_frames)
        
        # Pad or truncate landmarks
        if len(landmarks) < self.max_frames:
            pad_length = self.max_frames - len(landmarks)
            landmarks = np.pad(landmarks, ((0, pad_length), (0, 0)), mode='constant')
        else:
            landmarks = landmarks[:self.max_frames]
        
        # Convert text to character indices with special tokens
        # Format: [SOS, char1, char2, ..., EOS, PAD, PAD, ...]
        char_indices = [1]  # SOS token
        for char in text[:self.max_text_length-2]:  # Reserve space for SOS and EOS
            idx = self.vocab.char2idx.get(char, self.vocab.char2idx['<unk>'])
            char_indices.append(idx)
        char_indices.append(2)  # EOS token
        
        # Pad to max length
        text_length = len(char_indices)
        if text_length < self.max_text_length:
            char_indices.extend([0] * (self.max_text_length - text_length))  # PAD token
        else:
            char_indices = char_indices[:self.max_text_length]
            text_length = self.max_text_length
        
        return {
            'landmarks': torch.FloatTensor(landmarks),
            'text': torch.LongTensor(char_indices),
            'src_length': actual_length,
            'trg_length': text_length,
            'original_text': text
        }
    
    def _augment(self, landmarks):
        """Apply data augmentation"""
        # Random frame dropout
        if self.frame_dropout > 0 and np.random.random() < self.frame_dropout:
            keep_mask = np.random.random(len(landmarks)) > self.frame_dropout
            if keep_mask.any():
                landmarks = landmarks[keep_mask]
        
        # Add Gaussian noise
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, landmarks.shape)
            landmarks = landmarks + noise
        
        return landmarks
    
    def _get_dummy_sample(self):
        """Return dummy sample for missing data"""
        return {
            'landmarks': torch.zeros(self.max_frames, 138),
            'text': torch.LongTensor([1, 2] + [0] * (self.max_text_length - 2)),  # SOS, EOS, PAD
            'src_length': 1,
            'trg_length': 2,
            'original_text': ""
        }


def collate_fn(batch):
    """Custom collate function"""
    landmarks = torch.stack([item['landmarks'] for item in batch])
    texts = torch.stack([item['text'] for item in batch])
    src_lengths = torch.LongTensor([item['src_length'] for item in batch])
    trg_lengths = torch.LongTensor([item['trg_length'] for item in batch])
    original_texts = [item['original_text'] for item in batch]
    
    return {
        'landmarks': landmarks,
        'texts': texts,
        'src_lengths': src_lengths,
        'trg_lengths': trg_lengths,
        'original_texts': original_texts
    }


# ==================== TRAINING ====================
def train_epoch(model, dataloader, optimizer, criterion, config, epoch, total_epochs):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    # Calculate teacher forcing ratio (decay over epochs)
    progress = epoch / total_epochs
    teacher_forcing_ratio = config.teacher_forcing_ratio_start * (1 - progress) + \
                           config.teacher_forcing_ratio_end * progress
    
    pbar = tqdm(dataloader, desc=f"Training Epoch {epoch}/{total_epochs}")
    for batch in pbar:
        landmarks = batch['landmarks'].to(config.device)
        texts = batch['texts'].to(config.device)
        src_lengths = batch['src_lengths']
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs, _ = model(landmarks, src_lengths, texts, teacher_forcing_ratio)
        
        # Compute loss (ignore first token which is always SOS)
        # outputs: (batch, trg_len, vocab_size)
        # texts: (batch, trg_len)
        outputs = outputs[:, 1:].reshape(-1, outputs.shape[-1])  # (batch*(trg_len-1), vocab_size)
        texts = texts[:, 1:].reshape(-1)  # (batch*(trg_len-1))
        
        loss = criterion(outputs, texts)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'tf_ratio': f'{teacher_forcing_ratio:.2f}'})
    
    return total_loss / num_batches


def validate(model, dataloader, criterion, vocab, config):
    """Validate model"""
    model.eval()
    total_loss = 0
    correct_chars = 0
    total_chars = 0
    correct_sequences = 0
    total_sequences = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            landmarks = batch['landmarks'].to(config.device)
            texts = batch['texts'].to(config.device)
            src_lengths = batch['src_lengths']
            original_texts = batch['original_texts']
            
            # Forward pass with teacher forcing for loss
            outputs, _ = model(landmarks, src_lengths, texts, teacher_forcing_ratio=1.0)
            
            # Compute loss
            loss_outputs = outputs[:, 1:].reshape(-1, outputs.shape[-1])
            loss_texts = texts[:, 1:].reshape(-1)
            loss = criterion(loss_outputs, loss_texts)
            total_loss += loss.item()
            
            # Predict without teacher forcing for accuracy
            predictions, _ = model.predict(landmarks, src_lengths, max_len=config.max_text_length)
            
            # Calculate accuracy
            for i in range(len(predictions)):
                pred_chars = predictions[i].cpu().numpy()
                true_chars = texts[i].cpu().numpy()
                
                # Remove special tokens and padding
                pred_chars = pred_chars[(pred_chars != 0) & (pred_chars != 1) & (pred_chars != 2)]
                true_chars = true_chars[(true_chars != 0) & (true_chars != 1) & (true_chars != 2)]
                
                # Convert to text
                pred_text = ''.join([vocab.idx2char[idx] if idx in vocab.idx2char else '' 
                                    for idx in pred_chars])
                true_text = original_texts[i]
                
                # Character-level accuracy
                min_len = min(len(pred_text), len(true_text))
                if min_len > 0:
                    correct_chars += sum(p == t for p, t in zip(pred_text[:min_len], true_text[:min_len]))
                    total_chars += max(len(pred_text), len(true_text))
                
                # Sequence-level accuracy
                if pred_text.strip() == true_text.strip():
                    correct_sequences += 1
                total_sequences += 1
    
    avg_loss = total_loss / len(dataloader)
    char_accuracy = 100 * correct_chars / max(total_chars, 1)
    seq_accuracy = 100 * correct_sequences / max(total_sequences, 1)
    
    return avg_loss, char_accuracy, seq_accuracy


def main():
    config = Config()
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    print("="*60)
    print("Seq2Seq Sign Language Recognition Training")
    print("="*60)
    print(f"Device: {config.device}")
    print(f"Hidden dim: {config.hidden_dim}")
    print(f"Encoder layers: {config.encoder_layers}")
    print(f"Decoder layers: {config.decoder_layers}")
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Learning rate: {config.learning_rate}")
    
    # Load vocabulary
    print("\nLoading vocabulary...")
    vocab = VocabularyBuilder.load(config.vocab_path)
    config.vocab_size = len(vocab.char2idx)  # Use character vocabulary
    print(f"Character vocabulary size: {config.vocab_size}")
    
    # Load dataset
    print("\nLoading dataset...")
    df = pd.read_csv(config.csv_path)
    df = df.dropna(subset=['text', 'uid'])
    df = df[df['text'].str.strip() != '']
    print(f"Total samples: {len(df)}")
    
    # Split dataset
    train_size = int(config.train_split * len(df))
    val_size = int(config.val_split * len(df))
    
    train_df = df[:train_size]
    val_df = df[train_size:train_size + val_size]
    
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    
    # Create datasets
    train_dataset = Seq2SeqSignLanguageDataset(
        train_df, config.landmarks_dir, vocab, config.max_frames, config.max_text_length,
        augment=True, frame_dropout=config.frame_dropout_prob, noise_std=config.noise_std
    )
    
    val_dataset = Seq2SeqSignLanguageDataset(
        val_df, config.landmarks_dir, vocab, config.max_frames, config.max_text_length,
        augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=config.num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=config.num_workers, pin_memory=True
    )
    
    # Create model
    print("\nCreating Seq2Seq model...")
    model = create_seq2seq_model(
        vocab_size=config.vocab_size,
        hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim
    )
    model = model.to(config.device)
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'char_accuracy': [],
        'seq_accuracy': [],
        'learning_rate': []
    }
    
    best_val_loss = float('inf')
    best_char_acc = 0
    patience_counter = 0
    
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    for epoch in range(1, config.num_epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config, epoch, config.num_epochs)
        
        # Validate
        val_loss, char_acc, seq_acc = validate(model, val_loader, criterion, vocab, config)
        
        # Update scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['char_accuracy'].append(char_acc)
        history['seq_accuracy'].append(seq_acc)
        history['learning_rate'].append(current_lr)
        
        epoch_time = time.time() - epoch_start
        
        # Print progress
        print(f"\nEpoch {epoch}/{config.num_epochs} - {epoch_time:.1f}s")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Char Accuracy: {char_acc:.2f}%")
        print(f"Seq Accuracy: {seq_acc:.2f}%")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Save checkpoint
        if epoch % config.save_every == 0:
            checkpoint_path = Path(config.output_dir) / f"checkpoint_epoch_{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'char_accuracy': char_acc,
                'seq_accuracy': seq_acc,
                'config': config.__dict__
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if char_acc > best_char_acc:
            best_char_acc = char_acc
            best_val_loss = val_loss
            patience_counter = 0
            
            best_model_path = Path(config.output_dir) / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'char_accuracy': char_acc,
                'seq_accuracy': seq_acc,
                'val_loss': val_loss,
                'config': config.__dict__
            }, best_model_path)
            print(f"✓ Best model saved (Char Acc: {char_acc:.2f}%, Seq Acc: {seq_acc:.2f}%)")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break
        
        # Save history
        history_path = Path(config.output_dir) / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
    
    # Save final model
    final_model_path = Path(config.output_dir) / "final_model.pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'config': config.__dict__
    }, final_model_path)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best Character Accuracy: {best_char_acc:.2f}%")
    print(f"Best Validation Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
