"""
Mobile-Optimized Training Script for Sign Language Recognition
Trains lightweight CNN-BiGRU model for real-time inference
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

from model_mobile import create_mobile_model
from vocabulary_builder import VocabularyBuilder


# ==================== CONFIGURATION ====================
class Config:
    # Paths
    csv_path = "../data/iSign_v1.1.csv"
    landmarks_dir = "E:/5thsem el/output"  # Change to your landmarks directory
    vocab_path = "vocabulary.pkl"
    output_dir = "checkpoints"
    
    # Model
    hidden_dim = 128  # 128 for speed, 256 for accuracy
    vocab_size = 5004  # Will be updated from vocabulary
    
    # Training
    batch_size = 16
    num_epochs = 40
    learning_rate = 1e-4
    weight_decay = 1e-5
    
    # Data
    max_frames = 150
    train_split = 0.8
    val_split = 0.1
    # test_split = 0.1 (implicit)
    
    # Augmentation
    frame_dropout_prob = 0.05
    noise_std = 0.01
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 4
    
    # Checkpointing
    save_every = 5  # epochs
    early_stopping_patience = 10


# ==================== DATASET ====================
class SignLanguageDataset(Dataset):
    """Dataset for sign language landmark sequences"""
    
    def __init__(self, df, landmarks_dir, vocab, max_frames=150, 
                 augment=False, frame_dropout=0.0, noise_std=0.0):
        self.df = df.reset_index(drop=True)
        self.landmarks_dir = Path(landmarks_dir)
        self.vocab = vocab
        self.max_frames = max_frames
        self.augment = augment
        self.frame_dropout = frame_dropout
        self.noise_std = noise_std
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        uid = row['uid']
        text = row['text']
        
        # Load landmarks
        landmark_file = self.landmarks_dir / f"{uid}.npy"
        if not landmark_file.exists():
            # Return dummy data if file missing
            return self._get_dummy_sample()
        
        landmarks = np.load(landmark_file).astype(np.float32)
        
        # Apply augmentation if training
        if self.augment:
            landmarks = self._augment(landmarks)
        
        # Get actual length before padding
        actual_length = min(len(landmarks), self.max_frames)
        
        # Pad or truncate
        if len(landmarks) < self.max_frames:
            pad_length = self.max_frames - len(landmarks)
            landmarks = np.pad(landmarks, ((0, pad_length), (0, 0)), mode='constant')
        else:
            landmarks = landmarks[:self.max_frames]
        
        # Convert text to indices
        target_indices = self.vocab.text_to_word_indices(text)
        
        return {
            'landmarks': torch.FloatTensor(landmarks),
            'target': torch.LongTensor(target_indices),
            'length': actual_length,
            'target_length': len(target_indices),
            'text': text
        }
    
    def _augment(self, landmarks):
        """Apply data augmentation"""
        # Random frame dropout
        if self.frame_dropout > 0 and np.random.rand() < self.frame_dropout:
            keep_prob = 1.0 - self.frame_dropout
            num_frames = len(landmarks)
            keep_indices = np.random.rand(num_frames) < keep_prob
            if keep_indices.sum() > 0:  # Ensure at least some frames remain
                landmarks = landmarks[keep_indices]
        
        # Add noise to landmarks
        if self.noise_std > 0:
            noise = np.random.randn(*landmarks.shape) * self.noise_std
            landmarks = landmarks + noise
        
        return landmarks
    
    def _get_dummy_sample(self):
        """Return dummy sample for missing files"""
        return {
            'landmarks': torch.zeros(self.max_frames, 138),
            'target': torch.LongTensor([self.vocab.word2idx['<unk>']]),
            'length': 1,
            'target_length': 1,
            'text': '<unk>'
        }


def collate_fn(batch):
    """Custom collate function for DataLoader"""
    landmarks = torch.stack([b['landmarks'] for b in batch])
    lengths = torch.LongTensor([b['length'] for b in batch])
    
    # Targets need to be concatenated for CTC loss
    targets = []
    target_lengths = []
    for b in batch:
        targets.extend(b['target'].tolist())
        target_lengths.append(b['target_length'])
    
    targets = torch.LongTensor(targets)
    target_lengths = torch.LongTensor(target_lengths)
    texts = [b['text'] for b in batch]
    
    return landmarks, targets, lengths, target_lengths, texts


# ==================== TRAINING ====================
def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(loader, desc="Training")
    for landmarks, targets, lengths, target_lengths, texts in pbar:
        landmarks = landmarks.to(device)
        targets = targets.to(device)
        lengths = lengths.to(device)
        target_lengths = target_lengths.to(device)
        
        # Forward pass
        log_probs = model(landmarks, lengths)  # (batch, frames, vocab)
        
        # CTC expects (frames, batch, vocab)
        log_probs = log_probs.permute(1, 0, 2)
        
        # CTC Loss
        loss = criterion(log_probs, targets, lengths, target_lengths)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches


def validate(model, loader, criterion, device, vocab):
    """Validate model"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    # For calculating accuracy
    correct_chars = 0
    total_chars = 0
    
    with torch.no_grad():
        for landmarks, targets, lengths, target_lengths, texts in tqdm(loader, desc="Validating"):
            landmarks = landmarks.to(device)
            targets = targets.to(device)
            lengths = lengths.to(device)
            target_lengths = target_lengths.to(device)
            
            # Forward pass
            log_probs = model(landmarks, lengths)
            
            # CTC Loss
            log_probs_ctc = log_probs.permute(1, 0, 2)
            loss = criterion(log_probs_ctc, targets, lengths, target_lengths)
            total_loss += loss.item()
            num_batches += 1
            
            # Calculate accuracy (character-level)
            preds = torch.argmax(log_probs, dim=-1)  # (batch, frames)
            
            # Decode predictions (simple greedy)
            for i in range(len(preds)):
                pred_indices = preds[i].cpu().numpy()
                
                # Skip if text is NaN or not a string
                if not isinstance(texts[i], str):
                    continue
                    
                target_text = texts[i].lower()
                
                # Simple CTC decode (remove blanks and duplicates)
                decoded = []
                prev = -1
                for idx in pred_indices:
                    if idx != 0 and idx != prev:  # 0 is blank
                        decoded.append(idx)
                    prev = idx
                
                pred_text = vocab.indices_to_text(decoded, mode='word')
                
                # Character-level accuracy
                for c1, c2 in zip(pred_text, target_text):
                    if c1 == c2:
                        correct_chars += 1
                    total_chars += 1
    
    avg_loss = total_loss / num_batches
    char_accuracy = (correct_chars / total_chars * 100) if total_chars > 0 else 0
    
    return avg_loss, char_accuracy


# ==================== MAIN ====================
def main():
    config = Config()
    
    print("="*60)
    print("MOBILE-OPTIMIZED SIGN LANGUAGE TRAINING")
    print("="*60)
    print(f"Device: {config.device}")
    print(f"Batch size: {config.batch_size}")
    print(f"Hidden dim: {config.hidden_dim}")
    print(f"Epochs: {config.num_epochs}")
    print("="*60)
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Load or build vocabulary
    if os.path.exists(config.vocab_path):
        print(f"\nLoading vocabulary from {config.vocab_path}...")
        vocab = VocabularyBuilder.load(config.vocab_path)
    else:
        print(f"\nBuilding vocabulary...")
        vocab = VocabularyBuilder(config.csv_path, max_words=5000)
        vocab.build_from_csv()
        vocab.save(config.vocab_path)
    
    config.vocab_size = len(vocab.word2idx)
    print(f"Vocabulary size: {config.vocab_size}")
    
    # Load dataset
    print(f"\nLoading dataset from {config.csv_path}...")
    df = pd.read_csv(config.csv_path)
    
    # Filter out rows with NaN text
    df = df.dropna(subset=['text', 'uid'])
    df = df[df['text'].str.strip() != '']
    print(f"Total valid samples: {len(df)}")
    
    # Split data
    indices = np.random.permutation(len(df))
    train_size = int(len(df) * config.train_split)
    val_size = int(len(df) * config.val_split)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    test_df = df.iloc[test_idx]
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Create datasets
    train_dataset = SignLanguageDataset(
        train_df, config.landmarks_dir, vocab, config.max_frames,
        augment=True, frame_dropout=config.frame_dropout_prob, noise_std=config.noise_std
    )
    val_dataset = SignLanguageDataset(
        val_df, config.landmarks_dir, vocab, config.max_frames,
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=config.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=config.num_workers, pin_memory=True
    )
    
    # Create model
    print(f"\nCreating model...")
    model = create_mobile_model(vocab_size=config.vocab_size, hidden_dim=config.hidden_dim)
    model = model.to(config.device)
    
    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, 
                           weight_decay=config.weight_decay)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'learning_rate': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config.device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, config.device, vocab)
        
        # Update scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Char Accuracy: {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"  Time: {epoch_time:.2f}s")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        history['learning_rate'].append(current_lr)
        
        # Save checkpoint
        if (epoch + 1) % config.save_every == 0:
            checkpoint_path = os.path.join(config.output_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
            }, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_path = os.path.join(config.output_dir, "best_model.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'vocab_size': config.vocab_size,
                'hidden_dim': config.hidden_dim,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
            }, best_model_path)
            print(f"  ✅ Best model saved! Val Loss: {val_loss:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Save final model
    final_model_path = os.path.join(config.output_dir, "final_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': config.vocab_size,
        'hidden_dim': config.hidden_dim,
    }, final_model_path)
    print(f"\n✅ Final model saved: {final_model_path}")
    
    # Save history
    history_path = os.path.join(config.output_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✅ Training history saved: {history_path}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)


if __name__ == "__main__":
    main()
