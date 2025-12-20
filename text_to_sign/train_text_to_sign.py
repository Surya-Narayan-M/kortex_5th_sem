"""
Training Script for Text-to-Sign Generation Model
Trains Transformer to generate landmark sequences from text
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm
import sys
import os

# Add parent directory for vocabulary import
sys.path.append(str(Path(__file__).parent.parent / 'sign_to_text'))
from vocabulary_builder import VocabularyBuilder
from model_text_to_sign import create_text_to_sign_model


class TextToSignDataset(Dataset):
    """Dataset for text-to-sign generation"""
    
    def __init__(self, csv_path, landmarks_dir, vocab_builder, max_frames=150, split='train'):
        self.landmarks_dir = Path(landmarks_dir)
        self.vocab_builder = vocab_builder
        self.max_frames = max_frames
        
        # Load CSV
        df = pd.read_csv(csv_path, encoding='latin-1')
        
        # Split data (80/10/10)
        n = len(df)
        if split == 'train':
            self.data = df.iloc[:int(0.8*n)]
        elif split == 'val':
            self.data = df.iloc[int(0.8*n):int(0.9*n)]
        else:  # test
            self.data = df.iloc[int(0.9*n):]
        
        # Filter samples where landmark files exist
        self.samples = []
        for idx, row in self.data.iterrows():
            video_id = row['videoid']
            npy_path = self.landmarks_dir / f"{video_id}.npy"
            if npy_path.exists():
                self.samples.append({
                    'video_id': video_id,
                    'text': row['sentence'].strip(),
                    'npy_path': npy_path
                })
        
        print(f"{split.upper()} dataset: {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load landmarks
        landmarks = np.load(sample['npy_path'])  # (frames, 138)
        
        # Pad/truncate to max_frames
        num_frames = landmarks.shape[0]
        if num_frames < self.max_frames:
            # Pad with last frame
            padding = np.repeat(landmarks[-1:], self.max_frames - num_frames, axis=0)
            landmarks = np.vstack([landmarks, padding])
        else:
            landmarks = landmarks[:self.max_frames]
        
        # Convert text to tokens
        text_tokens = self.vocab_builder.text_to_word_indices(sample['text'])
        
        return {
            'text_tokens': torch.tensor(text_tokens, dtype=torch.long),
            'landmarks': torch.tensor(landmarks, dtype=torch.float32),
            'num_frames': min(num_frames, self.max_frames)
        }


def collate_fn(batch):
    """Custom collate function to handle variable-length sequences"""
    # Find max lengths
    max_text_len = max(item['text_tokens'].size(0) for item in batch)
    max_frames = max(item['num_frames'] for item in batch)
    
    # Pad text tokens
    text_tokens = []
    for item in batch:
        tokens = item['text_tokens']
        pad_len = max_text_len - tokens.size(0)
        if pad_len > 0:
            tokens = torch.cat([tokens, torch.zeros(pad_len, dtype=torch.long)])
        text_tokens.append(tokens)
    
    text_tokens = torch.stack(text_tokens)
    
    # Landmarks are already padded to max_frames in dataset
    landmarks = torch.stack([item['landmarks'] for item in batch])
    
    # Actual frame counts
    num_frames = torch.tensor([item['num_frames'] for item in batch])
    
    return {
        'text_tokens': text_tokens,
        'landmarks': landmarks,
        'num_frames': num_frames
    }


class TemporalSmoothingLoss(nn.Module):
    """
    Encourages smooth transitions between frames
    Penalizes large changes in consecutive frames
    """
    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight
    
    def forward(self, landmarks, num_frames):
        """
        Args:
            landmarks: (batch, frames, 138)
            num_frames: (batch,) - actual number of frames
        """
        # Compute frame-to-frame differences
        diff = landmarks[:, 1:, :] - landmarks[:, :-1, :]  # (batch, frames-1, 138)
        
        # L2 norm of differences
        smoothness_loss = torch.mean(torch.norm(diff, dim=2))
        
        return self.weight * smoothness_loss


def train_epoch(model, dataloader, optimizer, device, epoch, use_amp=False):
    """Train for one epoch"""
    model.train()
    
    mse_loss_fn = nn.MSELoss()
    smoothing_loss_fn = TemporalSmoothingLoss(weight=0.1)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    total_loss = 0
    total_mse = 0
    total_smooth = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        text_tokens = batch['text_tokens'].to(device)
        landmarks = batch['landmarks'].to(device)
        num_frames = batch['num_frames'].to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        if use_amp:
            with torch.cuda.amp.autocast():
                predicted_landmarks = model(text_tokens, landmarks, teacher_forcing_ratio=0.5)
                mse_loss = mse_loss_fn(predicted_landmarks, landmarks)
                smooth_loss = smoothing_loss_fn(predicted_landmarks, num_frames)
                loss = mse_loss + smooth_loss
            
            # Backward with gradient scaling
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training
            predicted_landmarks = model(text_tokens, landmarks, teacher_forcing_ratio=0.5)
            mse_loss = mse_loss_fn(predicted_landmarks, landmarks)
            smooth_loss = smoothing_loss_fn(predicted_landmarks, num_frames)
            loss = mse_loss + smooth_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        total_mse += mse_loss.item()
        total_smooth += smooth_loss.item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'mse': f'{mse_loss.item():.4f}',
            'smooth': f'{smooth_loss.item():.4f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_mse = total_mse / len(dataloader)
    avg_smooth = total_smooth / len(dataloader)
    
    return avg_loss, avg_mse, avg_smooth


def validate(model, dataloader, device):
    """Validate model"""
    model.eval()
    
    mse_loss_fn = nn.MSELoss()
    smoothing_loss_fn = TemporalSmoothingLoss(weight=0.1)
    
    total_loss = 0
    total_mse = 0
    total_smooth = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            text_tokens = batch['text_tokens'].to(device)
            landmarks = batch['landmarks'].to(device)
            num_frames = batch['num_frames'].to(device)
            
            # Generate landmarks (no teacher forcing)
            predicted_landmarks = model.generate(text_tokens, max_frames=150)
            
            # Ensure same shape
            min_frames = min(predicted_landmarks.size(1), landmarks.size(1))
            predicted_landmarks = predicted_landmarks[:, :min_frames, :]
            landmarks = landmarks[:, :min_frames, :]
            
            # Losses
            mse_loss = mse_loss_fn(predicted_landmarks, landmarks)
            smooth_loss = smoothing_loss_fn(predicted_landmarks, num_frames)
            loss = mse_loss + smooth_loss
            
            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_smooth += smooth_loss.item()
    
    avg_loss = total_loss / len(dataloader)
    avg_mse = total_mse / len(dataloader)
    avg_smooth = total_smooth / len(dataloader)
    
    return avg_loss, avg_mse, avg_smooth


class Config:
    """Training configuration"""
    # Paths
    CSV_PATH = "../data/iSign_v1.1.csv"
    LANDMARKS_DIR = "E:/5thsem el/output"  # Change this to your landmarks directory
    VOCAB_PATH = "../sign_to_text/vocabulary.pkl"
    MODEL_SAVE_DIR = "checkpoints"
    
    # Model
    VOCAB_SIZE = 5004
    HIDDEN_DIM = 256
    NUM_LAYERS = 4
    MAX_FRAMES = 150
    
    # Training
    BATCH_SIZE = 20  # Optimized for RTX 4060 (8GB VRAM)
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    PATIENCE = 8  # Early stopping
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    USE_AMP = True  # Mixed precision training for faster speed


def main():
    print("Text-to-Sign Training Pipeline")
    print("="*60)
    
    config = Config()
    
    # Create model save directory
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    
    # Load vocabulary
    print("\nLoading vocabulary...")
    if not Path(config.VOCAB_PATH).exists():
        print("Vocabulary not found. Please run vocabulary_builder.py first!")
        return
    
    vocab_builder = VocabularyBuilder.load(config.VOCAB_PATH)
    print(f"Vocabulary size: {len(vocab_builder.word_to_idx)}")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = TextToSignDataset(
        config.CSV_PATH, config.LANDMARKS_DIR, vocab_builder, 
        max_frames=config.MAX_FRAMES, split='train'
    )
    val_dataset = TextToSignDataset(
        config.CSV_PATH, config.LANDMARKS_DIR, vocab_builder,
        max_frames=config.MAX_FRAMES, split='val'
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, 
        shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True
    )
    
    # Create model
    print("\nCreating model...")
    model = create_text_to_sign_model(
        vocab_size=config.VOCAB_SIZE,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS
    )
    model = model.to(config.DEVICE)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Training loop
    print(f"\nTraining on {config.DEVICE}...")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print("="*60)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mse': [],
        'val_mse': []
    }
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        # Train
        train_loss, train_mse, train_smooth = train_epoch(
            model, train_loader, optimizer, config.DEVICE, epoch, use_amp=config.USE_AMP
        )
        
        # Validate
        val_loss, val_mse, val_smooth = validate(model, val_loader, config.DEVICE)
        
        # Log
        print(f"\nEpoch {epoch}/{config.NUM_EPOCHS}")
        print(f"Train - Loss: {train_loss:.4f}, MSE: {train_mse:.4f}, Smooth: {train_smooth:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, MSE: {val_mse:.4f}, Smooth: {val_smooth:.4f}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mse'].append(train_mse)
        history['val_mse'].append(val_mse)
        
        # Learning rate schedule
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            save_path = Path(config.MODEL_SAVE_DIR) / "best_text_to_sign.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': {
                    'vocab_size': config.VOCAB_SIZE,
                    'hidden_dim': config.HIDDEN_DIM,
                    'num_layers': config.NUM_LAYERS,
                    'max_frames': config.MAX_FRAMES
                }
            }, save_path)
            print(f"✅ Best model saved (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.PATIENCE:
            print(f"\n⏹️ Early stopping triggered (patience: {config.PATIENCE})")
            break
        
        print("-"*60)
    
    # Save training history
    history_path = Path(config.MODEL_SAVE_DIR) / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n✅ Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {config.MODEL_SAVE_DIR}/best_text_to_sign.pth")
    print(f"History saved to: {history_path}")


if __name__ == "__main__":
    main()
