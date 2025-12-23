"""Test model loading to diagnose issues"""
import torch
import numpy as np
from model_mobile import create_mobile_model
from vocabulary_builder import VocabularyBuilder

print("="*60)
print("TESTING MODEL LOADING")
print("="*60)

# Load vocabulary
print("\n1. Loading vocabulary...")
vocab = VocabularyBuilder.load('vocabulary.pkl')
print(f"   ✅ Vocab size: {len(vocab.word2idx)}")

# Load checkpoint
print("\n2. Loading checkpoint...")
checkpoint = torch.load('checkpoints/best_model.pth', map_location='cpu')
print(f"   Checkpoint keys: {list(checkpoint.keys())}")
print(f"   Epoch: {checkpoint['epoch']}")
print(f"   Val Loss: {checkpoint['val_loss']:.4f}")

# Create model
print("\n3. Creating model...")
model = create_mobile_model(vocab_size=len(vocab.word2idx))
print(f"   ✅ Model created")

# Load weights
print("\n4. Loading model weights...")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"   ✅ Weights loaded")

# Test forward pass
print("\n5. Testing forward pass...")
dummy_input = torch.randn(1, 60, 138)  # (batch=1, frames=60, features=138)
with torch.no_grad():
    output = model(dummy_input)
print(f"   Input shape: {dummy_input.shape}")
print(f"   Output shape: {output.shape}")
print(f"   Expected output shape: (1, 60, {len(vocab.word2idx)})")

if output.shape == (1, 60, len(vocab.word2idx)):
    print("\n✅ MODEL IS WORKING CORRECTLY!")
else:
    print("\n❌ OUTPUT SHAPE MISMATCH!")

# Test with actual prediction
print("\n6. Testing prediction...")
pred_indices = torch.argmax(output, dim=-1)[0].cpu().numpy()
print(f"   Raw predictions (first 20): {pred_indices[:20]}")

# Decode
decoded = []
prev = -1
for idx in pred_indices:
    if idx != 0 and idx != prev:  # 0 is blank
        if idx < len(vocab.idx2word):
            decoded.append(vocab.idx2word[idx])
    prev = idx
    
print(f"   Decoded words: {' '.join(decoded[:10])}...")

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
