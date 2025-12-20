"""
Mobile-Optimized Sign Language Recognition Model
Lightweight CNN-BiGRU architecture for real-time inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MobileSignLanguageModel(nn.Module):
    """
    Optimized for mobile deployment with streaming inference
    
    Architecture:
    - 1D CNN for local feature extraction
    - BiGRU for temporal modeling (faster than LSTM, fewer parameters)
    - CTC output for sequence prediction
    
    Target: <2MB model, <100ms inference on mid-range phones
    """
    
    def __init__(self, input_dim=138, hidden_dim=128, vocab_size=5004, num_layers=1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        # 1D CNN Feature Extractor (replaces attention, much faster on mobile)
        self.cnn = nn.Sequential(
            # First conv block
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
            # Second conv block
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
            # Optional pooling for speed (reduces temporal resolution)
            # nn.MaxPool1d(kernel_size=2, stride=2)  # Uncomment for 2x speed
        )
        
        # BiGRU Encoder (lightweight: 1 layer, faster than LSTM)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,  # Divided by 2 because bidirectional
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0  # No dropout for single layer
        )
        
        # CTC Output Layer
        self.ctc_head = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, lengths=None):
        """
        Forward pass
        
        Args:
            x: (batch, frames, features) - landmark sequences
            lengths: (batch,) - actual sequence lengths
        
        Returns:
            log_probs: (batch, frames, vocab_size) - CTC predictions
        """
        batch_size, frames, features = x.shape
        
        # CNN expects (batch, channels, time)
        x = x.permute(0, 2, 1)  # (batch, features, frames)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # (batch, frames, hidden_dim)
        
        # BiGRU
        if lengths is not None:
            # Pack for variable length sequences
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            x, _ = self.gru(x)
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        else:
            x, _ = self.gru(x)
        
        # CTC output
        logits = self.ctc_head(x)  # (batch, frames, vocab_size)
        log_probs = F.log_softmax(logits, dim=-1)
        
        return log_probs
    
    def get_num_params(self):
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_model_size_mb(self):
        """Estimate model size in MB"""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        size_mb = (param_size + buffer_size) / (1024 ** 2)
        return size_mb


class StreamingInference:
    """
    Handles streaming inference with sliding window
    Maintains frame buffer and produces real-time predictions
    """
    
    def __init__(self, model, vocab, window_size=60, stride=10, 
                 confidence_threshold=0.7, device='cpu'):
        self.model = model
        self.vocab = vocab
        self.window_size = window_size
        self.stride = stride
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        self.model.eval()
        self.model.to(device)
        
        # Frame buffer (circular)
        self.buffer = []
        self.predictions_history = []
        
    def add_frame(self, landmarks):
        """Add new frame to buffer"""
        self.buffer.append(landmarks)
        
        # Keep only window_size frames
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
    
    def should_predict(self):
        """Check if we should run prediction"""
        return len(self.buffer) >= self.window_size
    
    def predict(self, return_confidence=False):
        """
        Run prediction on current buffer
        
        Returns:
            text: Predicted text
            confidence: Prediction confidence (if return_confidence=True)
        """
        if len(self.buffer) < self.window_size:
            return "" if not return_confidence else ("", 0.0)
        
        # Prepare input
        landmarks = torch.FloatTensor(self.buffer).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            log_probs = self.model(landmarks)  # (1, frames, vocab)
        
        # CTC Decode (greedy for speed, can use beam search for accuracy)
        preds = torch.argmax(log_probs, dim=-1).squeeze(0).cpu().numpy()
        
        # Decode to text
        text, confidence = self._decode_ctc(preds, log_probs.squeeze(0))
        
        # Only return if confidence is high enough
        if confidence < self.confidence_threshold:
            return "" if not return_confidence else ("", confidence)
        
        if return_confidence:
            return text, confidence
        return text
    
    def _decode_ctc(self, indices, log_probs):
        """
        Decode CTC output to text
        
        Args:
            indices: Predicted indices
            log_probs: Log probabilities for confidence calculation
        """
        # Remove consecutive duplicates and blanks
        decoded = []
        prev_idx = -1
        
        for idx in indices:
            if idx != 0 and idx != prev_idx:  # 0 is blank
                decoded.append(idx)
            prev_idx = idx
        
        # Convert to text
        text = self.vocab.indices_to_text(decoded, mode='word')
        
        # Calculate confidence (average of top predictions)
        probs = torch.exp(log_probs)
        top_probs = torch.max(probs, dim=-1)[0]
        confidence = top_probs.mean().item()
        
        return text, confidence
    
    def reset(self):
        """Clear buffer for new sign"""
        self.buffer = []
        self.predictions_history = []


def create_mobile_model(vocab_size=5004, hidden_dim=128):
    """
    Factory function to create optimized mobile model
    
    Args:
        vocab_size: Size of vocabulary (default: 5000 words + 4 special tokens)
        hidden_dim: Hidden dimension (128 for speed, 256 for accuracy)
    
    Returns:
        model: MobileSignLanguageModel instance
    """
    model = MobileSignLanguageModel(
        input_dim=138,  # MediaPipe landmarks: 2 hands * 21 * 3 + 2 shoulders * 3 + 2 elbows * 3
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        num_layers=1  # Single layer for speed
    )
    
    print(f"Model created:")
    print(f"  Parameters: {model.get_num_params():,}")
    print(f"  Estimated size: {model.get_model_size_mb():.2f} MB")
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing Mobile Sign Language Model")
    print("="*60)
    
    model = create_mobile_model(vocab_size=5004, hidden_dim=128)
    
    # Test forward pass
    batch_size = 4
    frames = 60
    features = 138
    
    dummy_input = torch.randn(batch_size, frames, features)
    dummy_lengths = torch.tensor([60, 55, 50, 45])
    
    print("\nTesting forward pass...")
    output = model(dummy_input, dummy_lengths)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    print("\n✅ Model test passed!")
