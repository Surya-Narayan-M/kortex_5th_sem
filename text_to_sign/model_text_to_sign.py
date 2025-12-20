"""
Text-to-Sign Language Generation Model
Transformer-based architecture for generating landmark sequences from text
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TextToSignModel(nn.Module):
    """
    Transformer Encoder-Decoder for Text-to-Sign Generation
    
    Input: Text tokens (word indices)
    Output: Landmark sequences (138 features per frame)
    """
    
    def __init__(self, vocab_size=5004, hidden_dim=256, num_encoder_layers=4, 
                 num_decoder_layers=4, num_heads=8, max_frames=150):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_frames = max_frames
        
        # Text Embedding (Encoder input)
        self.text_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # Transformer Encoder (understands text)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Frame Embedding (Decoder input - for auto-regressive generation)
        self.frame_embedding = nn.Linear(138, hidden_dim)
        self.pos_decoder = PositionalEncoding(hidden_dim)
        
        # Transformer Decoder (generates landmarks frame-by-frame)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # Output projection (138 landmark features)
        self.landmark_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 138)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, text_tokens, target_landmarks=None, teacher_forcing_ratio=0.5):
        """
        Forward pass
        
        Args:
            text_tokens: (batch, text_len) - input text token indices
            target_landmarks: (batch, frames, 138) - ground truth landmarks (for training)
            teacher_forcing_ratio: probability of using ground truth during training
        
        Returns:
            landmarks: (batch, frames, 138) - generated landmark sequences
        """
        batch_size = text_tokens.size(0)
        
        # Encode text
        text_embedded = self.text_embedding(text_tokens)  # (batch, text_len, hidden_dim)
        text_embedded = self.pos_encoder(text_embedded)
        
        # Create padding mask for text
        text_padding_mask = (text_tokens == 0)  # Assuming 0 is padding
        
        memory = self.encoder(text_embedded, src_key_padding_mask=text_padding_mask)
        
        # Decode landmarks (auto-regressive or parallel)
        if self.training and target_landmarks is not None:
            # Teacher forcing during training
            generated_landmarks = self._decode_teacher_forcing(
                memory, target_landmarks, text_padding_mask, teacher_forcing_ratio
            )
        else:
            # Inference: auto-regressive generation
            generated_landmarks = self._decode_autoregressive(memory, batch_size)
        
        return generated_landmarks
    
    def _decode_teacher_forcing(self, memory, target_landmarks, memory_mask, teacher_forcing_ratio):
        """Decode with teacher forcing (training)"""
        batch_size, num_frames, _ = target_landmarks.shape
        
        # Embed target landmarks (shift right by 1 for auto-regressive)
        # Start with zero frame
        decoder_input = torch.zeros(batch_size, 1, 138, device=target_landmarks.device)
        decoder_input = torch.cat([decoder_input, target_landmarks[:, :-1, :]], dim=1)
        
        decoder_embedded = self.frame_embedding(decoder_input)  # (batch, frames, hidden_dim)
        decoder_embedded = self.pos_decoder(decoder_embedded)
        
        # Generate causal mask (prevent looking ahead)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(num_frames).to(target_landmarks.device)
        
        # Decode
        decoded = self.decoder(
            decoder_embedded,
            memory,
            tgt_mask=causal_mask,
            memory_key_padding_mask=memory_mask
        )
        
        # Project to landmarks
        landmarks = self.landmark_head(decoded)  # (batch, frames, 138)
        
        return landmarks
    
    def _decode_autoregressive(self, memory, batch_size):
        """Auto-regressive decoding (inference)"""
        device = memory.device
        
        # Start with zero frame
        generated = torch.zeros(batch_size, 1, 138, device=device)
        
        for step in range(self.max_frames):
            # Embed generated frames
            decoder_input = self.frame_embedding(generated)
            decoder_input = self.pos_decoder(decoder_input)
            
            # Generate causal mask
            seq_len = generated.size(1)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
            
            # Decode
            decoded = self.decoder(
                decoder_input,
                memory,
                tgt_mask=causal_mask
            )
            
            # Project last frame
            next_landmarks = self.landmark_head(decoded[:, -1:, :])  # (batch, 1, 138)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_landmarks], dim=1)
            
            # Stop if all sequences reach max length
            if generated.size(1) >= self.max_frames:
                break
        
        return generated[:, 1:, :]  # Remove initial zero frame
    
    def generate(self, text_tokens, max_frames=None):
        """
        Generate landmark sequence from text (inference only)
        
        Args:
            text_tokens: (batch, text_len) or (text_len,)
            max_frames: Maximum frames to generate
        
        Returns:
            landmarks: (batch, frames, 138) or (frames, 138)
        """
        self.eval()
        
        # Handle single sequence
        if text_tokens.dim() == 1:
            text_tokens = text_tokens.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        if max_frames is not None:
            original_max = self.max_frames
            self.max_frames = max_frames
        
        with torch.no_grad():
            landmarks = self.forward(text_tokens)
        
        if max_frames is not None:
            self.max_frames = original_max
        
        if squeeze_output:
            landmarks = landmarks.squeeze(0)
        
        return landmarks


def create_text_to_sign_model(vocab_size=5004, hidden_dim=256, num_layers=4):
    """
    Factory function to create text-to-sign model
    
    Args:
        vocab_size: Size of text vocabulary
        hidden_dim: Hidden dimension (256 for balance, 512 for quality)
        num_layers: Number of transformer layers (4-6)
    
    Returns:
        model: TextToSignModel instance
    """
    model = TextToSignModel(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        num_heads=8,
        max_frames=150
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    size_mb = num_params * 4 / (1024 ** 2)  # FP32
    
    print(f"Text-to-Sign Model Created:")
    print(f"  Parameters: {num_params:,}")
    print(f"  Estimated size: {size_mb:.2f} MB")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Hidden dim: {hidden_dim}")
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing Text-to-Sign Model")
    print("="*60)
    
    model = create_text_to_sign_model(vocab_size=5004, hidden_dim=256, num_layers=4)
    
    # Test forward pass
    batch_size = 2
    text_len = 10
    frames = 60
    
    dummy_text = torch.randint(1, 5000, (batch_size, text_len))
    dummy_landmarks = torch.randn(batch_size, frames, 138)
    
    print("\nTesting forward pass (training mode)...")
    model.train()
    output = model(dummy_text, dummy_landmarks, teacher_forcing_ratio=0.5)
    print(f"Input text shape: {dummy_text.shape}")
    print(f"Target landmarks shape: {dummy_landmarks.shape}")
    print(f"Output shape: {output.shape}")
    
    print("\nTesting generation (inference mode)...")
    model.eval()
    generated = model.generate(dummy_text, max_frames=60)
    print(f"Generated landmarks shape: {generated.shape}")
    
    print("\n✅ Model test passed!")
