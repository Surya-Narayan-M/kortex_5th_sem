"""
Sequence-to-Sequence Sign Language Recognition with Attention
For full sentence videos where we know the complete text

Architecture: BiGRU Encoder → Attention → GRU Decoder
Target: 90%+ accuracy on sentence-level prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class Attention(nn.Module):
    """Bahdanau (Additive) Attention mechanism"""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Attention components
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        """
        Args:
            decoder_hidden: (batch, hidden_dim) - current decoder state
            encoder_outputs: (batch, src_len, hidden_dim) - all encoder states
            mask: (batch, src_len) - padding mask
            
        Returns:
            attention_weights: (batch, src_len)
            context: (batch, hidden_dim)
        """
        batch_size, src_len, _ = encoder_outputs.shape
        
        # Repeat decoder hidden for each encoder output
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)  # (batch, src_len, hidden_dim)
        
        # Concatenate and compute attention scores
        energy = torch.tanh(self.attn(torch.cat([decoder_hidden, encoder_outputs], dim=2)))  # (batch, src_len, hidden_dim)
        attention = self.v(energy).squeeze(2)  # (batch, src_len)
        
        # Apply mask if provided (ignore padding)
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)
        
        # Softmax to get attention weights
        attention_weights = F.softmax(attention, dim=1)  # (batch, src_len)
        
        # Weighted sum of encoder outputs
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)  # (batch, 1, hidden_dim)
        context = context.squeeze(1)  # (batch, hidden_dim)
        
        return attention_weights, context


class Encoder(nn.Module):
    """
    BiGRU Encoder for sign language video features
    Processes landmark sequences into context vectors
    """
    
    def __init__(self, input_dim=138, hidden_dim=512, num_layers=3, dropout=0.3):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # CNN feature extractor (spatial patterns)
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        # BiGRU encoder (temporal patterns) - 3 layers for 90%+ accuracy
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, lengths=None):
        """
        Args:
            x: (batch, frames, 138) - landmark sequences
            lengths: (batch,) - actual sequence lengths
            
        Returns:
            outputs: (batch, frames, hidden_dim) - all encoder states
            hidden: (num_layers*2, batch, hidden_dim//2) - final hidden states
        """
        batch_size, frames, features = x.shape
        
        # CNN feature extraction
        x = x.permute(0, 2, 1)  # (batch, features, frames)
        cnn_out = self.cnn(x)
        x = cnn_out.permute(0, 2, 1)  # (batch, frames, hidden_dim)
        
        # BiGRU encoding
        if lengths is not None:
            packed_x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            outputs, hidden = self.gru(packed_x)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        else:
            outputs, hidden = self.gru(x)
        
        # Add residual connection
        outputs = outputs + x
        
        # Layer normalization
        outputs = self.layer_norm(outputs)
        
        return outputs, hidden


class Decoder(nn.Module):
    """
    GRU Decoder with Attention
    Generates text character-by-character attending to encoder outputs
    """
    
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Character embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Attention mechanism
        self.attention = Attention(hidden_dim)
        
        # GRU decoder (unidirectional)
        self.gru = nn.GRU(
            input_size=embedding_dim + hidden_dim,  # embedding + context
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Output projection
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim * 2 + embedding_dim, hidden_dim),  # context + hidden + embedding
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_char, hidden, encoder_outputs, mask=None):
        """
        Single decoding step
        
        Args:
            input_char: (batch,) - current character index
            hidden: (num_layers, batch, hidden_dim) - previous decoder state
            encoder_outputs: (batch, src_len, hidden_dim) - encoder outputs
            mask: (batch, src_len) - padding mask
            
        Returns:
            output: (batch, vocab_size) - character predictions
            hidden: (num_layers, batch, hidden_dim) - new decoder state
            attention_weights: (batch, src_len) - attention distribution
        """
        # Embed input character
        input_char = input_char.unsqueeze(1)  # (batch, 1)
        embedded = self.dropout(self.embedding(input_char))  # (batch, 1, embedding_dim)
        
        # Compute attention using top decoder layer
        decoder_hidden = hidden[-1]  # (batch, hidden_dim)
        attention_weights, context = self.attention(decoder_hidden, encoder_outputs, mask)
        
        # Concatenate embedding and context
        context = context.unsqueeze(1)  # (batch, 1, hidden_dim)
        gru_input = torch.cat([embedded, context], dim=2)  # (batch, 1, embedding_dim + hidden_dim)
        
        # GRU step
        gru_output, hidden = self.gru(gru_input, hidden)  # gru_output: (batch, 1, hidden_dim)
        
        # Output projection: combine gru_output, context, and embedding
        gru_output = gru_output.squeeze(1)  # (batch, hidden_dim)
        context = context.squeeze(1)  # (batch, hidden_dim)
        embedded = embedded.squeeze(1)  # (batch, embedding_dim)
        
        output = self.fc_out(torch.cat([gru_output, context, embedded], dim=1))  # (batch, vocab_size)
        
        return output, hidden, attention_weights


class Seq2SeqSignLanguageModel(nn.Module):
    """
    Complete Sequence-to-Sequence model with Attention
    Encoder-Decoder architecture for sentence-level sign language recognition
    """
    
    def __init__(self, input_dim=138, hidden_dim=512, vocab_size=100, 
                 embedding_dim=256, encoder_layers=3, decoder_layers=2, dropout=0.3):
        super().__init__()
        
        self.encoder = Encoder(input_dim, hidden_dim, encoder_layers, dropout)
        self.decoder = Decoder(vocab_size, embedding_dim, hidden_dim, decoder_layers, dropout)
        
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        # Bridge from encoder hidden to decoder hidden
        self.bridge = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, src, src_lengths, trg, teacher_forcing_ratio=0.5):
        """
        Args:
            src: (batch, src_len, 138) - landmark sequences
            src_lengths: (batch,) - actual sequence lengths
            trg: (batch, trg_len) - target character sequences
            teacher_forcing_ratio: probability of using teacher forcing
            
        Returns:
            outputs: (batch, trg_len, vocab_size) - character predictions
            attentions: (batch, trg_len, src_len) - attention weights
        """
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        
        # Encode
        encoder_outputs, encoder_hidden = self.encoder(src, src_lengths)
        
        # Create source mask (to ignore padding in attention)
        max_src_len = src.shape[1]
        mask = torch.arange(max_src_len, device=src.device).unsqueeze(0) < src_lengths.unsqueeze(1)
        
        # Initialize decoder hidden (use last encoder hidden from all layers)
        # encoder_hidden: (num_layers*2, batch, hidden_dim//2) -> need (decoder_layers, batch, hidden_dim)
        decoder_hidden = self._bridge_encoder_hidden(encoder_hidden)
        
        # Prepare outputs
        outputs = torch.zeros(batch_size, trg_len, self.vocab_size, device=src.device)
        attentions = torch.zeros(batch_size, trg_len, max_src_len, device=src.device)
        
        # First input is SOS token (index 1)
        input_char = trg[:, 0]
        
        # Decode step by step
        for t in range(1, trg_len):
            output, decoder_hidden, attention_weights = self.decoder(
                input_char, decoder_hidden, encoder_outputs, mask
            )
            
            outputs[:, t] = output
            attentions[:, t] = attention_weights
            
            # Teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_char = trg[:, t] if teacher_force else top1
        
        return outputs, attentions
    
    def _bridge_encoder_hidden(self, encoder_hidden):
        """
        Convert encoder hidden states to decoder initial hidden
        encoder_hidden: (encoder_layers*2, batch, hidden_dim//2) bidirectional
        decoder needs: (decoder_layers, batch, hidden_dim) unidirectional
        """
        # Combine forward and backward directions
        num_encoder_layers = encoder_hidden.shape[0] // 2
        batch_size = encoder_hidden.shape[1]
        
        # Take last layer, combine forward and backward
        forward_hidden = encoder_hidden[-2]  # (batch, hidden_dim//2)
        backward_hidden = encoder_hidden[-1]  # (batch, hidden_dim//2)
        combined = torch.cat([forward_hidden, backward_hidden], dim=1)  # (batch, hidden_dim)
        
        # Project and expand to decoder layers
        decoder_hidden = self.bridge(combined)  # (batch, hidden_dim)
        decoder_hidden = decoder_hidden.unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)
        
        return decoder_hidden
    
    def predict(self, src, src_lengths, max_len=100, sos_idx=1, eos_idx=2):
        """
        Greedy decoding for inference
        
        Args:
            src: (batch, src_len, 138)
            src_lengths: (batch,)
            max_len: maximum output length
            sos_idx: start-of-sequence index
            eos_idx: end-of-sequence index
            
        Returns:
            predictions: (batch, seq_len) - predicted character indices
            attentions: (batch, seq_len, src_len) - attention weights
        """
        self.eval()
        batch_size = src.shape[0]
        
        with torch.no_grad():
            # Encode
            encoder_outputs, encoder_hidden = self.encoder(src, src_lengths)
            
            # Create mask
            max_src_len = src.shape[1]
            mask = torch.arange(max_src_len, device=src.device).unsqueeze(0) < src_lengths.unsqueeze(1)
            
            # Initialize decoder
            decoder_hidden = self._bridge_encoder_hidden(encoder_hidden)
            
            # Start with SOS token
            input_char = torch.full((batch_size,), sos_idx, dtype=torch.long, device=src.device)
            
            predictions = [input_char]
            attentions_list = []
            finished = torch.zeros(batch_size, dtype=torch.bool, device=src.device)
            
            for t in range(max_len):
                output, decoder_hidden, attention_weights = self.decoder(
                    input_char, decoder_hidden, encoder_outputs, mask
                )
                
                # Get most likely character
                input_char = output.argmax(1)
                predictions.append(input_char)
                attentions_list.append(attention_weights)
                
                # Check for EOS
                finished |= (input_char == eos_idx)
                if finished.all():
                    break
            
            predictions = torch.stack(predictions, dim=1)  # (batch, seq_len)
            attentions = torch.stack(attentions_list, dim=1)  # (batch, seq_len, src_len)
        
        return predictions, attentions
    
    def get_num_params(self):
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())


def create_seq2seq_model(vocab_size=100, hidden_dim=512, embedding_dim=256):
    """
    Factory function to create Seq2Seq model
    
    Args:
        vocab_size: Character vocabulary size (~50 for English)
        hidden_dim: Hidden dimension (512 for 90%+ accuracy)
        embedding_dim: Character embedding dimension
        
    Returns:
        model: Seq2SeqSignLanguageModel
    """
    model = Seq2SeqSignLanguageModel(
        input_dim=138,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        encoder_layers=3,  # Deep encoder for complex patterns
        decoder_layers=2,  # 2-layer decoder
        dropout=0.3
    )
    
    print(f"Seq2Seq Model created:")
    print(f"  Parameters: {model.get_num_params():,}")
    print(f"  Encoder: 3-layer BiGRU")
    print(f"  Decoder: 2-layer GRU with Attention")
    print(f"  Hidden dim: {hidden_dim}")
    
    return model


if __name__ == "__main__":
    print("Testing Seq2Seq Sign Language Model")
    print("="*60)
    
    # Create model
    model = create_seq2seq_model(vocab_size=50, hidden_dim=512)
    
    # Test forward pass
    batch_size = 4
    src_len = 60
    trg_len = 20
    
    src = torch.randn(batch_size, src_len, 138)
    src_lengths = torch.tensor([60, 55, 50, 45])
    trg = torch.randint(0, 50, (batch_size, trg_len))
    
    print("\nTesting training forward pass...")
    outputs, attentions = model(src, src_lengths, trg, teacher_forcing_ratio=0.5)
    print(f"Input shape: {src.shape}")
    print(f"Target shape: {trg.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Attention shape: {attentions.shape}")
    
    print("\nTesting inference...")
    predictions, pred_attentions = model.predict(src, src_lengths, max_len=20)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Prediction attentions shape: {pred_attentions.shape}")
    
    print("\n✅ Seq2Seq model test passed!")
