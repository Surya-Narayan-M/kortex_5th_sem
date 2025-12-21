"""
Enhanced Hybrid CTC-Attention Model for Sign Language Recognition
Combines streaming CTC predictions with high-accuracy attention decoder

Architecture:
- Multi-scale CNN for capturing gestures at different temporal scales
- Squeeze-Excite blocks for learning landmark importance
- Deep BiGRU encoder with residual connections
- CTC head for streaming predictions
- Cross-attention decoder for final polished output
- Joint training with CTC + Attention loss

Target: 85%+ character accuracy, <200ms mobile inference, <10MB model
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== BUILDING BLOCKS ====================

class SqueezeExcite(nn.Module):
    """
    Squeeze-and-Excitation block
    Learns to weight channels (landmark groups) by importance
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, reduced),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: (batch, channels, time)
        batch, channels, time = x.shape
        weights = self.squeeze(x).view(batch, channels)
        weights = self.excite(weights).view(batch, channels, 1)
        return x * weights


class MultiScaleConv(nn.Module):
    """
    Multi-scale temporal convolution
    Captures gestures at different speeds with parallel kernels
    """
    def __init__(self, in_channels, out_channels, kernels=[3, 5, 7, 11]):
        super().__init__()
        
        assert out_channels % len(kernels) == 0
        branch_channels = out_channels // len(kernels)
        
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, branch_channels, kernel_size=k, 
                         padding=k//2, bias=False),
                nn.BatchNorm1d(branch_channels),
                nn.ReLU(inplace=True)
            ) for k in kernels
        ])
        
        self.se = SqueezeExcite(out_channels)
        
    def forward(self, x):
        # x: (batch, channels, time)
        branches = [branch(x) for branch in self.branches]
        out = torch.cat(branches, dim=1)
        out = self.se(out)
        return out


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for temporal position awareness
    """
    def __init__(self, d_model, max_len=1000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch, time, features)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ==================== ENCODER ====================

class HybridEncoder(nn.Module):
    """
    Multi-scale CNN + Deep BiGRU Encoder
    Shared between CTC and Attention heads
    """
    def __init__(self, input_dim=414, hidden_dim=384, num_layers=3, dropout=0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Multi-scale CNN feature extraction
        self.conv_layers = nn.Sequential(
            MultiScaleConv(hidden_dim, hidden_dim, kernels=[3, 5, 7, 11]),
            nn.Dropout(dropout),
            MultiScaleConv(hidden_dim, hidden_dim, kernels=[3, 5, 7, 11]),
            nn.Dropout(dropout),
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=2000, dropout=dropout)
        
        # BiGRU layers with residual connections
        self.gru_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for i in range(num_layers):
            self.gru_layers.append(
                nn.GRU(
                    input_size=hidden_dim,
                    hidden_size=hidden_dim // 2,
                    batch_first=True,
                    bidirectional=True
                )
            )
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, lengths=None):
        """
        Args:
            x: (batch, time, input_dim) preprocessed landmark features
            lengths: (batch,) actual sequence lengths
            
        Returns:
            outputs: (batch, time, hidden_dim) encoder states
            hidden: final hidden state for decoder initialization
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_proj(x)  # (batch, time, hidden_dim)
        
        # Multi-scale CNN
        x = x.permute(0, 2, 1)  # (batch, hidden_dim, time)
        x = self.conv_layers(x)
        x = x.permute(0, 2, 1)  # (batch, time, hidden_dim)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # BiGRU layers with residual connections
        hidden_states = []
        for gru, ln in zip(self.gru_layers, self.layer_norms):
            residual = x
            
            if lengths is not None:
                # Pack for variable length
                packed = nn.utils.rnn.pack_padded_sequence(
                    x, lengths.cpu().clamp(max=seq_len), 
                    batch_first=True, enforce_sorted=False
                )
                packed_out, hidden = gru(packed)
                x, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, 
                                                         total_length=seq_len)
            else:
                x, hidden = gru(x)
            
            # Residual + LayerNorm
            x = ln(x + residual)
            x = self.dropout(x)
            hidden_states.append(hidden)
        
        # Combine final hidden states
        final_hidden = hidden_states[-1]  # (2, batch, hidden_dim//2)
        
        return x, final_hidden


# ==================== CTC HEAD ====================

class CTCHead(nn.Module):
    """
    CTC output head for streaming predictions
    """
    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, vocab_size)
        )
    
    def forward(self, encoder_outputs):
        """
        Args:
            encoder_outputs: (batch, time, hidden_dim)
            
        Returns:
            log_probs: (batch, time, vocab_size) log probabilities
        """
        logits = self.proj(encoder_outputs)
        return F.log_softmax(logits, dim=-1)


# ==================== ATTENTION DECODER ====================

class MultiHeadAttention(nn.Module):
    """
    Multi-head cross-attention for decoder
    """
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch, tgt_len, hidden_dim)
            key: (batch, src_len, hidden_dim)
            value: (batch, src_len, hidden_dim)
            mask: (batch, src_len) source padding mask
            
        Returns:
            output: (batch, tgt_len, hidden_dim)
            attention_weights: (batch, num_heads, tgt_len, src_len)
        """
        batch_size = query.size(0)
        
        # Project and reshape to (batch, heads, len, head_dim)
        Q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch, heads, tgt, src)
        
        # Apply mask
        if mask is not None:
            # mask: (batch, src_len) -> (batch, 1, 1, src_len)
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)  # (batch, heads, tgt, head_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        
        output = self.out_proj(context)
        
        return output, attention_weights


class AttentionDecoder(nn.Module):
    """
    GRU decoder with multi-head cross-attention
    """
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=384, 
                 num_layers=2, num_heads=4, dropout=0.3):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embed_proj = nn.Linear(embedding_dim, hidden_dim)
        
        # Cross-attention
        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        
        # GRU decoder
        self.gru = nn.GRU(
            input_size=hidden_dim * 2,  # embedded + context
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # gru_out + context
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tgt, encoder_outputs, encoder_hidden, src_mask=None, 
                teacher_forcing_ratio=0.5):
        """
        Args:
            tgt: (batch, tgt_len) target token indices
            encoder_outputs: (batch, src_len, hidden_dim)
            encoder_hidden: (2, batch, hidden_dim//2) encoder final hidden
            src_mask: (batch, src_len) source padding mask
            teacher_forcing_ratio: probability of using ground truth
            
        Returns:
            outputs: (batch, tgt_len, vocab_size) output logits
            attention_weights: attention visualization
        """
        batch_size, tgt_len = tgt.shape
        
        # Initialize decoder hidden from encoder
        # Combine bidirectional hidden: (2, batch, hidden//2) -> (num_layers, batch, hidden)
        hidden = torch.cat([encoder_hidden[0], encoder_hidden[1]], dim=1)  # (batch, hidden)
        hidden = hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)  # (layers, batch, hidden)
        
        # Prepare outputs
        outputs = torch.zeros(batch_size, tgt_len, self.vocab_size, device=tgt.device)
        attentions = []
        
        # First input is SOS token
        input_token = tgt[:, 0]  # (batch,)
        
        for t in range(1, tgt_len):
            # Embed current token
            embedded = self.embedding(input_token)  # (batch, embed_dim)
            embedded = self.embed_proj(embedded)  # (batch, hidden_dim)
            embedded = self.dropout(embedded)
            
            # Cross-attention
            query = hidden[-1].unsqueeze(1)  # (batch, 1, hidden_dim)
            context, attn_weights = self.attention(query, encoder_outputs, encoder_outputs, src_mask)
            context = context.squeeze(1)  # (batch, hidden_dim)
            attentions.append(attn_weights)
            
            # GRU input: concatenate embedded and context
            gru_input = torch.cat([embedded, context], dim=1)  # (batch, hidden*2)
            gru_input = gru_input.unsqueeze(1)  # (batch, 1, hidden*2)
            
            # GRU step
            gru_out, hidden = self.gru(gru_input, hidden)
            gru_out = gru_out.squeeze(1)  # (batch, hidden)
            
            # Output projection
            combined = torch.cat([gru_out, context], dim=1)  # (batch, hidden*2)
            output = self.output_proj(combined)  # (batch, vocab_size)
            outputs[:, t] = output
            
            # Teacher forcing
            if torch.rand(1).item() < teacher_forcing_ratio:
                input_token = tgt[:, t]
            else:
                input_token = output.argmax(dim=1)
        
        return outputs, attentions
    
    def predict(self, encoder_outputs, encoder_hidden, src_mask=None, 
                max_len=100, sos_idx=1, eos_idx=2):
        """
        Greedy decoding for inference
        """
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        
        # Initialize hidden
        hidden = torch.cat([encoder_hidden[0], encoder_hidden[1]], dim=1)
        hidden = hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)
        
        # Start with SOS
        input_token = torch.full((batch_size,), sos_idx, dtype=torch.long, device=device)
        
        predictions = [input_token]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_len):
            # Embed
            embedded = self.embed_proj(self.embedding(input_token))
            
            # Attention
            query = hidden[-1].unsqueeze(1)
            context, _ = self.attention(query, encoder_outputs, encoder_outputs, src_mask)
            context = context.squeeze(1)
            
            # GRU
            gru_input = torch.cat([embedded, context], dim=1).unsqueeze(1)
            gru_out, hidden = self.gru(gru_input, hidden)
            gru_out = gru_out.squeeze(1)
            
            # Output
            combined = torch.cat([gru_out, context], dim=1)
            output = self.output_proj(combined)
            input_token = output.argmax(dim=1)
            
            predictions.append(input_token)
            
            # Check EOS
            finished |= (input_token == eos_idx)
            if finished.all():
                break
        
        return torch.stack(predictions, dim=1)


# ==================== COMPLETE MODEL ====================

class HybridCTCAttentionModel(nn.Module):
    """
    Complete Hybrid CTC-Attention model
    
    - Shared encoder between CTC and Attention heads
    - CTC for streaming predictions (partial results while signing)
    - Attention for final polished output
    - Joint training: L = λ*CTC + (1-λ)*Attention
    """
    def __init__(self, input_dim=414, hidden_dim=384, vocab_size=75,
                 embedding_dim=256, encoder_layers=3, decoder_layers=2,
                 num_heads=4, dropout=0.3):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        # Shared encoder
        self.encoder = HybridEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=encoder_layers,
            dropout=dropout
        )
        
        # CTC head (for streaming)
        self.ctc_head = CTCHead(hidden_dim, vocab_size)
        
        # Attention decoder (for final output)
        self.decoder = AttentionDecoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=decoder_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with Xavier/Kaiming"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, src, src_lengths, tgt=None, teacher_forcing_ratio=0.5):
        """
        Forward pass for training
        
        Args:
            src: (batch, src_len, input_dim) source landmarks
            src_lengths: (batch,) source lengths
            tgt: (batch, tgt_len) target token indices
            teacher_forcing_ratio: for attention decoder
            
        Returns:
            ctc_log_probs: (batch, src_len, vocab_size) CTC output
            attn_outputs: (batch, tgt_len, vocab_size) Attention output
        """
        batch_size = src.size(0)
        max_src_len = src.size(1)
        
        # Encode
        encoder_outputs, encoder_hidden = self.encoder(src, src_lengths)
        
        # CTC output
        ctc_log_probs = self.ctc_head(encoder_outputs)
        
        # Attention output (if target provided)
        attn_outputs = None
        if tgt is not None:
            # Create source mask
            src_mask = torch.arange(max_src_len, device=src.device).unsqueeze(0) < src_lengths.unsqueeze(1)
            
            attn_outputs, _ = self.decoder(
                tgt, encoder_outputs, encoder_hidden, 
                src_mask, teacher_forcing_ratio
            )
        
        return ctc_log_probs, attn_outputs
    
    def predict_streaming(self, src, src_lengths=None):
        """
        CTC prediction for streaming/real-time inference
        
        Args:
            src: (batch, src_len, input_dim) source landmarks
            src_lengths: optional sequence lengths
            
        Returns:
            predictions: CTC decoded indices
            log_probs: raw log probabilities
        """
        self.eval()
        with torch.no_grad():
            encoder_outputs, _ = self.encoder(src, src_lengths)
            log_probs = self.ctc_head(encoder_outputs)
            predictions = log_probs.argmax(dim=-1)
        return predictions, log_probs
    
    def predict_final(self, src, src_lengths=None, max_len=100):
        """
        Attention prediction for final polished output
        
        Args:
            src: (batch, src_len, input_dim) source landmarks
            src_lengths: optional sequence lengths
            max_len: maximum output length
            
        Returns:
            predictions: (batch, seq_len) predicted token indices
        """
        self.eval()
        with torch.no_grad():
            batch_size = src.size(0)
            max_src_len = src.size(1)
            
            encoder_outputs, encoder_hidden = self.encoder(src, src_lengths)
            
            # Create source mask
            if src_lengths is not None:
                src_mask = torch.arange(max_src_len, device=src.device).unsqueeze(0) < src_lengths.unsqueeze(1)
            else:
                src_mask = None
            
            predictions = self.decoder.predict(
                encoder_outputs, encoder_hidden, src_mask, max_len
            )
        
        return predictions
    
    def get_num_params(self):
        """Get total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size_mb(self):
        """Estimate model size in MB"""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 ** 2)


# ==================== FACTORY FUNCTIONS ====================

def create_hybrid_model(vocab_size=75, hidden_dim=384, input_dim=414,
                        encoder_layers=3, decoder_layers=2):
    """
    Create hybrid CTC-Attention model
    
    Args:
        vocab_size: Character vocabulary size (~75 with punctuation)
        hidden_dim: Hidden dimension (384 for accuracy/speed balance)
        input_dim: Input feature dimension (414 with velocity/acceleration)
        encoder_layers: Number of BiGRU encoder layers
        decoder_layers: Number of GRU decoder layers
        
    Returns:
        model: HybridCTCAttentionModel
    """
    model = HybridCTCAttentionModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        embedding_dim=256,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        num_heads=4,
        dropout=0.3
    )
    
    print(f"\n{'='*60}")
    print("Hybrid CTC-Attention Model Created")
    print(f"{'='*60}")
    print(f"  Input dimension: {input_dim}")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Encoder layers: {encoder_layers}")
    print(f"  Decoder layers: {decoder_layers}")
    print(f"  Attention heads: 4")
    print(f"  Total parameters: {model.get_num_params():,}")
    print(f"  Model size: {model.get_model_size_mb():.2f} MB")
    print(f"{'='*60}\n")
    
    return model


def create_mobile_hybrid_model(vocab_size=75, input_dim=414):
    """
    Create smaller model optimized for mobile deployment
    """
    return create_hybrid_model(
        vocab_size=vocab_size,
        hidden_dim=256,
        input_dim=input_dim,
        encoder_layers=2,
        decoder_layers=1
    )


# ==================== TEST ====================

if __name__ == "__main__":
    print("Testing Hybrid CTC-Attention Model")
    print("="*60)
    
    # Create model
    model = create_hybrid_model(vocab_size=75, hidden_dim=384, input_dim=414)
    
    # Test forward pass
    batch_size = 4
    src_len = 100
    tgt_len = 50
    
    src = torch.randn(batch_size, src_len, 414)
    src_lengths = torch.tensor([100, 90, 80, 70])
    tgt = torch.randint(0, 75, (batch_size, tgt_len))
    
    print("\nTesting training forward pass...")
    model.train()
    ctc_out, attn_out = model(src, src_lengths, tgt, teacher_forcing_ratio=0.5)
    
    print(f"  Source shape: {src.shape}")
    print(f"  Target shape: {tgt.shape}")
    print(f"  CTC output shape: {ctc_out.shape}")
    print(f"  Attention output shape: {attn_out.shape}")
    
    print("\nTesting streaming prediction (CTC)...")
    preds, log_probs = model.predict_streaming(src, src_lengths)
    print(f"  Streaming predictions shape: {preds.shape}")
    
    print("\nTesting final prediction (Attention)...")
    final_preds = model.predict_final(src, src_lengths, max_len=50)
    print(f"  Final predictions shape: {final_preds.shape}")
    
    print("\n✅ All tests passed!")
    
    # Test mobile model
    print("\n" + "="*60)
    print("Testing Mobile Model")
    print("="*60)
    mobile_model = create_mobile_hybrid_model(vocab_size=75)
    print(f"Mobile model parameters: {mobile_model.get_num_params():,}")
    print(f"Mobile model size: {mobile_model.get_model_size_mb():.2f} MB")
