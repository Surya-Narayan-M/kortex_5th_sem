"""
Model Export Utilities V2
Exports trained dual decoder model to ONNX format for deployment
Supports exporting both GRU (fast) and Transformer (accurate) decoder variants
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import time
from typing import Optional, Dict, Any

from model_hybrid_v2 import HybridCTCAttentionModelV2


class SingleDecoderWrapper(nn.Module):
    """
    Wrapper that exposes only one decoder for export.
    This allows exporting separate ONNX models for GRU and Transformer decoders.
    """
    
    def __init__(self, model: HybridCTCAttentionModelV2, decoder_type: str = 'gru'):
        super().__init__()
        self.model = model
        self.decoder_type = decoder_type
        
    def forward(self, landmarks: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        """
        Forward pass for inference.
        
        Args:
            landmarks: (batch, frames, features) - input landmarks
            lengths: (batch,) - actual sequence lengths (optional)
        
        Returns:
            predictions: (batch, max_len) - decoded character indices
        """
        batch_size = landmarks.size(0)
        
        if lengths is None:
            lengths = torch.full((batch_size,), landmarks.size(1), device=landmarks.device)
        
        # Use predict_final from the model with specified decoder
        predictions = self.model.predict_final(
            landmarks, lengths, 
            decoder=self.decoder_type,
            max_len=150
        )
        
        return predictions


class StreamingEncoderWrapper(nn.Module):
    """
    Wrapper for streaming encoder-only inference.
    Returns CTC log probabilities for real-time streaming recognition.
    """
    
    def __init__(self, model: HybridCTCAttentionModelV2):
        super().__init__()
        self.input_proj = model.input_proj
        self.encoder = model.encoder
        self.ctc_proj = model.ctc_proj
        
    def forward(self, landmarks: torch.Tensor):
        """
        Stream-friendly forward pass.
        
        Args:
            landmarks: (batch, frames, features)
        
        Returns:
            ctc_log_probs: (batch, frames, vocab_size)
        """
        # Project input
        x = self.input_proj(landmarks)  # (B, T, hidden)
        
        # Encode
        x = x.transpose(0, 1)  # (T, B, hidden)
        encoded, _ = self.encoder(x)
        encoded = encoded.transpose(0, 1)  # (B, T, hidden)
        
        # CTC projection
        ctc_logits = self.ctc_proj(encoded)
        ctc_log_probs = torch.log_softmax(ctc_logits, dim=-1)
        
        return ctc_log_probs


def load_trained_model(checkpoint_path: str, device: str = 'cpu') -> tuple:
    """
    Load trained dual decoder model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        model: Loaded model
        config: Model configuration dict
        vocab: (char2idx, idx2char) vocabulary mappings
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract config
    config = checkpoint.get('config', {})
    input_dim = config.get('input_dim', 612)  # Default to v2 features
    hidden_dim = config.get('hidden_dim', 256)
    vocab_size = config.get('vocab_size', len(checkpoint.get('char2idx', {})))
    encoder_layers = config.get('encoder_layers', 4)
    decoder_layers = config.get('decoder_layers', 2)
    use_dual_decoder = config.get('use_dual_decoder', True)
    
    # Load vocabulary
    char2idx = checkpoint.get('char2idx', {})
    idx2char = checkpoint.get('idx2char', {})
    
    if vocab_size == 0:
        vocab_size = len(char2idx)
    
    print(f"Model config:")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Encoder layers: {encoder_layers}")
    print(f"  Dual decoder: {use_dual_decoder}")
    
    # Create model
    model = HybridCTCAttentionModelV2(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        primary_decoder='gru',
        use_subsampling=True,
        dual_decoder=use_dual_decoder
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    
    print(f"Model loaded successfully!")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, config, (char2idx, idx2char)


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_shape: tuple = (1, 150, 612),
    opset_version: int = 14,
    dynamic_axes: bool = True
):
    """
    Export model to ONNX format.
    
    Args:
        model: PyTorch model to export
        output_path: Path to save ONNX model
        input_shape: (batch, frames, features)
        opset_version: ONNX opset version
        dynamic_axes: Whether to use dynamic batch/sequence axes
    """
    print(f"\n📦 Exporting to ONNX: {output_path}")
    
    model.eval()
    
    # Create dummy inputs
    batch_size, max_frames, features = input_shape
    dummy_landmarks = torch.randn(batch_size, max_frames, features)
    
    # Configure dynamic axes
    dyn_axes = None
    if dynamic_axes:
        dyn_axes = {
            'landmarks': {0: 'batch', 1: 'frames'},
            'output': {0: 'batch'}
        }
    
    # Export
    torch.onnx.export(
        model,
        (dummy_landmarks,),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['landmarks'],
        output_names=['output'],
        dynamic_axes=dyn_axes
    )
    
    print(f"✅ ONNX model saved: {output_path}")
    
    # Verify if onnx is available
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX verification passed")
    except ImportError:
        print("⚠️  onnx package not installed, skipping verification")
    except Exception as e:
        print(f"⚠️  ONNX verification warning: {e}")


def export_ctc_streaming(
    model: HybridCTCAttentionModelV2,
    output_path: str,
    input_dim: int = 612
):
    """
    Export encoder-only model for streaming CTC inference.
    
    Args:
        model: Full dual decoder model
        output_path: Path to save ONNX model
        input_dim: Input feature dimension
    """
    print(f"\n🌊 Exporting streaming CTC encoder...")
    
    streaming_model = StreamingEncoderWrapper(model)
    streaming_model.eval()
    
    export_to_onnx(
        streaming_model,
        output_path,
        input_shape=(1, 60, input_dim),  # Shorter for streaming chunks
        dynamic_axes=True
    )
    
    print("✅ Streaming encoder exported")


def quantize_model(model: nn.Module) -> nn.Module:
    """
    Apply dynamic INT8 quantization to model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Quantized model
    """
    print("\n🔧 Applying dynamic quantization...")
    
    # Get original size
    def get_size_mb(m):
        param_size = sum(p.numel() * p.element_size() for p in m.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in m.buffers())
        return (param_size + buffer_size) / (1024 ** 2)
    
    original_size = get_size_mb(model)
    
    # Quantize
    quantized = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.LSTM, nn.GRU, nn.Conv1d},
        dtype=torch.qint8
    )
    
    quantized_size = get_size_mb(quantized)
    
    print(f"  Original: {original_size:.2f} MB")
    print(f"  Quantized: {quantized_size:.2f} MB")
    print(f"  Compression: {original_size/max(quantized_size, 0.01):.1f}x")
    
    return quantized


def benchmark_inference(
    model: nn.Module,
    input_shape: tuple = (1, 100, 612),
    device: str = 'cpu',
    num_iterations: int = 50,
    warmup: int = 10
) -> Dict[str, float]:
    """
    Benchmark model inference speed.
    
    Args:
        model: Model to benchmark
        input_shape: Input tensor shape
        device: Device for benchmarking
        num_iterations: Number of timed iterations
        warmup: Warmup iterations
        
    Returns:
        Dict with timing statistics
    """
    print(f"\n⚡ Benchmarking on {device}...")
    
    model.eval()
    model.to(device)
    
    dummy_input = torch.randn(*input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = model(dummy_input)
            if device == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
    
    avg_ms = np.mean(times)
    std_ms = np.std(times)
    p95_ms = np.percentile(times, 95)
    
    print(f"  Average: {avg_ms:.2f} ± {std_ms:.2f} ms")
    print(f"  P95: {p95_ms:.2f} ms")
    print(f"  FPS: {1000/avg_ms:.1f}")
    print(f"  Real-time ready (<100ms): {'✅' if avg_ms < 100 else '❌'}")
    
    return {
        'avg_ms': avg_ms,
        'std_ms': std_ms,
        'p95_ms': p95_ms,
        'fps': 1000 / avg_ms
    }


def export_dual_decoder_package(
    checkpoint_path: str,
    output_dir: str,
    export_gru: bool = True,
    export_transformer: bool = True,
    export_streaming: bool = True,
    export_quantized: bool = True
):
    """
    Export complete dual decoder model package for deployment.
    
    Args:
        checkpoint_path: Path to trained checkpoint
        output_dir: Directory to save exported models
        export_gru: Export GRU decoder variant
        export_transformer: Export Transformer decoder variant
        export_streaming: Export streaming CTC encoder
        export_quantized: Export quantized PyTorch models
    """
    print("="*60)
    print("DUAL DECODER MODEL EXPORT PIPELINE")
    print("="*60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model, config, (char2idx, idx2char) = load_trained_model(checkpoint_path)
    input_dim = config.get('input_dim', 612)
    
    results = {}
    
    # Export GRU decoder variant
    if export_gru:
        print("\n" + "-"*40)
        print("Exporting GRU Decoder (Fast)")
        print("-"*40)
        
        gru_wrapper = SingleDecoderWrapper(model, decoder_type='gru')
        
        # Benchmark
        results['gru'] = benchmark_inference(gru_wrapper, (1, 100, input_dim))
        
        # ONNX export
        onnx_gru_path = output_dir / "kortex_gru_decoder.onnx"
        try:
            export_to_onnx(gru_wrapper, str(onnx_gru_path), (1, 150, input_dim))
        except Exception as e:
            print(f"⚠️  GRU ONNX export failed: {e}")
        
        # Quantized PyTorch
        if export_quantized:
            quantized_gru = quantize_model(gru_wrapper)
            torch.save({
                'model_state_dict': quantized_gru.state_dict(),
                'decoder_type': 'gru',
                'config': config,
                'char2idx': char2idx,
                'idx2char': idx2char
            }, output_dir / "kortex_gru_quantized.pth")
    
    # Export Transformer decoder variant
    if export_transformer:
        print("\n" + "-"*40)
        print("Exporting Transformer Decoder (Accurate)")
        print("-"*40)
        
        tf_wrapper = SingleDecoderWrapper(model, decoder_type='transformer')
        
        # Benchmark
        results['transformer'] = benchmark_inference(tf_wrapper, (1, 100, input_dim))
        
        # ONNX export
        onnx_tf_path = output_dir / "kortex_transformer_decoder.onnx"
        try:
            export_to_onnx(tf_wrapper, str(onnx_tf_path), (1, 150, input_dim))
        except Exception as e:
            print(f"⚠️  Transformer ONNX export failed: {e}")
        
        # Quantized PyTorch
        if export_quantized:
            quantized_tf = quantize_model(tf_wrapper)
            torch.save({
                'model_state_dict': quantized_tf.state_dict(),
                'decoder_type': 'transformer',
                'config': config,
                'char2idx': char2idx,
                'idx2char': idx2char
            }, output_dir / "kortex_transformer_quantized.pth")
    
    # Export streaming CTC encoder
    if export_streaming:
        print("\n" + "-"*40)
        print("Exporting Streaming CTC Encoder")
        print("-"*40)
        
        streaming_path = output_dir / "kortex_streaming_ctc.onnx"
        try:
            export_ctc_streaming(model, str(streaming_path), input_dim)
        except Exception as e:
            print(f"⚠️  Streaming export failed: {e}")
    
    # Save full model (for fine-tuning)
    print("\n" + "-"*40)
    print("Saving Full Dual Decoder Model")
    print("-"*40)
    
    full_model_path = output_dir / "kortex_full_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'char2idx': char2idx,
        'idx2char': idx2char
    }, full_model_path)
    print(f"✅ Full model: {full_model_path}")
    
    # Save vocabulary
    vocab_path = output_dir / "vocabulary.json"
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump({
            'char2idx': char2idx,
            'idx2char': {str(k): v for k, v in idx2char.items()},
            'vocab_size': len(char2idx)
        }, f, indent=2, ensure_ascii=False)
    print(f"✅ Vocabulary: {vocab_path}")
    
    # Save deployment info
    deployment_info = {
        'model_name': 'Kortex ISL Recognition',
        'version': '2.0',
        'architecture': 'Dual Decoder CTC-Attention',
        'input_dim': input_dim,
        'feature_version': config.get('feature_version', 'v2'),
        'vocab_size': len(char2idx),
        'decoders': {
            'gru': {
                'description': 'Fast decoder for real-time streaming',
                'use_case': 'Mobile apps, real-time transcription',
                'latency_ms': results.get('gru', {}).get('avg_ms', 'N/A')
            },
            'transformer': {
                'description': 'Accurate decoder for polished output',
                'use_case': 'Offline processing, high-accuracy needs',
                'latency_ms': results.get('transformer', {}).get('avg_ms', 'N/A')
            }
        },
        'files': {
            'full_model': 'kortex_full_model.pth',
            'gru_quantized': 'kortex_gru_quantized.pth',
            'transformer_quantized': 'kortex_transformer_quantized.pth',
            'gru_onnx': 'kortex_gru_decoder.onnx',
            'transformer_onnx': 'kortex_transformer_decoder.onnx',
            'streaming_onnx': 'kortex_streaming_ctc.onnx',
            'vocabulary': 'vocabulary.json'
        },
        'input_format': {
            'landmarks': 'MediaPipe pose (33) + hands (21x2) + mouth (20) + head_pose (6)',
            'derivatives': 'velocity + acceleration',
            'total_dims': input_dim,
            'frame_rate': 30
        }
    }
    
    info_path = output_dir / "deployment_info.json"
    with open(info_path, 'w') as f:
        json.dump(deployment_info, f, indent=2)
    print(f"✅ Deployment info: {info_path}")
    
    print("\n" + "="*60)
    print("EXPORT COMPLETE!")
    print(f"All files saved to: {output_dir}")
    print("="*60)
    
    return results


def main():
    """Main export function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Export Kortex dual decoder model')
    parser.add_argument('--checkpoint', type=str, 
                        default='e:/5thsem el/kortex_5th_sem/sign_to_text/checkpoints_hybrid/best_model.pth',
                        help='Path to checkpoint')
    parser.add_argument('--output', type=str,
                        default='e:/5thsem el/kortex_5th_sem/exported_models',
                        help='Output directory')
    parser.add_argument('--no-gru', action='store_true', help='Skip GRU export')
    parser.add_argument('--no-transformer', action='store_true', help='Skip Transformer export')
    parser.add_argument('--no-streaming', action='store_true', help='Skip streaming export')
    parser.add_argument('--no-quantize', action='store_true', help='Skip quantization')
    
    args = parser.parse_args()
    
    export_dual_decoder_package(
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        export_gru=not args.no_gru,
        export_transformer=not args.no_transformer,
        export_streaming=not args.no_streaming,
        export_quantized=not args.no_quantize
    )


if __name__ == "__main__":
    main()
