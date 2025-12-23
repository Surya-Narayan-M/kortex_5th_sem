"""
Model Export Utilities
Exports trained PyTorch model to ONNX and TFLite formats for mobile deployment
"""

import torch
import torch.onnx
import numpy as np
from pathlib import Path

from model_mobile import create_mobile_model
from vocabulary_builder import VocabularyBuilder


def export_to_onnx(model, output_path, input_shape=(1, 150, 138)):
    """
    Export PyTorch model to ONNX format
    
    Args:
        model: PyTorch model
        output_path: Path to save ONNX model
        input_shape: Input tensor shape (batch, frames, features)
    """
    print(f"\n📦 Exporting to ONNX: {output_path}")
    
    model.eval()
    
    # Create dummy input
    batch_size, max_frames, features = input_shape
    dummy_landmarks = torch.randn(batch_size, max_frames, features)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_landmarks,),  # Model input
        output_path,
        export_params=True,
        opset_version=13,  # Use stable opset
        do_constant_folding=True,
        input_names=['landmarks'],
        output_names=['predictions'],
        dynamic_axes={
            'landmarks': {0: 'batch', 1: 'frames'},
            'predictions': {0: 'batch', 1: 'frames'}
        }
    )
    
    print(f"✅ ONNX model saved to {output_path}")
    
    # Verify ONNX model
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model verification passed")
    except ImportError:
        print("⚠️  onnx package not installed, skipping verification")
    except Exception as e:
        print(f"⚠️  ONNX verification failed: {e}")


def quantize_model(model):
    """
    Apply dynamic quantization to reduce model size
    
    Args:
        model: PyTorch model
    
    Returns:
        quantized_model: Quantized model
    """
    print("\n🔧 Applying dynamic quantization...")
    
    # Quantize Linear and LSTM layers to INT8
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.LSTM, torch.nn.Conv1d},
        dtype=torch.qint8
    )
    
    # Calculate size reduction
    def get_size_mb(model):
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / (1024 ** 2)
    
    original_size = get_size_mb(model)
    quantized_size = get_size_mb(quantized_model)
    
    print(f"Original model size: {original_size:.2f} MB")
    print(f"Quantized model size: {quantized_size:.2f} MB")
    print(f"Compression ratio: {original_size/quantized_size:.2f}x")
    
    return quantized_model


def export_to_tflite(onnx_path, output_path):
    """
    Convert ONNX model to TensorFlow Lite format
    
    Note: Requires onnx-tf and tensorflow packages
    
    Args:
        onnx_path: Path to ONNX model
        output_path: Path to save TFLite model
    """
    print(f"\n📱 Converting to TFLite: {output_path}")
    
    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf
        
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        
        # Convert to TensorFlow
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph('temp_tf_model')
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model('temp_tf_model')
        
        # Optimization flags
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]  # FP16 quantization
        
        tflite_model = converter.convert()
        
        # Save TFLite model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✅ TFLite model saved to {output_path}")
        print(f"TFLite model size: {len(tflite_model) / (1024**2):.2f} MB")
        
        # Cleanup
        import shutil
        shutil.rmtree('temp_tf_model', ignore_errors=True)
        
    except ImportError as e:
        print(f"⚠️  TFLite conversion requires onnx-tf and tensorflow packages")
        print(f"Install with: pip install onnx-tf tensorflow")
        print(f"Error: {e}")
    except Exception as e:
        print(f"❌ TFLite conversion failed: {e}")


def benchmark_inference(model, device='cpu', num_iterations=100):
    """
    Benchmark model inference speed
    
    Args:
        model: PyTorch model
        device: Device to run on
        num_iterations: Number of iterations for benchmarking
    """
    print(f"\n⚡ Benchmarking inference speed on {device}...")
    
    model.eval()
    model.to(device)
    
    # Create dummy input
    dummy_input = torch.randn(1, 60, 138).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Benchmark
    import time
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)
            if device == 'cuda':
                torch.cuda.synchronize()
    
    end_time = time.time()
    
    avg_time_ms = (end_time - start_time) / num_iterations * 1000
    fps = 1000 / avg_time_ms
    
    print(f"Average inference time: {avg_time_ms:.2f} ms")
    print(f"Throughput: {fps:.2f} FPS")
    print(f"✅ Target for real-time (<100ms): {'PASS' if avg_time_ms < 100 else 'FAIL'}")


def export_model_package(checkpoint_path, output_dir, vocab_path):
    """
    Export complete model package for deployment
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        output_dir: Directory to save exported models
        vocab_path: Path to vocabulary JSON
    """
    print("="*60)
    print("MODEL EXPORT PIPELINE")
    print("="*60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    vocab_size = checkpoint['vocab_size']
    hidden_dim = checkpoint['hidden_dim']
    
    # Create model
    model = create_mobile_model(vocab_size=vocab_size, hidden_dim=hidden_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Parameters: {model.get_num_params():,}")
    print(f"  Size: {model.get_model_size_mb():.2f} MB")
    
    # Benchmark
    benchmark_inference(model, device='cpu')
    
    # Export PyTorch model
    pytorch_path = output_dir / "sign_language_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'hidden_dim': hidden_dim,
    }, pytorch_path)
    print(f"\n✅ PyTorch model: {pytorch_path}")
    
    # Quantize and save
    quantized_model = quantize_model(model)
    quantized_path = output_dir / "sign_language_model_quantized.pth"
    torch.save({
        'model_state_dict': quantized_model.state_dict(),
        'vocab_size': vocab_size,
        'hidden_dim': hidden_dim,
    }, quantized_path)
    print(f"✅ Quantized model: {quantized_path}")
    
    # Export to ONNX
    onnx_path = output_dir / "sign_language_model.onnx"
    export_to_onnx(model, onnx_path)
    
    # Export to TFLite (optional, requires dependencies)
    tflite_path = output_dir / "sign_language_model.tflite"
    export_to_tflite(str(onnx_path), str(tflite_path))
    
    # Copy vocabulary
    import shutil
    vocab_dest = output_dir / "vocabulary.json"
    shutil.copy(vocab_path, vocab_dest)
    print(f"\n✅ Vocabulary copied: {vocab_dest}")
    
    # Create deployment info
    deployment_info = {
        'model_type': 'Mobile Sign Language Recognition',
        'architecture': 'CNN-BiLSTM + CTC',
        'vocab_size': vocab_size,
        'hidden_dim': hidden_dim,
        'input_shape': [None, 150, 138],  # (batch, frames, features)
        'output_shape': [None, 150, vocab_size],  # (batch, frames, vocab)
        'files': {
            'pytorch': str(pytorch_path.name),
            'pytorch_quantized': str(quantized_path.name),
            'onnx': str(onnx_path.name),
            'tflite': str(tflite_path.name),
            'vocabulary': str(vocab_dest.name)
        },
        'inference': {
            'expected_fps': '10-20 FPS on mid-range mobile',
            'latency': '50-100ms per inference'
        }
    }
    
    import json
    info_path = output_dir / "deployment_info.json"
    with open(info_path, 'w') as f:
        json.dump(deployment_info, f, indent=2)
    print(f"✅ Deployment info: {info_path}")
    
    print("\n" + "="*60)
    print("EXPORT COMPLETED!")
    print(f"All files saved to: {output_dir}")
    print("="*60)


def main():
    """Main export function"""
    # Paths
    checkpoint_path = "E:/5thsem el/kortex_5th_sem/models/best_model.pth"
    output_dir = "E:/5thsem el/kortex_5th_sem/exported_models"
    vocab_path = "E:/5thsem el/kortex_5th_sem/vocabulary.json"
    
    # Export
    export_model_package(checkpoint_path, output_dir, vocab_path)


if __name__ == "__main__":
    main()
