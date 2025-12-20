"""
Export Text-to-Sign Model to Mobile Formats
Converts trained PyTorch model to ONNX and TFLite
"""

import torch
import torch.nn as nn
import onnx
from pathlib import Path
import sys
import json

# Import from same directory
from model_text_to_sign import create_text_to_sign_model


def export_to_onnx(model, output_path, vocab_size=5004, max_frames=150):
    """
    Export model to ONNX format
    
    Args:
        model: Trained PyTorch model
        output_path: Path to save ONNX model
        vocab_size: Vocabulary size
        max_frames: Maximum frame length
    """
    model.eval()
    
    # Create dummy inputs
    dummy_text = torch.randint(1, vocab_size, (1, 10), dtype=torch.long)
    dummy_landmarks = torch.randn(1, max_frames, 138)
    
    # Export
    print(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        model,
        (dummy_text, dummy_landmarks),
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['text_tokens', 'target_landmarks'],
        output_names=['generated_landmarks'],
        dynamic_axes={
            'text_tokens': {0: 'batch', 1: 'text_len'},
            'target_landmarks': {0: 'batch', 1: 'frames'},
            'generated_landmarks': {0: 'batch', 1: 'frames'}
        }
    )
    
    # Verify
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX model verified")
    
    # Get file size
    size_mb = Path(output_path).stat().st_size / (1024 ** 2)
    print(f"Model size: {size_mb:.2f} MB")


def quantize_model(model):
    """
    Apply dynamic quantization to model (INT8)
    
    Args:
        model: PyTorch model
    
    Returns:
        quantized_model: Quantized model
    """
    print("Applying dynamic quantization...")
    
    quantized = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Embedding},
        dtype=torch.qint8
    )
    
    # Calculate size reduction
    original_params = sum(p.numel() * 4 for p in model.parameters()) / (1024 ** 2)
    quantized_params = sum(p.numel() for p in quantized.parameters() if p.dtype == torch.qint8) / (1024 ** 2)
    
    print(f"Original size: ~{original_params:.2f} MB")
    print(f"Quantized size: ~{quantized_params:.2f} MB")
    print(f"Reduction: {(1 - quantized_params/original_params)*100:.1f}%")
    
    return quantized


def export_to_tflite(onnx_path, output_path):
    """
    Convert ONNX to TFLite (requires onnx-tf and tensorflow)
    
    Args:
        onnx_path: Path to ONNX model
        output_path: Path to save TFLite model
    """
    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf
        
        print(f"Converting ONNX to TFLite: {onnx_path} -> {output_path}")
        
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        
        # Convert to TensorFlow
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph('temp_tf_model')
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model('temp_tf_model')
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        # Save
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        size_mb = Path(output_path).stat().st_size / (1024 ** 2)
        print(f"✅ TFLite model saved: {size_mb:.2f} MB")
        
        # Clean up temp files
        import shutil
        if Path('temp_tf_model').exists():
            shutil.rmtree('temp_tf_model')
        
    except ImportError as e:
        print(f"⚠️ TFLite conversion requires: onnx-tf, tensorflow")
        print(f"Install with: pip install onnx-tf tensorflow")
        print(f"Error: {e}")


def benchmark_inference(model, device='cpu', num_runs=100):
    """
    Benchmark inference speed
    
    Args:
        model: PyTorch model
        device: Device to run on
        num_runs: Number of inference runs
    """
    import time
    
    model.eval()
    model = model.to(device)
    
    # Dummy input
    dummy_text = torch.randint(1, 5000, (1, 10), dtype=torch.long).to(device)
    
    # Warmup
    print(f"\nBenchmarking on {device}...")
    with torch.no_grad():
        for _ in range(10):
            _ = model.generate(dummy_text, max_frames=60)
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model.generate(dummy_text, max_frames=60)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"Average inference time: {avg_time:.2f} ms")
    print(f"Min: {min_time:.2f} ms, Max: {max_time:.2f} ms")
    print(f"Throughput: ~{1000/avg_time:.1f} generations/sec")


def main():
    print("Text-to-Sign Model Export Pipeline")
    print("="*60)
    
    # Paths
    model_dir = Path("experiment/models")
    checkpoint_path = model_dir / "best_text_to_sign.pth"
    onnx_path = model_dir / "text_to_sign.onnx"
    onnx_quantized_path = model_dir / "text_to_sign_quantized.onnx"
    tflite_path = model_dir / "text_to_sign.tflite"
    
    # Check if checkpoint exists
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Please train the model first using train_text_to_sign.py")
        return
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    
    print(f"Checkpoint info:")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Val loss: {checkpoint['val_loss']:.4f}")
    print(f"  Config: {json.dumps(config, indent=2)}")
    
    # Create model
    print("\nCreating model...")
    model = create_text_to_sign_model(
        vocab_size=config['vocab_size'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Export to ONNX
    print("\n1. Exporting to ONNX...")
    export_to_onnx(model, onnx_path, vocab_size=config['vocab_size'])
    
    # Quantize and export
    print("\n2. Quantizing model...")
    quantized_model = quantize_model(model)
    export_to_onnx(quantized_model, onnx_quantized_path, vocab_size=config['vocab_size'])
    
    # Export to TFLite (optional)
    print("\n3. Exporting to TFLite...")
    export_to_tflite(onnx_quantized_path, tflite_path)
    
    # Benchmark
    print("\n4. Benchmarking performance...")
    
    print("\nOriginal model (FP32):")
    benchmark_inference(model, device='cpu', num_runs=50)
    
    print("\nQuantized model (INT8):")
    benchmark_inference(quantized_model, device='cpu', num_runs=50)
    
    # Summary
    print("\n" + "="*60)
    print("Export Summary:")
    print(f"✅ PyTorch checkpoint: {checkpoint_path}")
    print(f"✅ ONNX (FP32): {onnx_path}")
    print(f"✅ ONNX (INT8): {onnx_quantized_path}")
    if tflite_path.exists():
        print(f"✅ TFLite: {tflite_path}")
    print("\nNext steps:")
    print("1. Test generation with: python experiment/test_generation.py")
    print("2. Integrate TFLite model into Flutter app")
    print("3. Implement 2D skeleton renderer in Flutter")


if __name__ == "__main__":
    main()
