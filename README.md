# Kortex - Bidirectional Sign Language Translation System

Complete sign language translation system with **sign→text** and **text→sign** capabilities, optimized for mobile deployment.

## 📁 Project Structure

```
kortex_5th_sem/
├── sign_to_text/              # Sign → Text Recognition (Main Pipeline)
│   ├── model_mobile.py        # CNN-BiGRU model architecture
│   ├── vocabulary_builder.py  # Vocabulary management (5K words)
│   ├── train.py              # Training script (optimized for RTX 4070)
│   ├── export.py             # Export to ONNX/TFLite
│   ├── test_realtime.py      # Webcam real-time testing
│   └── checkpoints/          # Saved models (created during training)
│
├── text_to_sign/              # Text → Sign Generation (Experimental)
│   ├── model_text_to_sign.py # Transformer encoder-decoder
│   ├── train_text_to_sign.py # Training script (optimized for RTX 4060)
│   ├── export_text_to_sign.py# Export to mobile formats
│   ├── test_generation.py    # Generate & visualize signs
│   ├── EXPERIMENT_PLAN.md    # Complete experimental guide
│   └── checkpoints/          # Saved models (created during training)
│
├── data/                      # Dataset & Processing
│   ├── iSign_v1.1.csv        # Dataset metadata (127K samples)
│   ├── extract_landmarks.py  # Multi-core landmark extraction
│   └── number_of_frames_test.py
│
├── docs/                      # Documentation
│   └── README_SIGN_TO_TEXT.md # Detailed sign→text guide
│
├── SURYA/                     # Original experimental scripts
│   └── (legacy code)
│
└── kortex/                    # Python virtual environment
```

---

## 🚀 Quick Start

### Prerequisites
```bash
# Ensure CUDA is available
nvcc --version

# Activate environment (if using venv)
cd kortex_5th_sem
.\kortex\Scripts\Activate.ps1

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas opencv-python mediapipe tqdm matplotlib
```

---

## 📊 Dataset Setup

**Dataset**: iSign v1.1 (Indian Sign Language)
- **Location**: `data/iSign_v1.1.csv`
- **Size**: 127,237 full-sentence samples
- **Landmarks**: Must be extracted first (see below)

### Extract Landmarks (One-time)
```bash
cd data
python extract_landmarks.py
```
**Output**: `.npy` files in `E:/5thsem el/output/` (138 features per frame)

---

## 🎯 Training Both Models (Dual-GPU Setup)

### Machine 1: RTX 4070 (12GB) - Sign → Text
```bash
cd sign_to_text

# 1. Build vocabulary (first time only)
python vocabulary_builder.py

# 2. Train model (2-3 hours with batch_size=64)
python train.py

# 3. Export to mobile
python export.py

# 4. Test with webcam
python test_realtime.py
```

**Model**: CNN-BiGRU, ~1.4-1.6 MB quantized, <100ms inference

---

### Machine 2: RTX 4060 (8GB) - Text → Sign
```bash
cd text_to_sign

# 1. Train model (3-4 hours with batch_size=20, mixed precision)
python train_text_to_sign.py

# 2. Export to mobile
python export_text_to_sign.py

# 3. Test generation
python test_generation.py
```

**Model**: Transformer, ~8-10 MB quantized, ~100ms inference

---

## ⚙️ Configuration

### Sign → Text (`sign_to_text/train.py`)
```python
class Config:
    csv_path = "../data/iSign_v1.1.csv"
    landmarks_dir = "E:/5thsem el/output"  # Update to your path
    vocab_path = "vocabulary.pkl"
    output_dir = "checkpoints"
    
    hidden_dim = 128
    batch_size = 64  # For RTX 4070 (use 32 for 4060)
    num_epochs = 40
    learning_rate = 1e-3
```

### Text → Sign (`text_to_sign/train_text_to_sign.py`)
```python
class Config:
    CSV_PATH = "../data/iSign_v1.1.csv"
    LANDMARKS_DIR = "E:/5thsem el/output"
    VOCAB_PATH = "../sign_to_text/vocabulary.pkl"
    MODEL_SAVE_DIR = "checkpoints"
    
    HIDDEN_DIM = 256
    BATCH_SIZE = 20  # Optimized for RTX 4060
    NUM_EPOCHS = 50
    USE_AMP = True  # Mixed precision (30% faster)
```

---

## 🔥 Optimizations Applied

### Sign → Text (4070)
- ✅ BiGRU instead of BiLSTM (25% fewer params, 20% faster)
- ✅ Batch size: 64 (utilize 12GB VRAM)
- ✅ num_workers: 4, pin_memory: True

### Text → Sign (4060)
- ✅ Mixed precision training (30% speedup)
- ✅ Batch size: 20 (optimal for 8GB VRAM)
- ✅ Temporal smoothing loss (natural animations)
- ✅ num_workers: 2, pin_memory: True

---

## 📈 Expected Training Times

| Model         | GPU        | Time      | Final Size |
|---------------|------------|-----------|------------|
| Sign→Text     | RTX 4070   | 2-3 hours | 1.6 MB     |
| Text→Sign     | RTX 4060   | 3-4 hours | 8-10 MB    |

**Total**: ~6 hours for both models trained simultaneously!

---

## 🎮 Testing

### Sign → Text (Webcam)
```bash
cd sign_to_text
python test_realtime.py
```
- Opens webcam
- Real-time hand tracking with MediaPipe
- Streaming text predictions (GPT-style)
- Press 'q' to quit

### Text → Sign (Generate Animations)
```bash
cd text_to_sign
python test_generation.py
```
- Interactive text input
- Generates 2D skeleton sequences
- Saves as .npy files
- Displays first frame visualization

---

## 📱 Mobile Deployment

### Flutter Integration

1. **Copy TFLite models**:
   ```
   sign_to_text/checkpoints/sign_to_text_quantized.tflite
   text_to_sign/checkpoints/text_to_sign.tflite
   ```

2. **Add to Flutter** (`pubspec.yaml`):
   ```yaml
   dependencies:
     tflite_flutter: ^0.10.0
   
   flutter:
     assets:
       - assets/models/sign_to_text_quantized.tflite
       - assets/models/text_to_sign.tflite
   ```

3. **Load & Run**:
   ```dart
   import 'package:tflite_flutter/tflite_flutter.dart';
   
   final interpreter = await Interpreter.fromAsset(
     'assets/models/sign_to_text_quantized.tflite'
   );
   ```

See `docs/README_SIGN_TO_TEXT.md` and `text_to_sign/EXPERIMENT_PLAN.md` for complete Flutter code samples.

---

## 🐛 Troubleshooting

### CUDA Out of Memory
- **4070**: Reduce batch_size to 32
- **4060**: Reduce batch_size to 16 or HIDDEN_DIM to 192

### Vocabulary Not Found
```bash
cd sign_to_text
python vocabulary_builder.py
```

### Landmarks Not Found
```bash
cd data
python extract_landmarks.py
```

### Import Errors
All imports are now relative within each folder - no need for `NEW_TRAIN` or `experiment` paths.

---

## 📚 Documentation

- **Sign → Text**: `docs/README_SIGN_TO_TEXT.md`
- **Text → Sign**: `text_to_sign/EXPERIMENT_PLAN.md`
- **Dataset**: iSign v1.1 (127K Indian Sign Language samples)

---

## 🎓 Credits

- **Dataset**: iSign v1.1 (Indian Sign Language)
- **MediaPipe**: Hand & body landmark extraction
- **PyTorch**: Model training & inference
- **Flutter**: Mobile deployment (TFLite)

---

## 🚦 Status

| Component       | Status | Notes                          |
|-----------------|--------|--------------------------------|
| Landmark Extraction | ✅ | Multi-core, 127K videos processed |
| Sign→Text Model | ✅ | CNN-BiGRU, ready to train      |
| Text→Sign Model | ✅ | Transformer, ready to train    |
| Vocabulary      | ✅ | 5K words, character fallback   |
| Export Pipeline | ✅ | ONNX, TFLite, quantization     |
| Documentation   | ✅ | Complete guides                |

**Ready for dual-GPU training! 🚀**

---

**Start training both models now - they'll be done in ~6 hours!**
