"""
Enhanced Preprocessing Pipeline for Sign Language Landmarks
Adds normalization, velocity, acceleration features for better recognition

Features:
- Wrist-centered normalization (translation invariant)
- Hand-span scaling (scale invariant) 
- Velocity features (motion dynamics)
- Acceleration features (gesture sharpness)
- Gaussian smoothing (reduces MediaPipe jitter)
- Robust handling of variable-length and long sequences
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
from tqdm import tqdm
import os
from multiprocessing import Pool, cpu_count


# ==================== CONFIGURATION ====================
class PreprocessConfig:
    # Paths
    input_dir = "E:/5thsem el/output"  # Raw MediaPipe landmarks
    output_dir = "E:/5thsem el/output_preprocessed"  # Preprocessed features
    
    # Feature dimensions
    raw_dim = 138  # 2 hands × 21 × 3 + 2 shoulders × 3 + 2 elbows × 3
    
    # Preprocessing parameters
    smoothing_sigma = 1.0  # Gaussian smoothing for jitter reduction
    min_hand_span = 0.01  # Minimum hand span to avoid division by zero
    
    # Augmentation (applied during extraction)
    add_velocity = True
    add_acceleration = True
    
    # Multi-processing
    num_workers = max(1, cpu_count() - 2)


# ==================== PREPROCESSING FUNCTIONS ====================

def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalize landmarks to be person and camera invariant
    
    Args:
        landmarks: (T, 138) raw MediaPipe output
        
    Returns:
        normalized: (T, 138) normalized landmarks
    """
    T = landmarks.shape[0]
    
    # Reshape for processing
    # Hands: 2 hands × 21 landmarks × 3 coords = 126
    # Shoulders: 2 × 3 = 6
    # Elbows: 2 × 3 = 6
    hands = landmarks[:, :126].reshape(T, 2, 21, 3).copy()
    shoulders = landmarks[:, 126:132].reshape(T, 2, 3).copy()
    elbows = landmarks[:, 132:138].reshape(T, 2, 3).copy()
    
    # ===== 1. Normalize each hand to wrist origin =====
    # Wrist is landmark index 0 for each hand
    for h in range(2):
        wrist = hands[:, h, 0:1, :].copy()  # (T, 1, 3)
        hands[:, h] = hands[:, h] - wrist  # Center to wrist
    
    # ===== 2. Scale by hand span =====
    # Hand span = distance from wrist (0) to middle fingertip (12)
    for h in range(2):
        # Calculate span for each frame
        span = np.linalg.norm(hands[:, h, 12] - hands[:, h, 0], axis=1, keepdims=True)  # (T, 1)
        span = np.maximum(span, PreprocessConfig.min_hand_span)  # Avoid division by zero
        
        # Scale all landmarks
        hands[:, h] = hands[:, h] / span[:, :, np.newaxis]
    
    # ===== 3. Normalize shoulders/elbows relative to body center =====
    body_center = shoulders.mean(axis=1, keepdims=True)  # (T, 1, 3)
    shoulders = shoulders - body_center
    elbows = elbows - body_center
    
    # ===== 4. Scale body landmarks by shoulder width =====
    shoulder_width = np.linalg.norm(shoulders[:, 0] - shoulders[:, 1], axis=1, keepdims=True)  # (T, 1)
    shoulder_width = np.maximum(shoulder_width, PreprocessConfig.min_hand_span)
    shoulders = shoulders / shoulder_width[:, :, np.newaxis]
    elbows = elbows / shoulder_width[:, :, np.newaxis]
    
    # ===== 5. Flatten back to (T, 138) =====
    normalized = np.concatenate([
        hands.reshape(T, -1),       # 126
        shoulders.reshape(T, -1),   # 6
        elbows.reshape(T, -1)       # 6
    ], axis=1)
    
    return normalized


def compute_velocity(features: np.ndarray) -> np.ndarray:
    """
    Compute velocity (temporal derivative) of features
    
    Args:
        features: (T, D) feature sequence
        
    Returns:
        velocity: (T, D) velocity features (first frame is zero)
    """
    velocity = np.zeros_like(features)
    velocity[1:] = features[1:] - features[:-1]
    return velocity


def compute_acceleration(velocity: np.ndarray) -> np.ndarray:
    """
    Compute acceleration (second temporal derivative)
    
    Args:
        velocity: (T, D) velocity sequence
        
    Returns:
        acceleration: (T, D) acceleration features
    """
    acceleration = np.zeros_like(velocity)
    acceleration[1:] = velocity[1:] - velocity[:-1]
    return acceleration


def smooth_features(features: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Apply Gaussian smoothing to reduce MediaPipe jitter
    
    Args:
        features: (T, D) feature sequence
        sigma: Gaussian kernel standard deviation
        
    Returns:
        smoothed: (T, D) smoothed features
    """
    if sigma <= 0 or features.shape[0] < 3:
        return features
    
    return gaussian_filter1d(features, sigma=sigma, axis=0, mode='nearest')


def preprocess_single_file(landmarks: np.ndarray, 
                           add_velocity: bool = True,
                           add_acceleration: bool = True,
                           smooth: bool = True) -> np.ndarray:
    """
    Complete preprocessing pipeline for a single landmark sequence
    
    Args:
        landmarks: (T, 138) raw MediaPipe landmarks
        add_velocity: Whether to add velocity features
        add_acceleration: Whether to add acceleration features
        smooth: Whether to apply Gaussian smoothing
        
    Returns:
        features: (T, D) preprocessed features
            D = 138 (base) + 138 (velocity) + 138 (acceleration) = 414 if all enabled
    """
    if landmarks.shape[0] == 0:
        return landmarks
    
    # Handle edge case of very short sequences
    if landmarks.shape[0] < 3:
        # Pad to minimum length
        pad_length = 3 - landmarks.shape[0]
        landmarks = np.pad(landmarks, ((0, pad_length), (0, 0)), mode='edge')
    
    # 1. Normalize landmarks
    normalized = normalize_landmarks(landmarks)
    
    # 2. Apply smoothing to normalized landmarks
    if smooth:
        normalized = smooth_features(normalized, sigma=PreprocessConfig.smoothing_sigma)
    
    # 3. Compute velocity
    if add_velocity:
        velocity = compute_velocity(normalized)
        if smooth:
            velocity = smooth_features(velocity, sigma=PreprocessConfig.smoothing_sigma)
    
    # 4. Compute acceleration
    if add_acceleration and add_velocity:
        acceleration = compute_acceleration(velocity)
        if smooth:
            acceleration = smooth_features(acceleration, sigma=PreprocessConfig.smoothing_sigma)
    
    # 5. Concatenate all features
    features = [normalized]
    if add_velocity:
        features.append(velocity)
    if add_acceleration and add_velocity:
        features.append(acceleration)
    
    return np.concatenate(features, axis=1)


def get_feature_dim(add_velocity: bool = True, add_acceleration: bool = True) -> int:
    """Get output feature dimension based on settings"""
    dim = 138  # Base landmarks
    if add_velocity:
        dim += 138
    if add_acceleration and add_velocity:
        dim += 138
    return dim


# ==================== BATCH PROCESSING ====================

def process_single_video(args):
    """Process a single .npy file"""
    input_path, output_path = args
    
    try:
        # Load raw landmarks
        landmarks = np.load(input_path).astype(np.float32)
        
        if landmarks.shape[0] == 0:
            return f"⚠️ Empty file: {input_path.name}"
        
        # Preprocess
        features = preprocess_single_file(
            landmarks,
            add_velocity=PreprocessConfig.add_velocity,
            add_acceleration=PreprocessConfig.add_acceleration,
            smooth=True
        )
        
        # Save preprocessed features
        np.save(output_path, features.astype(np.float32))
        
        return None
        
    except Exception as e:
        return f"❌ Error processing {input_path.name}: {str(e)}"


def preprocess_all_landmarks():
    """Preprocess all landmark files in input directory"""
    config = PreprocessConfig()
    
    input_dir = Path(config.input_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all .npy files
    npy_files = sorted(list(input_dir.glob("*.npy")))
    print(f"Found {len(npy_files)} landmark files")
    
    # Prepare tasks (skip already processed)
    tasks = []
    for input_path in npy_files:
        output_path = output_dir / input_path.name
        if not output_path.exists():
            tasks.append((input_path, output_path))
    
    print(f"Files to process: {len(tasks)} (skipping {len(npy_files) - len(tasks)} already done)")
    
    if len(tasks) == 0:
        print("✅ All files already preprocessed!")
        return
    
    # Process with multiprocessing
    print(f"Using {config.num_workers} workers...")
    
    with Pool(processes=config.num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_video, tasks, chunksize=32),
            total=len(tasks),
            desc="Preprocessing"
        ))
    
    # Print errors/warnings
    errors = [r for r in results if r is not None]
    if errors:
        print(f"\n⚠️ {len(errors)} files had issues:")
        for err in errors[:10]:
            print(f"  {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
    
    # Verify output
    output_files = list(output_dir.glob("*.npy"))
    print(f"\n✅ Preprocessing complete!")
    print(f"   Output directory: {output_dir}")
    print(f"   Total files: {len(output_files)}")
    
    # Show sample stats
    if output_files:
        sample = np.load(output_files[0])
        print(f"   Feature dimension: {sample.shape[1]}")
        print(f"   Sample sequence length: {sample.shape[0]}")


# ==================== ONLINE PREPROCESSING (for inference) ====================

class OnlinePreprocessor:
    """
    Real-time preprocessing for streaming inference
    Maintains history for velocity/acceleration computation
    """
    
    def __init__(self, add_velocity: bool = True, add_acceleration: bool = True):
        self.add_velocity = add_velocity
        self.add_acceleration = add_acceleration
        
        self.prev_frame = None
        self.prev_velocity = None
        
    def reset(self):
        """Reset state for new sequence"""
        self.prev_frame = None
        self.prev_velocity = None
    
    def process_frame(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Process a single frame in streaming mode
        
        Args:
            landmarks: (138,) raw MediaPipe landmarks for one frame
            
        Returns:
            features: (D,) preprocessed features
        """
        # Reshape to (1, 138) for processing
        landmarks = landmarks.reshape(1, -1)
        
        # Normalize
        normalized = normalize_landmarks(landmarks)[0]  # (138,)
        
        # Compute velocity
        if self.add_velocity:
            if self.prev_frame is not None:
                velocity = normalized - self.prev_frame
            else:
                velocity = np.zeros_like(normalized)
        
        # Compute acceleration
        if self.add_acceleration and self.add_velocity:
            if self.prev_velocity is not None:
                acceleration = velocity - self.prev_velocity
            else:
                acceleration = np.zeros_like(velocity)
        
        # Update history
        self.prev_frame = normalized.copy()
        if self.add_velocity:
            self.prev_velocity = velocity.copy()
        
        # Concatenate features
        features = [normalized]
        if self.add_velocity:
            features.append(velocity)
        if self.add_acceleration and self.add_velocity:
            features.append(acceleration)
        
        return np.concatenate(features)
    
    def process_batch(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Process a batch of frames (for buffered inference)
        
        Args:
            landmarks: (T, 138) landmark sequence
            
        Returns:
            features: (T, D) preprocessed features
        """
        self.reset()
        features = []
        for i in range(landmarks.shape[0]):
            feat = self.process_frame(landmarks[i])
            features.append(feat)
        return np.stack(features)


# ==================== UTILITIES ====================

def analyze_dataset_stats():
    """Analyze statistics of the preprocessed dataset"""
    config = PreprocessConfig()
    output_dir = Path(config.output_dir)
    
    npy_files = list(output_dir.glob("*.npy"))
    
    if not npy_files:
        print("No preprocessed files found!")
        return
    
    lengths = []
    dims = []
    
    print("Analyzing dataset...")
    for f in tqdm(npy_files[:1000]):  # Sample first 1000
        data = np.load(f)
        lengths.append(data.shape[0])
        dims.append(data.shape[1])
    
    lengths = np.array(lengths)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Files sampled: {len(lengths)}")
    print(f"   Feature dimension: {dims[0]}")
    print(f"   Sequence lengths:")
    print(f"     Min: {lengths.min()}")
    print(f"     Max: {lengths.max()}")
    print(f"     Mean: {lengths.mean():.1f}")
    print(f"     Median: {np.median(lengths):.1f}")
    print(f"     Std: {lengths.std():.1f}")
    print(f"   Percentiles:")
    print(f"     10%: {np.percentile(lengths, 10):.0f}")
    print(f"     50%: {np.percentile(lengths, 50):.0f}")
    print(f"     90%: {np.percentile(lengths, 90):.0f}")
    print(f"     95%: {np.percentile(lengths, 95):.0f}")
    print(f"     99%: {np.percentile(lengths, 99):.0f}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        analyze_dataset_stats()
    else:
        preprocess_all_landmarks()
