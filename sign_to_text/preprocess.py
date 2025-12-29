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
    # Paths - V1 (138 dims: hands + body only)
    input_dir = "E:/5thsem el/output"  # Raw MediaPipe landmarks
    output_dir = "E:/5thsem el/output_preprocessed"  # Preprocessed features
    
    # Paths - V2 (204 dims: hands + body + mouth + head pose)
    input_dir_v2 = "E:/5thsem el/output_v2"  # Raw v2 landmarks
    output_dir_v2 = "E:/5thsem el/output_preprocessed_v2"  # Preprocessed v2
    
    # Feature dimensions
    raw_dim_v1 = 138  # 2 hands × 21 × 3 + 2 shoulders × 3 + 2 elbows × 3
    raw_dim_v2 = 204  # v1 + 20 mouth × 3 + 6 head pose
    
    # Legacy alias
    raw_dim = raw_dim_v1
    
    # Preprocessing parameters
    smoothing_sigma = 1.0  # Gaussian smoothing for jitter reduction
    min_hand_span = 0.01  # Minimum hand span to avoid division by zero
    
    # Augmentation (applied during extraction)
    add_velocity = True
    add_acceleration = True
    
    # Multi-processing
    num_workers = max(1, cpu_count() - 2)
    
    # Feature version to use ('v1' or 'v2')
    feature_version = 'v1'  # Change to 'v2' when using face landmarks


# ==================== PREPROCESSING FUNCTIONS ====================

def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalize landmarks to be person and camera invariant
    
    Supports both v1 (138 dims) and v2 (204 dims) formats.
    
    Args:
        landmarks: (T, D) raw MediaPipe output where D is 138 (v1) or 204 (v2)
        
    Returns:
        normalized: (T, D) normalized landmarks (same dimension as input)
    """
    T = landmarks.shape[0]
    D = landmarks.shape[1]
    
    # Determine version from dimension
    is_v2 = D > 138
    
    # ===== Extract components =====
    # Hands: 2 hands × 21 landmarks × 3 coords = 126
    hands = landmarks[:, :126].reshape(T, 2, 21, 3).copy()
    # Shoulders: 2 × 3 = 6
    shoulders = landmarks[:, 126:132].reshape(T, 2, 3).copy()
    # Elbows: 2 × 3 = 6
    elbows = landmarks[:, 132:138].reshape(T, 2, 3).copy()
    
    # V2 features (if present)
    if is_v2:
        # Mouth: 20 × 3 = 60
        mouth = landmarks[:, 138:198].reshape(T, 20, 3).copy() if D >= 198 else None
        # Head pose: 6 (already normalized, don't modify)
        head_pose = landmarks[:, 198:204].copy() if D >= 204 else None
    else:
        mouth = None
        head_pose = None
    
    # ===== 1. Normalize each hand to wrist origin =====
    for h in range(2):
        wrist = hands[:, h, 0:1, :].copy()  # (T, 1, 3)
        hands[:, h] = hands[:, h] - wrist  # Center to wrist
    
    # ===== 2. Scale by hand span =====
    for h in range(2):
        span = np.linalg.norm(hands[:, h, 12] - hands[:, h, 0], axis=1, keepdims=True)
        span = np.maximum(span, PreprocessConfig.min_hand_span)
        hands[:, h] = hands[:, h] / span[:, :, np.newaxis]
    
    # ===== 3. Normalize shoulders/elbows relative to body center =====
    body_center = shoulders.mean(axis=1, keepdims=True)
    shoulders = shoulders - body_center
    elbows = elbows - body_center
    
    # ===== 4. Scale body landmarks by shoulder width =====
    shoulder_width = np.linalg.norm(shoulders[:, 0] - shoulders[:, 1], axis=1, keepdims=True)
    shoulder_width = np.maximum(shoulder_width, PreprocessConfig.min_hand_span)
    shoulders = shoulders / shoulder_width[:, :, np.newaxis]
    elbows = elbows / shoulder_width[:, :, np.newaxis]
    
    # ===== 5. Normalize mouth landmarks (if v2) =====
    if mouth is not None:
        # Center mouth to nose-tip-ish center (average of all mouth points)
        mouth_center = mouth.mean(axis=1, keepdims=True)
        mouth = mouth - mouth_center
        # Scale by mouth width (roughly between corners)
        mouth_width = np.linalg.norm(mouth[:, 0] - mouth[:, 6], axis=1, keepdims=True)
        mouth_width = np.maximum(mouth_width, PreprocessConfig.min_hand_span)
        mouth = mouth / mouth_width[:, :, np.newaxis]
    
    # ===== 6. Flatten back to (T, D) =====
    parts = [
        hands.reshape(T, -1),       # 126
        shoulders.reshape(T, -1),   # 6
        elbows.reshape(T, -1)       # 6
    ]
    
    if is_v2:
        if mouth is not None:
            parts.append(mouth.reshape(T, -1))  # 60
        if head_pose is not None:
            parts.append(head_pose)  # 6 (unchanged)
    
    normalized = np.concatenate(parts, axis=1)
    
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


def get_feature_dim(add_velocity: bool = True, add_acceleration: bool = True,
                    version: str = 'v1') -> int:
    """
    Get output feature dimension based on settings
    
    Args:
        add_velocity: Include velocity features
        add_acceleration: Include acceleration features
        version: 'v1' (138 base) or 'v2' (204 base with face landmarks)
    
    Returns:
        Total feature dimension
    """
    if version == 'v2':
        dim = 204  # Base v2 landmarks (hands + body + mouth + head pose)
    else:
        dim = 138  # Base v1 landmarks (hands + body only)
    
    if add_velocity:
        dim += dim // (1 + int(add_acceleration))  # Add velocity dim
    if add_acceleration and add_velocity:
        dim += dim // (2 if add_velocity else 1)  # Add acceleration dim
    
    # Simpler calculation:
    multiplier = 1
    if add_velocity:
        multiplier += 1
    if add_acceleration and add_velocity:
        multiplier += 1
    
    base_dim = 204 if version == 'v2' else 138
    return base_dim * multiplier


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
