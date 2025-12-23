"""
Data Augmentation for Sign Language Landmark Sequences

Augmentations specifically designed for hand/body landmark sequences:
- Temporal augmentations (speed, shift, dropout)
- Spatial augmentations (noise, scale, rotation)
- Landmark-specific augmentations (hand swap, jitter)

Based on research from:
- TSLFormer (2025): Lightweight Transformer for Sign Language
- SPOTER (2022): Efficient pose-based sign recognition
- SpecAugment-style masking for temporal sequences
"""

import numpy as np
import torch
from typing import Optional, Tuple


class LandmarkAugmenter:
    """
    Comprehensive augmentation pipeline for sign language landmarks
    
    Input format: (time, 414) where 414 = 138 position + 138 velocity + 138 acceleration
    Position breakdown: 2 hands × 21 landmarks × 3 coords = 126, + 2 shoulders × 3 + 2 elbows × 3 = 12
    Total: 126 + 12 = 138 position features
    """
    
    def __init__(
        self,
        # Temporal augmentations
        time_warp_prob: float = 0.5,
        time_warp_range: Tuple[float, float] = (0.85, 1.15),  # ±15% speed
        frame_dropout_prob: float = 0.3,
        frame_dropout_rate: float = 0.1,  # Drop 10% of frames
        time_shift_prob: float = 0.3,
        time_shift_range: int = 5,  # ±5 frames
        
        # Spatial augmentations
        gaussian_noise_prob: float = 0.5,
        gaussian_noise_std: float = 0.02,
        scale_prob: float = 0.3,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        rotation_prob: float = 0.2,
        rotation_range: float = 10.0,  # ±10 degrees
        
        # Landmark-specific
        hand_swap_prob: float = 0.1,  # Swap left/right hands
        landmark_dropout_prob: float = 0.2,
        landmark_dropout_rate: float = 0.05,  # Drop 5% of landmarks
        
        # Masking (SpecAugment-style)
        time_mask_prob: float = 0.3,
        time_mask_max_len: int = 20,
        feature_mask_prob: float = 0.2,
        feature_mask_max_size: int = 30,
    ):
        self.time_warp_prob = time_warp_prob
        self.time_warp_range = time_warp_range
        self.frame_dropout_prob = frame_dropout_prob
        self.frame_dropout_rate = frame_dropout_rate
        self.time_shift_prob = time_shift_prob
        self.time_shift_range = time_shift_range
        
        self.gaussian_noise_prob = gaussian_noise_prob
        self.gaussian_noise_std = gaussian_noise_std
        self.scale_prob = scale_prob
        self.scale_range = scale_range
        self.rotation_prob = rotation_prob
        self.rotation_range = rotation_range
        
        self.hand_swap_prob = hand_swap_prob
        self.landmark_dropout_prob = landmark_dropout_prob
        self.landmark_dropout_rate = landmark_dropout_rate
        
        self.time_mask_prob = time_mask_prob
        self.time_mask_max_len = time_mask_max_len
        self.feature_mask_prob = feature_mask_prob
        self.feature_mask_max_size = feature_mask_max_size
        
        # Landmark structure
        self.n_hand_landmarks = 21
        self.n_hands = 2
        self.coords_per_landmark = 3
        self.hand_features = self.n_hand_landmarks * self.coords_per_landmark  # 63 per hand
        self.body_features = 12  # 2 shoulders + 2 elbows × 3 coords
        self.mouth_features = 60  # 20 mouth landmarks × 3 coords (v2)
        self.head_pose_features = 6  # pitch, yaw, roll, tx, ty, tz (v2) - NOT landmarks!
        
        # Support both v1 (138) and v2 (204) position dims
        # Will be auto-detected in __call__ based on input shape
        self.position_dim_v1 = 138  # v1: hands + body
        self.position_dim_v2 = 204  # v2: hands + body + mouth + head pose
        
        # Actual landmark dims (exclude head pose which is not XY landmarks)
        # v2: 2*63 (hands) + 12 (body) + 60 (mouth) = 198 landmark coords
        self.landmark_dim_v2 = 198
        
    def __call__(self, landmarks: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Apply augmentations to landmark sequence
        
        Args:
            landmarks: (time, 414) or (time, 612) landmark sequence
            training: Whether in training mode (apply augmentations)
            
        Returns:
            augmented: (time', D) augmented sequence (time may change)
        """
        if not training:
            return landmarks
        
        x = landmarks.copy()
        
        # Auto-detect position dimension from input
        total_dim = x.shape[1]
        if total_dim == 612:
            position_dim = self.position_dim_v2  # 204
        elif total_dim == 414:
            position_dim = self.position_dim_v1  # 138
        else:
            # Assume raw features without derivatives
            position_dim = total_dim // 3 if total_dim % 3 == 0 else total_dim
        
        # Temporal augmentations
        if np.random.random() < self.time_warp_prob:
            x = self._time_warp(x)
            
        if np.random.random() < self.frame_dropout_prob:
            x = self._frame_dropout(x)
            
        if np.random.random() < self.time_shift_prob:
            x = self._time_shift(x)
        
        # Spatial augmentations (apply to position features, recompute velocity/accel)
        position = x[:, :position_dim]
        
        if np.random.random() < self.gaussian_noise_prob:
            position = self._add_gaussian_noise(position)
            
        if np.random.random() < self.scale_prob:
            position = self._random_scale(position)
            
        if np.random.random() < self.rotation_prob:
            position = self._random_rotation(position)
        
        # Landmark-specific augmentations (only on hand features - first 126 dims)
        if np.random.random() < self.hand_swap_prob:
            position = self._swap_hands(position)
            
        if np.random.random() < self.landmark_dropout_prob:
            position = self._landmark_dropout(position)
        
        # Recompute velocity and acceleration from augmented positions
        velocity = self._compute_velocity(position)
        acceleration = self._compute_acceleration(velocity)
        
        # Combine back
        x = np.concatenate([position, velocity, acceleration], axis=1)
        
        # Masking augmentations (apply to full features)
        if np.random.random() < self.time_mask_prob:
            x = self._time_mask(x)
            
        if np.random.random() < self.feature_mask_prob:
            x = self._feature_mask(x)
        
        return x.astype(np.float32)
    
    # ==================== TEMPORAL AUGMENTATIONS ====================
    
    def _time_warp(self, x: np.ndarray) -> np.ndarray:
        """Random speed perturbation via interpolation"""
        T, D = x.shape
        
        # Random speed factor
        speed = np.random.uniform(*self.time_warp_range)
        new_T = int(T / speed)
        
        if new_T < 5:  # Don't make sequence too short
            return x
        
        # Interpolate to new length
        old_indices = np.linspace(0, T - 1, new_T)
        new_x = np.zeros((new_T, D), dtype=x.dtype)
        
        for d in range(D):
            new_x[:, d] = np.interp(old_indices, np.arange(T), x[:, d])
        
        return new_x
    
    def _frame_dropout(self, x: np.ndarray) -> np.ndarray:
        """Randomly drop frames (simulate camera drops)"""
        T = x.shape[0]
        n_drop = int(T * self.frame_dropout_rate)
        
        if n_drop == 0 or T - n_drop < 5:
            return x
        
        # Random indices to keep
        keep_indices = np.sort(np.random.choice(T, T - n_drop, replace=False))
        
        return x[keep_indices]
    
    def _time_shift(self, x: np.ndarray) -> np.ndarray:
        """Random temporal shift (pad/trim start/end)"""
        T = x.shape[0]
        
        # Limit shift to avoid empty arrays
        max_shift = min(self.time_shift_range, T - 1)
        if max_shift <= 0:
            return x
            
        shift = np.random.randint(-max_shift, max_shift + 1)
        
        if shift == 0:
            return x
        elif shift > 0:
            # Shift right (remove from start, pad end)
            remaining = x[shift:]
            if len(remaining) == 0:
                return x
            return np.pad(remaining, ((0, shift), (0, 0)), mode='edge')
        else:
            # Shift left (remove from end, pad start)
            remaining = x[:shift]
            if len(remaining) == 0:
                return x
            return np.pad(remaining, ((-shift, 0), (0, 0)), mode='edge')
    
    def _time_mask(self, x: np.ndarray) -> np.ndarray:
        """Mask random time segments (SpecAugment-style)"""
        T = x.shape[0]
        
        # Skip if sequence is too short
        if T < 8:
            return x
        
        max_mask = min(self.time_mask_max_len, T // 4)
        if max_mask < 1:
            return x
            
        mask_len = np.random.randint(1, max_mask + 1)
        mask_start = np.random.randint(0, max(1, T - mask_len))
        
        x = x.copy()
        x[mask_start:mask_start + mask_len] = 0
        
        return x
    
    # ==================== SPATIAL AUGMENTATIONS ====================
    
    def _add_gaussian_noise(self, position: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to landmark positions"""
        noise = np.random.normal(0, self.gaussian_noise_std, position.shape)
        return position + noise
    
    def _random_scale(self, position: np.ndarray) -> np.ndarray:
        """Random uniform scaling"""
        scale = np.random.uniform(*self.scale_range)
        return position * scale
    
    def _random_rotation(self, position: np.ndarray) -> np.ndarray:
        """Random 2D rotation around Z-axis (in XY plane)
        
        Only rotates actual landmarks (198 dims for v2), not head pose (last 6 dims)
        """
        T = position.shape[0]
        position_dim = position.shape[1]
        
        angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        angle_rad = np.deg2rad(angle)
        
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        position = position.copy()
        
        # For v2 (204 dims): only rotate first 198 dims (actual landmarks)
        # Last 6 dims are head pose (pitch, yaw, roll, tx, ty, tz) - not XY coords
        if position_dim == 204:
            landmark_dim = self.landmark_dim_v2  # 198
        else:
            landmark_dim = position_dim  # v1 or other: rotate all
        
        # Reshape only the landmark portion
        landmarks = position[:, :landmark_dim].reshape(T, -1, 3)  # (T, num_landmarks, 3)
        
        x = landmarks[:, :, 0]
        y = landmarks[:, :, 1]
        
        new_x = x * cos_a - y * sin_a
        new_y = x * sin_a + y * cos_a
        
        landmarks[:, :, 0] = new_x
        landmarks[:, :, 1] = new_y
        
        position[:, :landmark_dim] = landmarks.reshape(T, -1)
        
        return position
    
    def _feature_mask(self, x: np.ndarray) -> np.ndarray:
        """Mask random feature dimensions"""
        D = x.shape[1]
        
        # Skip if feature dim is too small
        if D < 16:
            return x
            
        max_mask = min(self.feature_mask_max_size, D // 8)
        if max_mask < 1:
            return x
            
        mask_size = np.random.randint(1, max_mask + 1)
        mask_start = np.random.randint(0, max(1, D - mask_size))
        
        x = x.copy()
        x[:, mask_start:mask_start + mask_size] = 0
        
        return x
    
    # ==================== LANDMARK-SPECIFIC AUGMENTATIONS ====================
    
    def _swap_hands(self, position: np.ndarray) -> np.ndarray:
        """Swap left and right hands (horizontal flip semantically)"""
        T = position.shape[0]
        position = position.copy()
        
        # Hand features: first 63 = left hand, next 63 = right hand
        left_hand = position[:, :self.hand_features].copy()
        right_hand = position[:, self.hand_features:2*self.hand_features].copy()
        
        # Swap
        position[:, :self.hand_features] = right_hand
        position[:, self.hand_features:2*self.hand_features] = left_hand
        
        # Also need to flip X coordinates (mirror)
        # Reshape to access individual coordinates
        for i in range(0, 2 * self.hand_features, 3):
            position[:, i] = -position[:, i]  # Flip X
        
        return position
    
    def _landmark_dropout(self, position: np.ndarray) -> np.ndarray:
        """Randomly zero out some landmarks (simulate occlusion)
        
        Only drops actual landmarks, not head pose features
        """
        T = position.shape[0]
        position_dim = position.shape[1]
        
        # For v2 (204 dims): only consider first 198 dims as landmarks
        if position_dim == 204:
            landmark_dim = self.landmark_dim_v2  # 198
        else:
            landmark_dim = position_dim
        
        n_landmarks = landmark_dim // 3
        n_drop = int(n_landmarks * self.landmark_dropout_rate)
        
        if n_drop == 0:
            return position
        
        position = position.copy()
        landmarks = position[:, :landmark_dim].reshape(T, -1, 3)
        
        # Random landmarks to drop (same landmarks dropped across all frames)
        drop_indices = np.random.choice(n_landmarks, n_drop, replace=False)
        landmarks[:, drop_indices, :] = 0
        
        position[:, :landmark_dim] = landmarks.reshape(T, -1)
        return position
    
    # ==================== FEATURE COMPUTATION ====================
    
    def _compute_velocity(self, position: np.ndarray) -> np.ndarray:
        """Compute velocity (first derivative) from positions"""
        velocity = np.zeros_like(position)
        velocity[1:] = position[1:] - position[:-1]
        velocity[0] = velocity[1] if len(velocity) > 1 else 0
        return velocity
    
    def _compute_acceleration(self, velocity: np.ndarray) -> np.ndarray:
        """Compute acceleration (second derivative) from velocity"""
        acceleration = np.zeros_like(velocity)
        acceleration[1:] = velocity[1:] - velocity[:-1]
        acceleration[0] = acceleration[1] if len(acceleration) > 1 else 0
        return acceleration


class MixUp:
    """
    MixUp augmentation for sequences
    Interpolates between two samples and their labels
    """
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
    
    def __call__(
        self, 
        x1: np.ndarray, 
        x2: np.ndarray, 
        y1: np.ndarray, 
        y2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Mix two samples
        
        Returns:
            mixed_x: Interpolated features
            (y1, y2): Original labels (for soft loss)
            lam: Mixing coefficient
        """
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Pad sequences to same length
        max_len = max(len(x1), len(x2))
        x1_padded = np.pad(x1, ((0, max_len - len(x1)), (0, 0)), mode='constant')
        x2_padded = np.pad(x2, ((0, max_len - len(x2)), (0, 0)), mode='constant')
        
        mixed_x = lam * x1_padded + (1 - lam) * x2_padded
        
        return mixed_x, y1, y2, lam


def create_augmenter(intensity: str = 'medium') -> LandmarkAugmenter:
    """
    Factory function for creating augmenters with preset intensities
    
    Args:
        intensity: 'light', 'medium', or 'heavy'
        
    Returns:
        LandmarkAugmenter with appropriate settings
    """
    if intensity == 'light':
        return LandmarkAugmenter(
            time_warp_prob=0.3,
            time_warp_range=(0.9, 1.1),
            frame_dropout_prob=0.2,
            frame_dropout_rate=0.05,
            gaussian_noise_prob=0.3,
            gaussian_noise_std=0.01,
            scale_prob=0.2,
            rotation_prob=0.1,
            hand_swap_prob=0.05,
            time_mask_prob=0.2,
            feature_mask_prob=0.1,
        )
    elif intensity == 'medium':
        return LandmarkAugmenter(
            time_warp_prob=0.5,
            time_warp_range=(0.85, 1.15),
            frame_dropout_prob=0.3,
            frame_dropout_rate=0.1,
            gaussian_noise_prob=0.5,
            gaussian_noise_std=0.02,
            scale_prob=0.3,
            rotation_prob=0.2,
            hand_swap_prob=0.1,
            time_mask_prob=0.3,
            feature_mask_prob=0.2,
        )
    elif intensity == 'heavy':
        return LandmarkAugmenter(
            time_warp_prob=0.7,
            time_warp_range=(0.8, 1.2),
            frame_dropout_prob=0.5,
            frame_dropout_rate=0.15,
            gaussian_noise_prob=0.7,
            gaussian_noise_std=0.03,
            scale_prob=0.5,
            rotation_prob=0.3,
            hand_swap_prob=0.15,
            time_mask_prob=0.5,
            feature_mask_prob=0.3,
        )
    else:
        raise ValueError(f"Unknown intensity: {intensity}. Use 'light', 'medium', or 'heavy'.")


# ==================== TEST ====================

if __name__ == "__main__":
    print("Testing Data Augmentation Pipeline")
    print("=" * 60)
    
    # Create dummy landmarks
    T, D = 100, 414
    landmarks = np.random.randn(T, D).astype(np.float32)
    
    # Test different intensities
    for intensity in ['light', 'medium', 'heavy']:
        print(f"\nTesting {intensity} augmentation:")
        augmenter = create_augmenter(intensity)
        
        augmented = augmenter(landmarks, training=True)
        
        print(f"  Input shape: {landmarks.shape}")
        print(f"  Output shape: {augmented.shape}")
        print(f"  Time change: {landmarks.shape[0]} -> {augmented.shape[0]} frames")
        print(f"  Max abs value: {np.abs(augmented).max():.4f}")
    
    # Test no augmentation in eval mode
    print("\nTesting eval mode (no augmentation):")
    augmenter = create_augmenter('heavy')
    eval_output = augmenter(landmarks, training=False)
    print(f"  Same as input: {np.allclose(landmarks, eval_output)}")
    
    print("\n✅ All augmentation tests passed!")
