"""
Enhanced Landmark Extraction v2 for Indian Sign Language
Adds mouth landmarks and head pose for ISL-specific grammar markers

Features:
- Hands: 2 × 21 × 3 = 126 dims (same as v1)
- Shoulders: 2 × 3 = 6 dims (same as v1)  
- Elbows: 2 × 3 = 6 dims (same as v1)
- Mouth: 20 × 3 = 60 dims (NEW - for question/emotion markers)
- Head Pose: 6 dims (NEW - pitch, yaw, roll + translation)

Total raw features: 138 + 60 + 6 = 204 dims
After preprocessing (with velocity + acceleration): 612 dims

ISL-Specific Importance:
- Mouth shape: Questions in ISL use specific lip patterns
- Head pose: Nodding/shaking conveys yes/no, head tilt for questions
- Raised eyebrows: Question markers (captured via face landmarks)

Optimizations:
- ProcessPoolExecutor for Windows compatibility (faster than Pool)
- Chunked processing with progress tracking
- Resume-safe (skips existing files)
- Lower resolution for speed (320x240)
"""

import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =====================
# CONFIGURATION
# =====================

class ExtractConfig:
    """Configuration for landmark extraction"""
    # Paths
    VIDEO_DIR = "E:/iSign-videos_v1.1"
    OUT_DIR = "E:/5thsem el/output_v2"  # New output for v2 features
    
    # Processing
    MAX_HANDS = 2
    FPS_SKIP = 4  # ~7-8 FPS effective (balance quality/speed)
    NUM_WORKERS = max(1, cpu_count() - 2)  # Leave 2 cores for system
    CHUNK_SIZE = 16  # Videos per worker batch
    
    # Resolution for processing
    PROCESS_WIDTH = 320
    PROCESS_HEIGHT = 240
    
    # Confidence thresholds (lower = faster but may miss some frames)
    HAND_CONFIDENCE = 0.4
    POSE_CONFIDENCE = 0.4
    FACE_CONFIDENCE = 0.4
    
    # Feature dimensions
    HAND_DIM = 126  # 2 hands × 21 landmarks × 3 coords
    BODY_DIM = 12   # 2 shoulders × 3 + 2 elbows × 3
    MOUTH_DIM = 60  # 20 mouth landmarks × 3 coords
    HEAD_POSE_DIM = 6  # pitch, yaw, roll, tx, ty, tz
    
    # Which features to extract
    EXTRACT_HANDS = True
    EXTRACT_BODY = True
    EXTRACT_MOUTH = True
    EXTRACT_HEAD_POSE = True


# MediaPipe mouth landmark indices (20 key points)
# These capture lip shape which is important for ISL question markers
MOUTH_LANDMARK_INDICES = [
    # Outer lip
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    # Inner lip  
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
][:20]  # Take first 20 for consistency


def extract_head_pose(face_landmarks, image_width, image_height):
    """
    Estimate head pose from face landmarks
    
    Uses nose tip, chin, and eye corners to estimate 3D rotation
    Returns: [pitch, yaw, roll, tx, ty, tz] normalized values
    """
    if face_landmarks is None:
        return np.zeros(6, dtype=np.float32)
    
    try:
        # Key landmarks for pose estimation
        # Nose tip (1), chin (152), left eye outer (33), right eye outer (263)
        # left mouth corner (61), right mouth corner (291)
        
        nose = face_landmarks.landmark[1]
        chin = face_landmarks.landmark[152]
        left_eye = face_landmarks.landmark[33]
        right_eye = face_landmarks.landmark[263]
        left_mouth = face_landmarks.landmark[61]
        right_mouth = face_landmarks.landmark[291]
        
        # Calculate relative positions
        # Yaw: horizontal rotation (left/right head turn)
        eye_dist = right_eye.x - left_eye.x
        nose_offset = (nose.x - (left_eye.x + right_eye.x) / 2) / max(eye_dist, 0.01)
        yaw = np.clip(nose_offset * 2, -1, 1)
        
        # Pitch: vertical rotation (looking up/down)
        nose_chin_dist = chin.y - nose.y
        pitch = np.clip((nose_chin_dist - 0.15) * 5, -1, 1)  # Normalized
        
        # Roll: head tilt
        eye_slope = (right_eye.y - left_eye.y) / max(eye_dist, 0.01)
        roll = np.clip(eye_slope * 3, -1, 1)
        
        # Translation (head position in frame)
        tx = (nose.x - 0.5) * 2  # Centered at 0
        ty = (nose.y - 0.5) * 2
        tz = nose.z * 2  # Depth
        
        return np.array([pitch, yaw, roll, tx, ty, tz], dtype=np.float32)
        
    except Exception:
        return np.zeros(6, dtype=np.float32)


def extract_mouth_landmarks(face_landmarks):
    """
    Extract 20 mouth landmarks for ISL question/emotion markers
    
    Returns: (20, 3) array of normalized coordinates
    """
    if face_landmarks is None:
        return np.zeros((20, 3), dtype=np.float32)
    
    try:
        mouth = np.zeros((20, 3), dtype=np.float32)
        for i, idx in enumerate(MOUTH_LANDMARK_INDICES):
            if i >= 20:
                break
            lm = face_landmarks.landmark[idx]
            mouth[i] = [lm.x, lm.y, lm.z]
        return mouth
    except Exception:
        return np.zeros((20, 3), dtype=np.float32)


def extract_from_video(video_path, config=None):
    """
    Extract all landmarks from a single video
    
    Returns:
        features: (T, D) array where D depends on config
            - Base (hands + body): 138 dims
            - With mouth: +60 dims = 198 dims  
            - With head pose: +6 dims = 204 dims
    """
    if config is None:
        config = ExtractConfig()
    
    # Initialize MediaPipe solutions per worker
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    mp_face = mp.solutions.face_mesh
    
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=config.MAX_HANDS,
        min_detection_confidence=config.HAND_CONFIDENCE,
        min_tracking_confidence=config.HAND_CONFIDENCE
    )
    
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,  # Fastest
        smooth_landmarks=False,
        enable_segmentation=False,
        min_detection_confidence=config.POSE_CONFIDENCE,
        min_tracking_confidence=config.POSE_CONFIDENCE
    )
    
    face_mesh = None
    if config.EXTRACT_MOUTH or config.EXTRACT_HEAD_POSE:
        face_mesh = mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,  # Faster without iris refinement
            min_detection_confidence=config.FACE_CONFIDENCE,
            min_tracking_confidence=config.FACE_CONFIDENCE
        )
    
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer
    
    frames = []
    frame_id = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_id += 1
        if frame_id % config.FPS_SKIP != 0:
            continue
        
        # Resize for faster processing
        frame = cv2.resize(frame, (config.PROCESS_WIDTH, config.PROCESS_HEIGHT), 
                          interpolation=cv2.INTER_NEAREST)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # ===== HANDS =====
        hand_landmarks = np.zeros((config.MAX_HANDS, 21, 3), dtype=np.float32)
        has_hand = False
        
        if config.EXTRACT_HANDS:
            hands_result = hands.process(rgb)
            if hands_result.multi_hand_landmarks:
                has_hand = True
                for h_id, hand_lm in enumerate(hands_result.multi_hand_landmarks):
                    if h_id >= config.MAX_HANDS:
                        break
                    for lm_id, lm in enumerate(hand_lm.landmark):
                        hand_landmarks[h_id, lm_id] = [lm.x, lm.y, lm.z]
        
        # ===== POSE (shoulders + elbows) =====
        shoulder_landmarks = np.zeros((2, 3), dtype=np.float32)
        elbow_landmarks = np.zeros((2, 3), dtype=np.float32)
        
        if config.EXTRACT_BODY and has_hand:
            pose_result = pose.process(rgb)
            if pose_result.pose_landmarks:
                lm = pose_result.pose_landmarks.landmark
                shoulder_landmarks[0] = [lm[11].x, lm[11].y, lm[11].z]  # Left
                shoulder_landmarks[1] = [lm[12].x, lm[12].y, lm[12].z]  # Right
                elbow_landmarks[0] = [lm[13].x, lm[13].y, lm[13].z]     # Left
                elbow_landmarks[1] = [lm[14].x, lm[14].y, lm[14].z]     # Right
        
        # ===== FACE (mouth + head pose) =====
        mouth_landmarks = np.zeros((20, 3), dtype=np.float32)
        head_pose = np.zeros(6, dtype=np.float32)
        
        if face_mesh is not None and has_hand:
            face_result = face_mesh.process(rgb)
            if face_result.multi_face_landmarks:
                face_lm = face_result.multi_face_landmarks[0]
                
                if config.EXTRACT_MOUTH:
                    mouth_landmarks = extract_mouth_landmarks(face_lm)
                
                if config.EXTRACT_HEAD_POSE:
                    head_pose = extract_head_pose(
                        face_lm, config.PROCESS_WIDTH, config.PROCESS_HEIGHT
                    )
        
        # ===== COMBINE =====
        combined_parts = []
        
        if config.EXTRACT_HANDS:
            combined_parts.append(hand_landmarks.reshape(-1))  # 126
        
        if config.EXTRACT_BODY:
            combined_parts.append(shoulder_landmarks.reshape(-1))  # 6
            combined_parts.append(elbow_landmarks.reshape(-1))     # 6
        
        if config.EXTRACT_MOUTH:
            combined_parts.append(mouth_landmarks.reshape(-1))  # 60
        
        if config.EXTRACT_HEAD_POSE:
            combined_parts.append(head_pose)  # 6
        
        combined = np.concatenate(combined_parts)
        frames.append(combined)
    
    cap.release()
    hands.close()
    pose.close()
    if face_mesh is not None:
        face_mesh.close()
    
    return np.asarray(frames, dtype=np.float32)


def process_single_video(args):
    """Worker function for parallel processing"""
    vid, video_dir, out_dir, config = args
    
    if not vid.endswith(".mp4"):
        return None
    
    out_path = os.path.join(out_dir, vid.replace(".mp4", ".npy"))
    if os.path.exists(out_path):
        return None  # Skip existing
    
    video_path = os.path.join(video_dir, vid)
    
    try:
        data = extract_from_video(video_path, config)
        
        if len(data) == 0:
            return f"⚠️ No usable frames: {vid}"
        
        np.save(out_path, data)
        return None
        
    except Exception as e:
        return f"❌ Error processing {vid}: {str(e)}"


def extract_all_videos(config=None):
    """
    Extract landmarks from all videos using parallel processing
    
    Uses ProcessPoolExecutor which works better on Windows than Pool
    """
    if config is None:
        config = ExtractConfig()
    
    os.makedirs(config.OUT_DIR, exist_ok=True)
    
    # Get video list
    video_files = sorted([v for v in os.listdir(config.VIDEO_DIR) if v.endswith(".mp4")])
    
    # Count already processed
    existing = set(os.listdir(config.OUT_DIR))
    to_process = [v for v in video_files if v.replace(".mp4", ".npy") not in existing]
    
    print(f"{'='*60}")
    print("LANDMARK EXTRACTION V2 (Enhanced for ISL)")
    print(f"{'='*60}")
    print(f"Video directory: {config.VIDEO_DIR}")
    print(f"Output directory: {config.OUT_DIR}")
    print(f"Total videos: {len(video_files)}")
    print(f"Already processed: {len(video_files) - len(to_process)}")
    print(f"To process: {len(to_process)}")
    print(f"Workers: {config.NUM_WORKERS}")
    print(f"Features: hands={config.EXTRACT_HANDS}, body={config.EXTRACT_BODY}, "
          f"mouth={config.EXTRACT_MOUTH}, head_pose={config.EXTRACT_HEAD_POSE}")
    
    # Calculate output dimension
    out_dim = 0
    if config.EXTRACT_HANDS:
        out_dim += 126
    if config.EXTRACT_BODY:
        out_dim += 12
    if config.EXTRACT_MOUTH:
        out_dim += 60
    if config.EXTRACT_HEAD_POSE:
        out_dim += 6
    print(f"Output dimension: {out_dim} (with vel+acc: {out_dim * 3})")
    print(f"{'='*60}\n")
    
    if len(to_process) == 0:
        print("✅ All videos already processed!")
        return
    
    # Prepare tasks
    tasks = [
        (vid, config.VIDEO_DIR, config.OUT_DIR, config) 
        for vid in to_process
    ]
    
    # Process with progress bar
    errors = []
    
    with ProcessPoolExecutor(max_workers=config.NUM_WORKERS) as executor:
        futures = {executor.submit(process_single_video, task): task[0] for task in tasks}
        
        with tqdm(total=len(tasks), desc="Extracting", ncols=100) as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    errors.append(result)
                pbar.update(1)
    
    # Report errors
    if errors:
        print(f"\n⚠️ {len(errors)} files had issues:")
        for err in errors[:10]:
            print(f"  {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
    
    # Verify output
    output_files = list(Path(config.OUT_DIR).glob("*.npy"))
    print(f"\n✅ Extraction complete!")
    print(f"   Output files: {len(output_files)}")
    
    if output_files:
        sample = np.load(output_files[0])
        print(f"   Sample shape: {sample.shape}")
        print(f"   Feature dimension: {sample.shape[1]}")


def extract_single_for_testing(video_path, config=None):
    """
    Extract landmarks from a single video for testing
    
    Returns the raw features without saving
    """
    if config is None:
        config = ExtractConfig()
    
    features = extract_from_video(video_path, config)
    
    print(f"Extracted: {features.shape}")
    print(f"  Frames: {features.shape[0]}")
    print(f"  Features: {features.shape[1]}")
    
    return features


# =====================
# LEGACY COMPATIBILITY
# =====================

def convert_v1_to_v2_format(v1_path, v2_path):
    """
    Convert v1 features (138 dims) to v2 format by padding with zeros
    
    Useful for testing without re-extraction
    """
    v1_data = np.load(v1_path)
    T = v1_data.shape[0]
    
    # v1: 138 (hands + body)
    # v2: 204 (hands + body + mouth + head pose)
    v2_data = np.zeros((T, 204), dtype=np.float32)
    v2_data[:, :138] = v1_data  # Copy existing features
    # Mouth (60) and head pose (6) remain zeros
    
    np.save(v2_path, v2_data)
    print(f"Converted {v1_path} → {v2_path}")
    print(f"  Shape: {v1_data.shape} → {v2_data.shape}")


# =====================
# MAIN
# =====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract landmarks v2 with face features")
    parser.add_argument("--video-dir", type=str, default=ExtractConfig.VIDEO_DIR,
                       help="Directory containing videos")
    parser.add_argument("--output-dir", type=str, default=ExtractConfig.OUT_DIR,
                       help="Output directory for .npy files")
    parser.add_argument("--workers", type=int, default=ExtractConfig.NUM_WORKERS,
                       help="Number of parallel workers")
    parser.add_argument("--no-mouth", action="store_true",
                       help="Skip mouth landmark extraction")
    parser.add_argument("--no-head-pose", action="store_true",
                       help="Skip head pose extraction")
    parser.add_argument("--test-video", type=str, default=None,
                       help="Test extraction on a single video")
    
    args = parser.parse_args()
    
    # Create config
    config = ExtractConfig()
    config.VIDEO_DIR = args.video_dir
    config.OUT_DIR = args.output_dir
    config.NUM_WORKERS = args.workers
    config.EXTRACT_MOUTH = not args.no_mouth
    config.EXTRACT_HEAD_POSE = not args.no_head_pose
    
    if args.test_video:
        # Test single video
        features = extract_single_for_testing(args.test_video, config)
    else:
        # Process all videos
        extract_all_videos(config)
