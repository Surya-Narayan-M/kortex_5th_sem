import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import os
from multiprocessing import Pool, cpu_count

# =====================
# CONFIG
# =====================
VIDEO_DIR = "E:\iSign-videos_v1.1"
OUT_DIR   = "E://5thsem el//output"

MAX_HANDS = 2
FPS_SKIP  = 5   # Skip more frames for speed (was 3, now ~6 FPS)
NUM_WORKERS = max(1, cpu_count() - 2)  # Use all cores except 2
BATCH_SIZE = NUM_WORKERS * 4  # Prefetch more videos

os.makedirs(OUT_DIR, exist_ok=True)

# =====================
# Extraction Function
# =====================
def extract_from_video(video_path):
    # Initialize MediaPipe per worker process
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_HANDS,
        min_detection_confidence=0.4,  # Lower for speed (was 0.5)
        min_tracking_confidence=0.4    # Lower for speed (was 0.5)
    )
    
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        smooth_landmarks=False,
        enable_segmentation=False,     # Disable segmentation
        min_detection_confidence=0.4,  # Lower for speed (was 0.5)
        min_tracking_confidence=0.4    # Lower for speed (was 0.5)
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
        if frame_id % FPS_SKIP != 0:
            continue

        # Resize for faster processing (320x240 is enough for landmarks)
        frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_NEAREST)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # -------- Hands --------
        hands_result = hands.process(rgb)

        hand_landmarks = np.zeros((MAX_HANDS, 21, 3), dtype=np.float32)
        shoulder_landmarks = np.zeros((2, 3), dtype=np.float32)
        elbow_landmarks = np.zeros((2, 3), dtype=np.float32)

        has_hand = False

        if hands_result.multi_hand_landmarks:
            has_hand = True
            for h_id, hand_lm in enumerate(hands_result.multi_hand_landmarks):
                if h_id >= MAX_HANDS:
                    break
                for lm_id, lm in enumerate(hand_lm.landmark):
                    hand_landmarks[h_id, lm_id] = [lm.x, lm.y, lm.z]

        # -------- Pose (ONLY if hands exist) --------
        if has_hand:
            pose_result = pose.process(rgb)
            if pose_result.pose_landmarks:
                lm = pose_result.pose_landmarks.landmark

                # Shoulders
                shoulder_landmarks[0] = [lm[11].x, lm[11].y, lm[11].z]  # Left
                shoulder_landmarks[1] = [lm[12].x, lm[12].y, lm[12].z]  # Right

                # Elbows
                elbow_landmarks[0] = [lm[13].x, lm[13].y, lm[13].z]     # Left
                elbow_landmarks[1] = [lm[14].x, lm[14].y, lm[14].z]     # Right

        # -------- Combine --------
        combined = np.concatenate([
            hand_landmarks.reshape(-1),      # 126
            shoulder_landmarks.reshape(-1),  # 6
            elbow_landmarks.reshape(-1)      # 6
        ])                                   # = 138

        frames.append(combined)

    cap.release()
    hands.close()
    pose.close()
    return np.asarray(frames, dtype=np.float32)  # (T, 138)

# =====================
# Worker Function
# =====================
def process_single_video(vid):
    """Process a single video and save output"""
    if not vid.endswith(".mp4"):
        return None
    
    out_path = os.path.join(OUT_DIR, vid.replace(".mp4", ".npy"))
    if os.path.exists(out_path):
        return None
    
    video_path = os.path.join(VIDEO_DIR, vid)
    data = extract_from_video(video_path)
    
    if len(data) == 0:
        return f"⚠️ No usable frames: {vid}"
    
    np.save(out_path, data)
    return None

# =====================
# Batch Processing (Resume-Safe, Multi-Core)
# =====================
if __name__ == '__main__':
    video_files = sorted([v for v in os.listdir(VIDEO_DIR) if v.endswith(".mp4")])
    
    print(f"🚀 Processing {len(video_files)} videos using {NUM_WORKERS} workers...")
    print(f"⚡ Optimizations: FPS_SKIP={FPS_SKIP}, frame_size=320x240, lower confidence thresholds")
    
    with Pool(processes=NUM_WORKERS) as pool:
        results = list(tqdm(
            pool.imap(process_single_video, video_files, chunksize=8),
            total=len(video_files)
        ))
    
    # Print any warnings
    for result in results:
        if result:
            print(result)
    
    print("✅ Landmark extraction completed.")