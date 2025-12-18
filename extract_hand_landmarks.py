import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import os

# =====================
# CONFIG
# =====================
VIDEO_DIR = "temp_dataset"
OUT_DIR   = "temp_poses"

MAX_HANDS = 2
FPS_SKIP  = 3   # <- safer for 128k videos (~8–10 FPS)

os.makedirs(OUT_DIR, exist_ok=True)

# =====================
# MediaPipe Setup
# =====================
mp_hands = mp.solutions.hands
mp_pose  = mp.solutions.pose

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_HANDS,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,      # 🔥 lightweight
    smooth_landmarks=False,  # 🔥 faster
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =====================
# Extraction Function
# =====================
def extract_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        if frame_id % FPS_SKIP != 0:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # -------- Hands --------
        hands_result = hands.process(rgb)

        # Initialize outputs
        hand_landmarks = np.zeros((MAX_HANDS, 21, 3), dtype=np.float32)
        shoulder_landmarks = np.zeros((2, 3), dtype=np.float32)

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
                # Left shoulder = 11, Right shoulder = 12
                shoulder_landmarks[0] = [lm[11].x, lm[11].y, lm[11].z]
                shoulder_landmarks[1] = [lm[12].x, lm[12].y, lm[12].z]

        # -------- Combine --------
        combined = np.concatenate([
            hand_landmarks.reshape(-1),      # 126
            shoulder_landmarks.reshape(-1)   # 6
        ])                                   # = 132

        frames.append(combined)

    cap.release()
    return np.asarray(frames, dtype=np.float32)  # (T, 132)

# =====================
# Batch Processing (Resume-Safe)
# =====================
for vid in tqdm(sorted(os.listdir(VIDEO_DIR))):
    if not vid.endswith(".mp4"):
        continue

    out_path = os.path.join(OUT_DIR, vid.replace(".mp4", ".npy"))
    if os.path.exists(out_path):
        continue  # 🔥 resume safety

    video_path = os.path.join(VIDEO_DIR, vid)
    data = extract_from_video(video_path)

    if len(data) == 0:
        print(f"⚠️ No usable frames: {vid}")
        continue

    np.save(out_path, data)

# =====================
# Cleanup
# =====================
hands.close()
pose.close()

print("✅ Landmark extraction completed.")
