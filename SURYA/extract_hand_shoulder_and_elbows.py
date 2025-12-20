import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import os
from multiprocessing import Pool, cpu_count, Manager
from functools import partial

# =====================
# CONFIG
# =====================
VIDEO_DIR = "E:\iSign-videos_v1.1"
OUT_DIR   = "E:\5thsem el\output"

MAX_HANDS = 2
FPS_SKIP  = 3   # ~8–10 FPS effective
NUM_WORKERS = max(1, cpu_count() - 1)  # Leave 1 core free

os.makedirs(OUT_DIR, exist_ok=True)

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
    return np.asarray(frames, dtype=np.float32)  # (T, 138)

# =====================
# Batch Processing (Resume-Safe)
# =====================
for vid in tqdm(sorted(os.listdir(VIDEO_DIR))):
    if not vid.endswith(".mp4"):
        continue

    out_path = os.path.join(OUT_DIR, vid.replace(".mp4", ".npy"))
    if os.path.exists(out_path):
        continue

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
