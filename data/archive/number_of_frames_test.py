import cv2

video_path = "G:\kortex 5th sem\iSign_videos\train\_-Db5LNVqTM--0.mp4"
cap = cv2.VideoCapture(video_path)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0 or fps is None:
    print("⚠️ FPS not available from metadata")
    duration = None
else:
    duration = total_frames / fps

cap.release()

print("Frames:", total_frames)
print("FPS:", fps)
print("Duration:", duration)
