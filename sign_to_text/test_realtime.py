"""
Real-time Sign Language Recognition Test Script
Desktop demo with webcam and streaming predictions (GPT-style)
"""

import cv2
import numpy as np
import mediapipe as mp
import torch
from collections import deque
import time

from model_mobile import create_mobile_model, StreamingInference
from vocabulary_builder import VocabularyBuilder


class RealtimeSignRecognition:
    """Real-time sign language recognition with webcam"""
    
    def __init__(self, model_path, vocab_path, device='cpu'):
        print("🚀 Initializing Real-time Sign Language Recognition...")
        
        self.device = device
        
        # Load vocabulary
        print(f"Loading vocabulary from {vocab_path}...")
        self.vocab = VocabularyBuilder.load(vocab_path)
        
        # Load model
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)
        vocab_size = checkpoint['vocab_size']
        hidden_dim = checkpoint['hidden_dim']
        
        self.model = create_mobile_model(vocab_size=vocab_size, hidden_dim=hidden_dim)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.model.to(device)
        
        print(f"Model loaded: {vocab_size} words, {hidden_dim} hidden dim")
        
        # MediaPipe setup
        print("Initializing MediaPipe...")
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Streaming inference
        self.streaming = StreamingInference(
            model=self.model,
            vocab=self.vocab,
            window_size=60,  # 2 seconds at 30 FPS
            stride=10,       # Update every ~0.3 seconds
            confidence_threshold=0.6,
            device=device
        )
        
        # Display state
        self.current_text = ""
        self.text_history = deque(maxlen=5)  # Keep last 5 predictions
        self.fps_history = deque(maxlen=30)
        
        print("✅ Initialization complete!\n")
    
    def extract_landmarks(self, frame):
        """Extract hand and pose landmarks from frame"""
        # Resize for faster processing
        frame_small = cv2.resize(frame, (320, 240))
        rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        
        # Extract landmarks
        hand_landmarks = np.zeros((2, 21, 3), dtype=np.float32)
        shoulder_landmarks = np.zeros((2, 3), dtype=np.float32)
        elbow_landmarks = np.zeros((2, 3), dtype=np.float32)
        
        has_hand = False
        
        # Hands
        hands_result = self.hands.process(rgb)
        if hands_result.multi_hand_landmarks:
            has_hand = True
            for h_id, hand_lm in enumerate(hands_result.multi_hand_landmarks):
                if h_id >= 2:
                    break
                for lm_id, lm in enumerate(hand_lm.landmark):
                    hand_landmarks[h_id, lm_id] = [lm.x, lm.y, lm.z]
                
                # Draw on original frame
                self.mp_drawing.draw_landmarks(
                    frame, 
                    hand_lm, 
                    self.mp_hands.HAND_CONNECTIONS
                )
        
        # Pose (only if hands detected)
        if has_hand:
            pose_result = self.pose.process(rgb)
            if pose_result.pose_landmarks:
                lm = pose_result.pose_landmarks.landmark
                shoulder_landmarks[0] = [lm[11].x, lm[11].y, lm[11].z]  # Left
                shoulder_landmarks[1] = [lm[12].x, lm[12].y, lm[12].z]  # Right
                elbow_landmarks[0] = [lm[13].x, lm[13].y, lm[13].z]     # Left
                elbow_landmarks[1] = [lm[14].x, lm[14].y, lm[14].z]     # Right
        
        # Combine landmarks
        combined = np.concatenate([
            hand_landmarks.reshape(-1),      # 126
            shoulder_landmarks.reshape(-1),  # 6
            elbow_landmarks.reshape(-1)      # 6
        ])  # Total: 138
        
        return combined, has_hand
    
    def draw_ui(self, frame):
        """Draw UI elements on frame"""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay for text
        overlay = frame.copy()
        
        # Top bar (status)
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        
        # Bottom bar (predictions)
        cv2.rectangle(overlay, (0, h-150), (w, h), (0, 0, 0), -1)
        
        # Blend
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # FPS
        if self.fps_history:
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Title
        cv2.putText(frame, "Sign Language Recognition (Real-time)", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Current prediction (GPT-style streaming)
        if self.current_text:
            # Word-wrap text
            words = self.current_text.split()
            lines = []
            current_line = ""
            
            for word in words:
                test_line = current_line + " " + word if current_line else word
                # Check if line is too long (approximate)
                if len(test_line) > 50:
                    lines.append(current_line)
                    current_line = word
                else:
                    current_line = test_line
            
            if current_line:
                lines.append(current_line)
            
            # Draw lines
            y_start = h - 120
            for i, line in enumerate(lines[-3:]):  # Show last 3 lines
                cv2.putText(frame, line, (20, y_start + i*35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "Waiting for sign...", (20, h - 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
        
        # Instructions
        cv2.putText(frame, "Press 'Q' to quit | 'R' to reset | 'SPACE' to clear", 
                   (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def run(self, camera_id=0):
        """Run real-time recognition"""
        print(f"Opening camera {camera_id}...")
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"❌ Failed to open camera {camera_id}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("\n" + "="*60)
        print("STARTING REAL-TIME RECOGNITION")
        print("="*60)
        print("Controls:")
        print("  Q - Quit")
        print("  R - Reset buffer")
        print("  SPACE - Clear text")
        print("="*60 + "\n")
        
        frame_count = 0
        last_prediction_time = time.time()
        
        try:
            while True:
                start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to read frame")
                    break
                
                # Mirror frame for natural interaction
                frame = cv2.flip(frame, 1)
                
                # Extract landmarks
                landmarks, has_hand = self.extract_landmarks(frame)
                
                # Add to buffer if hand detected
                if has_hand:
                    self.streaming.add_frame(landmarks)
                    
                    # Predict every 10 frames (~0.3s at 30 FPS)
                    if frame_count % 10 == 0 and self.streaming.should_predict():
                        current_time = time.time()
                        
                        # Run inference
                        prediction, confidence = self.streaming.predict(return_confidence=True)
                        
                        inference_time = (time.time() - current_time) * 1000
                        
                        # Update text if prediction changed
                        if prediction and prediction != self.current_text:
                            # GPT-style: append new words
                            if self.current_text and prediction.startswith(self.current_text):
                                # Prediction is extension of current text
                                self.current_text = prediction
                            else:
                                # New prediction, replace
                                self.current_text = prediction
                            
                            self.text_history.append(prediction)
                            print(f"[{confidence:.2f}] {prediction} ({inference_time:.1f}ms)")
                
                # Draw UI
                self.draw_ui(frame)
                
                # Show frame
                cv2.imshow('Sign Language Recognition', frame)
                
                # Calculate FPS
                fps = 1.0 / (time.time() - start_time)
                self.fps_history.append(fps)
                
                frame_count += 1
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # Q or ESC
                    break
                elif key == ord('r'):  # Reset
                    self.streaming.reset()
                    print("Buffer reset")
                elif key == ord(' '):  # Clear text
                    self.current_text = ""
                    self.text_history.clear()
                    print("Text cleared")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.hands.close()
            self.pose.close()
            
            print("\n" + "="*60)
            print("SESSION SUMMARY")
            print("="*60)
            print(f"Total frames processed: {frame_count}")
            if self.fps_history:
                print(f"Average FPS: {sum(self.fps_history)/len(self.fps_history):.2f}")
            print(f"Predictions made: {len(self.text_history)}")
            if self.text_history:
                print("\nRecent predictions:")
                for i, text in enumerate(list(self.text_history)[-5:], 1):
                    print(f"  {i}. {text}")
            print("="*60)


def main():
    """Main test function"""
    # Paths
    model_path = "checkpoints/best_model.pth"
    vocab_path = "vocabulary.pkl"
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Create recognizer
    recognizer = RealtimeSignRecognition(model_path, vocab_path, device=device)
    
    # Run
    recognizer.run(camera_id=0)


if __name__ == "__main__":
    main()
