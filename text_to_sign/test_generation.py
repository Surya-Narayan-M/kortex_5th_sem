"""
Test Text-to-Sign Generation
Generate sign language sequences from text inputs
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
import sys
import json

# Add parent directory for vocabulary import
sys.path.append(str(Path(__file__).parent.parent / 'sign_to_text'))
from vocabulary_builder import VocabularyBuilder
from model_text_to_sign import create_text_to_sign_model


class SignVisualizer:
    """Visualize generated sign language as 2D skeleton"""
    
    def __init__(self):
        # MediaPipe hand connections (21 landmarks per hand)
        self.hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
        ]
        
        # Body connections
        self.body_connections = [
            (63, 64),  # Left shoulder to right shoulder
            (63, 65),  # Left shoulder to left elbow
            (64, 66)   # Right shoulder to right elbow
        ]
    
    def extract_landmarks(self, frame_data):
        """
        Extract landmarks from 138-feature vector
        
        Layout:
        - 0-62: Left hand (21 points × 3 coords)
        - 63-125: Right hand (21 points × 3 coords)
        - 126-128: Left shoulder (x, y, z)
        - 129-131: Right shoulder (x, y, z)
        - 132-134: Left elbow (x, y, z)
        - 135-137: Right elbow (x, y, z)
        """
        left_hand = frame_data[:63].reshape(21, 3)
        right_hand = frame_data[63:126].reshape(21, 3)
        left_shoulder = frame_data[126:129]
        right_shoulder = frame_data[129:132]
        left_elbow = frame_data[132:135]
        right_elbow = frame_data[135:138]
        
        return {
            'left_hand': left_hand,
            'right_hand': right_hand,
            'left_shoulder': left_shoulder,
            'right_shoulder': right_shoulder,
            'left_elbow': left_elbow,
            'right_elbow': right_elbow
        }
    
    def plot_frame(self, ax, frame_data):
        """Plot single frame as 2D skeleton"""
        ax.clear()
        
        landmarks = self.extract_landmarks(frame_data)
        
        # Plot left hand
        left_hand = landmarks['left_hand'][:, :2]  # Use x, y only
        for conn in self.hand_connections:
            ax.plot([left_hand[conn[0], 0], left_hand[conn[1], 0]],
                   [left_hand[conn[0], 1], left_hand[conn[1], 1]],
                   'b-', linewidth=2)
        ax.scatter(left_hand[:, 0], left_hand[:, 1], c='blue', s=50, zorder=5)
        
        # Plot right hand
        right_hand = landmarks['right_hand'][:, :2]
        for conn in self.hand_connections:
            ax.plot([right_hand[conn[0], 0], right_hand[conn[1], 0]],
                   [right_hand[conn[0], 1], right_hand[conn[1], 1]],
                   'r-', linewidth=2)
        ax.scatter(right_hand[:, 0], right_hand[:, 1], c='red', s=50, zorder=5)
        
        # Plot body parts
        all_points = np.vstack([
            landmarks['left_shoulder'].reshape(1, 3)[:, :2],
            landmarks['right_shoulder'].reshape(1, 3)[:, :2],
            landmarks['left_elbow'].reshape(1, 3)[:, :2],
            landmarks['right_elbow'].reshape(1, 3)[:, :2]
        ])
        
        # Shoulder line
        ax.plot([all_points[0, 0], all_points[1, 0]],
               [all_points[0, 1], all_points[1, 1]],
               'g-', linewidth=3)
        
        # Left arm
        ax.plot([all_points[0, 0], all_points[2, 0]],
               [all_points[0, 1], all_points[2, 1]],
               'b--', linewidth=2)
        
        # Right arm
        ax.plot([all_points[1, 0], all_points[3, 0]],
               [all_points[1, 1], all_points[3, 1]],
               'r--', linewidth=2)
        
        ax.scatter(all_points[:, 0], all_points[:, 1], c='green', s=100, zorder=5)
        
        # Set axis properties
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title('Generated Sign Language Sequence')
        ax.grid(True, alpha=0.3)
    
    def animate_sequence(self, landmarks, save_path=None):
        """
        Animate landmark sequence
        
        Args:
            landmarks: (frames, 138) numpy array
            save_path: Optional path to save animation
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        def update(frame):
            self.plot_frame(ax, landmarks[frame])
            ax.set_xlabel(f'Frame {frame+1}/{len(landmarks)}')
        
        anim = FuncAnimation(fig, update, frames=len(landmarks), interval=50, repeat=True)
        
        if save_path:
            print(f"Saving animation to {save_path}...")
            anim.save(save_path, writer='pillow', fps=20)
            print("✅ Animation saved")
        else:
            plt.show()
        
        return anim


def load_model_and_vocab(checkpoint_path, vocab_path):
    """Load trained model and vocabulary"""
    
    # Load vocabulary
    print(f"Loading vocabulary from {vocab_path}...")
    vocab_builder = VocabularyBuilder.load(vocab_path)
    
    # Load checkpoint
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    
    # Create model
    model = create_text_to_sign_model(
        vocab_size=config['vocab_size'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded (val_loss: {checkpoint['val_loss']:.4f})")
    
    return model, vocab_builder


def generate_from_text(model, vocab_builder, text, max_frames=100):
    """
    Generate sign language landmarks from text
    
    Args:
        model: Trained text-to-sign model
        vocab_builder: VocabularyBuilder instance
        text: Input text string
        max_frames: Maximum frames to generate
    
    Returns:
        landmarks: (frames, 138) numpy array
    """
    # Convert text to tokens
    tokens = vocab_builder.text_to_word_indices(text)
    text_tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # (1, len)
    
    # Generate
    print(f"\nGenerating signs for: \"{text}\"")
    print(f"Tokens: {tokens}")
    
    with torch.no_grad():
        landmarks = model.generate(text_tokens, max_frames=max_frames)
    
    # Convert to numpy
    landmarks = landmarks.squeeze(0).cpu().numpy()  # (frames, 138)
    
    print(f"Generated {landmarks.shape[0]} frames")
    
    return landmarks


def main():
    print("Text-to-Sign Generation Test")
    print("="*60)
    
    # Paths
    checkpoint_path = Path("experiment/models/best_text_to_sign.pth")
    vocab_path = Path("vocabulary.pkl")
    output_dir = Path("experiment/outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Check files
    if not checkpoint_path.exists():
        print(f"❌ Model not found: {checkpoint_path}")
        print("Please train the model first: python experiment/train_text_to_sign.py")
        return
    
    if not vocab_path.exists():
        print(f"❌ Vocabulary not found: {vocab_path}")
        print("Please build vocabulary first: python vocabulary_builder.py")
        return
    
    # Load model
    model, vocab_builder = load_model_and_vocab(checkpoint_path, vocab_path)
    
    # Test sentences
    test_sentences = [
        "hello how are you",
        "i am fine thank you",
        "what is your name",
        "good morning everyone",
        "please help me"
    ]
    
    # Create visualizer
    visualizer = SignVisualizer()
    
    # Generate and visualize
    for i, sentence in enumerate(test_sentences):
        print(f"\n{'='*60}")
        print(f"Test {i+1}/{len(test_sentences)}")
        
        # Generate landmarks
        landmarks = generate_from_text(model, vocab_builder, sentence, max_frames=100)
        
        # Save landmarks
        npy_path = output_dir / f"test_{i+1}.npy"
        np.save(npy_path, landmarks)
        print(f"✅ Saved landmarks to {npy_path}")
        
        # Save first frame as image
        fig, ax = plt.subplots(figsize=(8, 8))
        visualizer.plot_frame(ax, landmarks[0])
        ax.set_title(f'"{sentence}" - Frame 1')
        img_path = output_dir / f"test_{i+1}_frame1.png"
        plt.savefig(img_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved visualization to {img_path}")
        
        # Save animation (optional - can be slow)
        # anim_path = output_dir / f"test_{i+1}.gif"
        # visualizer.animate_sequence(landmarks, save_path=anim_path)
    
    print("\n" + "="*60)
    print("✅ Generation test complete!")
    print(f"Outputs saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Review generated landmarks in outputs/ folder")
    print("2. Export model to TFLite: python experiment/export_text_to_sign.py")
    print("3. Integrate into Flutter app with 2D skeleton renderer")
    
    # Interactive mode
    print("\n" + "="*60)
    print("Interactive mode - Enter text to generate signs (or 'quit' to exit)")
    
    while True:
        text = input("\nEnter text: ").strip()
        if text.lower() in ['quit', 'exit', 'q']:
            break
        
        if not text:
            continue
        
        try:
            landmarks = generate_from_text(model, vocab_builder, text, max_frames=100)
            
            # Visualize first frame
            fig, ax = plt.subplots(figsize=(8, 8))
            visualizer.plot_frame(ax, landmarks[0])
            ax.set_title(f'"{text}" - Frame 1/{len(landmarks)}')
            plt.show()
            
            # Ask to save
            save = input("Save this sequence? (y/n): ").strip().lower()
            if save == 'y':
                filename = input("Filename (without extension): ").strip()
                if filename:
                    npy_path = output_dir / f"{filename}.npy"
                    np.save(npy_path, landmarks)
                    print(f"✅ Saved to {npy_path}")
        
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
