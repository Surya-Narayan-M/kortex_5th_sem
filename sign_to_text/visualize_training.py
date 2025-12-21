"""
Training Visualization Script
Plots training history and creates nice visualizations

Usage:
    python visualize_training.py
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_history(history_path):
    """Load training history from JSON"""
    with open(history_path, 'r') as f:
        return json.load(f)


def plot_training_curves(history, save_path=None):
    """Create comprehensive training visualization"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Hybrid CTC-Attention Training Progress', fontsize=14, fontweight='bold')
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1. Total Loss
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Total Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. CTC Loss
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['train_ctc_loss'], 'b-', label='Train CTC', linewidth=2)
    ax2.plot(epochs, history['val_ctc_loss'], 'r-', label='Val CTC', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('CTC Loss')
    ax2.set_title('CTC Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Attention Loss
    ax3 = axes[0, 2]
    ax3.plot(epochs, history['train_attn_loss'], 'b-', label='Train Attn', linewidth=2)
    ax3.plot(epochs, history['val_attn_loss'], 'r-', label='Val Attn', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Attention Loss')
    ax3.set_title('Attention Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Accuracy
    ax4 = axes[1, 0]
    train_acc = [a * 100 for a in history['train_acc']]
    val_acc = [a * 100 for a in history['val_acc']]
    ax4.plot(epochs, train_acc, 'b-', label='Train', linewidth=2)
    ax4.plot(epochs, val_acc, 'r-', label='Validation', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy (%)')
    ax4.set_title('Character Accuracy')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=80, color='g', linestyle='--', alpha=0.5, label='80% target')
    
    # 5. Learning Rate
    ax5 = axes[1, 1]
    ax5.plot(epochs, history['learning_rate'], 'g-', linewidth=2)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Learning Rate')
    ax5.set_title('Learning Rate Schedule')
    ax5.set_yscale('log')
    ax5.grid(True, alpha=0.3)
    
    # 6. CTC Weight & Teacher Forcing
    ax6 = axes[1, 2]
    ax6.plot(epochs, history['ctc_weight'], 'purple', label='CTC Weight (λ)', linewidth=2)
    ax6.plot(epochs, history['teacher_forcing'], 'orange', label='Teacher Forcing', linewidth=2)
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Value')
    ax6.set_title('Training Schedule')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()


def print_summary(history):
    """Print training summary"""
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    epochs = len(history['train_loss'])
    best_val_loss_idx = np.argmin(history['val_loss'])
    best_val_acc_idx = np.argmax(history['val_acc'])
    
    print(f"Total epochs trained: {epochs}")
    print(f"\nBest Validation Loss:")
    print(f"  Epoch {best_val_loss_idx + 1}: {history['val_loss'][best_val_loss_idx]:.4f}")
    print(f"\nBest Validation Accuracy:")
    print(f"  Epoch {best_val_acc_idx + 1}: {history['val_acc'][best_val_acc_idx] * 100:.2f}%")
    
    print(f"\nFinal Metrics (Epoch {epochs}):")
    print(f"  Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"  Val Loss:   {history['val_loss'][-1]:.4f}")
    print(f"  Train Acc:  {history['train_acc'][-1] * 100:.2f}%")
    print(f"  Val Acc:    {history['val_acc'][-1] * 100:.2f}%")
    
    # Check if overfitting
    if epochs > 5:
        train_trend = np.mean(history['train_loss'][-5:]) - np.mean(history['train_loss'][-10:-5]) if epochs > 10 else 0
        val_trend = np.mean(history['val_loss'][-5:]) - np.mean(history['val_loss'][-10:-5]) if epochs > 10 else 0
        
        if train_trend < 0 and val_trend > 0:
            print("\n⚠️  Warning: Possible overfitting detected!")
            print("   Train loss decreasing but val loss increasing.")
    
    print("="*60)


def main():
    history_path = Path("checkpoints_hybrid/training_history.json")
    
    if not history_path.exists():
        print(f"History file not found: {history_path}")
        print("Make sure training has completed at least one epoch.")
        return
    
    history = load_history(history_path)
    
    if len(history['train_loss']) == 0:
        print("No training data yet. Wait for at least one epoch to complete.")
        return
    
    print_summary(history)
    
    # Save plot
    plot_path = Path("checkpoints_hybrid/training_curves.png")
    plot_training_curves(history, save_path=plot_path)


if __name__ == "__main__":
    main()
