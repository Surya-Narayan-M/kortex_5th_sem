"""Check training history"""
import json

hist = json.load(open('checkpoints/training_history.json'))

print("="*60)
print("TRAINING HISTORY ANALYSIS")
print("="*60)

train_loss = hist['train_loss']
val_loss = hist['val_loss']
val_acc = hist['val_accuracy']

print(f"\nEpochs trained: {len(train_loss)}")
print(f"\nTrain Loss:")
print(f"  Initial: {train_loss[0]:.4f}")
print(f"  Final:   {train_loss[-1]:.4f}")
print(f"  Change:  {train_loss[0] - train_loss[-1]:.4f} ({'↓' if train_loss[-1] < train_loss[0] else '↑'})")

print(f"\nValidation Loss:")
print(f"  Initial: {val_loss[0]:.4f}")
print(f"  Final:   {val_loss[-1]:.4f}")
print(f"  Best:    {min(val_loss):.4f} (epoch {val_loss.index(min(val_loss))+1})")

print(f"\nValidation Accuracy:")
print(f"  Initial: {val_acc[0]:.2f}%")
print(f"  Final:   {val_acc[-1]:.2f}%")
print(f"  Best:    {max(val_acc):.2f}% (epoch {val_acc.index(max(val_acc))+1})")

print("\n" + "="*60)

if val_acc[-1] < 5:
    print("⚠️  WARNING: Accuracy is very low!")
    print("   Possible issues:")
    print("   1. Model architecture doesn't fit the data")
    print("   2. Learning rate too high/low")
    print("   3. Data preprocessing issues")
    print("   4. Needs more training epochs")
elif val_acc[-1] < 20:
    print("⚠️  Model learned something but accuracy is still low")
    print("   Consider training longer or adjusting hyperparameters")
else:
    print("✅ Model trained reasonably well")
