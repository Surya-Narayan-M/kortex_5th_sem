import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm

# ==================== FILTER DATASET ====================
def filter_dataset_by_landmarks(csv_path, landmarks_dir, output_csv_path):
    """
    Filter the CSV to only include rows where landmark files exist.

    Args:
        csv_path: Path to original CSV file
        landmarks_dir: Directory containing .npy landmark files
        output_csv_path: Path to save filtered CSV
    """
    print("Loading original CSV...")
    df = pd.read_csv(csv_path)
    print(f"Original dataset size: {len(df)} samples")

    landmarks_path = Path(landmarks_dir)

    # Get list of available landmark files
    print("\nScanning for available landmark files...")
    available_files = set()
    for file in tqdm(landmarks_path.glob("*.npy")):
        # Extract uid from filename (remove .npy extension)
        uid = file.stem
        available_files.add(uid)

    print(f"Found {len(available_files)} landmark files")

    # Filter dataframe
    print("\nFiltering dataset...")
    filtered_df = df[df['uid'].isin(available_files)]

    print(f"Filtered dataset size: {len(filtered_df)} samples")
    print(f"Removed: {len(df) - len(filtered_df)} samples")

    # Save filtered dataset
    filtered_df.to_csv(output_csv_path, index=False)
    print(f"\n✓ Filtered dataset saved to: {output_csv_path}")

    # Show sample
    print("\nSample of filtered data:")
    print(filtered_df.head(10))

    return filtered_df


# ==================== CHECK DATASET INTEGRITY ====================
def check_dataset_integrity(csv_path, landmarks_dir):
    """
    Verify that all UIDs in CSV have corresponding landmark files.
    """
    df = pd.read_csv(csv_path)
    landmarks_path = Path(landmarks_dir)

    print(f"Checking {len(df)} samples...")
    missing_files = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        uid = row['uid']
        landmark_file = landmarks_path / f"{uid}.npy"

        if not landmark_file.exists():
            missing_files.append(uid)

    if missing_files:
        print(f"\n⚠️  Warning: {len(missing_files)} landmark files are missing!")
        print("First 10 missing UIDs:")
        for uid in missing_files[:10]:
            print(f"  - {uid}")
    else:
        print("\n✓ All landmark files exist! Dataset is ready for training.")

    return len(missing_files) == 0


# ==================== GET DATASET STATISTICS ====================
def get_dataset_stats(csv_path, landmarks_dir):
    """
    Get statistics about the dataset and landmark files.
    """
    df = pd.read_csv(csv_path)
    landmarks_path = Path(landmarks_dir)

    print("=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    print(f"\nCSV Statistics:")
    print(f"  Total samples: {len(df)}")
    print(f"  Unique texts: {df['text'].nunique()}")
    print(f"  Average text length: {df['text'].str.len().mean():.1f} characters")

    print(f"\nLandmark Files:")
    landmark_files = list(landmarks_path.glob("*.npy"))
    print(f"  Total files: {len(landmark_files)}")

    if landmark_files:
        # Sample a few files to get statistics
        import numpy as np
        sample_files = landmark_files[:min(10, len(landmark_files))]

        shapes = []
        for file in sample_files:
            data = np.load(file)
            shapes.append(data.shape)

        print(f"\n  Sample file shapes (first 10):")
        for i, shape in enumerate(shapes):
            print(f"    {sample_files[i].name}: {shape}")

        # Calculate average frames
        avg_frames = sum(s[0] for s in shapes) / len(shapes)
        print(f"\n  Average frames: {avg_frames:.1f}")
        print(f"  Landmark dimensions: {shapes[0][1:]} (num_landmarks, coords)")

    print("=" * 60)





# ==================== MAIN FUNCTION ====================
def main():
    """
    Main function to run all preprocessing steps.
    Update these paths for your setup!
    """

    # UPDATE THESE PATHS
    original_csv = '/content/iSign_v1.1.csv'
    landmarks_directory = '/content/landmarks'
    filtered_csv = '/content/filtered_data.csv'

    print("=" * 60)
    print("SIGN LANGUAGE DATASET PREPROCESSING")
    print("=" * 60)

    # Step 1: Filter dataset
    print("\n[STEP 1] Filtering dataset by available landmarks...")
    filtered_df = filter_dataset_by_landmarks(
        original_csv,
        landmarks_directory,
        filtered_csv
    )

    # Step 2: Verify integrity
    print("\n[STEP 2] Verifying dataset integrity...")
    is_valid = check_dataset_integrity(filtered_csv, landmarks_directory)

    # Step 3: Show statistics
    print("\n[STEP 3] Dataset statistics...")
    get_dataset_stats(filtered_csv, landmarks_directory)

    if is_valid:
        print("\n" + "=" * 60)
        print("✓ PREPROCESSING COMPLETE!")
        print(f"✓ Use this CSV for training: {filtered_csv}")
        print(f"✓ Dataset size: {len(filtered_df)} samples")
        print("=" * 60)
    else:
        print("\n⚠️  Some landmark files are missing. Please check!")


if __name__ == "__main__":
    main()


# ==================== QUICK COLAB COMMANDS ====================
"""
# In Google Colab, run:

# 1. Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Run the preprocessing
main()

# 3. Update the training script to use the filtered CSV:
config = {
    'csv_path': '/content/drive/MyDrive/sign_language/train_filtered.csv',  # <-- Use filtered CSV
    'landmarks_dir': '/content/drive/MyDrive/sign_language/landmarks/',
    # ... rest of config
}
"""

# ======================================================
# SIGN LANGUAGE TO TEXT TRAINING (SINGLE CELL VERSION)
# Encoder: Bi-GRU + Attention + CTC
# Decoder: T5-small + LoRA
# ======================================================

# -------------------- SETUP --------------------
import os, time, json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType
from google.colab import drive

print("=" * 60)
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️ No GPU detected!")
print("=" * 60)

drive.mount('/content/drive')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------- DATASET FILTERING --------------------
def filter_dataset_by_landmarks(csv_path, landmarks_dir, output_csv_path):
    df = pd.read_csv(csv_path)
    available = {f.stem for f in Path(landmarks_dir).glob("*.npy")}
    df = df[df["uid"].isin(available)]
    df.to_csv(output_csv_path, index=False)
    print(f"Filtered dataset saved: {output_csv_path} | Samples: {len(df)}")
    return df

filter_dataset_by_landmarks(
    csv_path="/content/filtered_data.csv",
    landmarks_dir="/content/landmarks",
    output_csv_path="/content/filtered_training_data.csv"
)

# -------------------- DATASET --------------------
class SignLanguageDataset(Dataset):
    def __init__(self, csv_path, landmarks_dir, max_frames=150):
        self.df = pd.read_csv(csv_path)
        self.landmarks_dir = Path(landmarks_dir)
        self.max_frames = max_frames

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        uid, text = row["uid"], row["text"]
        landmarks = np.load(self.landmarks_dir / f"{uid}.npy")

        frames = landmarks.shape[0]
        if frames < self.max_frames:
            landmarks = np.pad(landmarks, ((0, self.max_frames - frames), (0, 0)))
        else:
            landmarks = landmarks[:self.max_frames]
            frames = self.max_frames

        return {
            "landmarks": torch.FloatTensor(landmarks),
            "text": text,
            "length": frames
        }

def collate_fn(batch):
    return (
        torch.stack([b["landmarks"] for b in batch]),
        [b["text"] for b in batch],
        torch.LongTensor([b["length"] for b in batch])
    )

# -------------------- Bi-GRU ENCODER --------------------
class SignLanguageEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2,
                 num_heads=4, num_glosses=5000):
        super().__init__()

        self.proj = nn.Linear(input_dim, hidden_dim)

        self.bigru = nn.GRU(
            hidden_dim,
            hidden_dim // 2,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )

        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=0.1
        )

        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ctc = nn.Linear(hidden_dim, num_glosses + 1)

    def forward(self, x, lengths):
        x = self.proj(x)

        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.bigru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        attn_out, _ = self.attn(out, out, out)
        x = self.norm1(out + attn_out)

        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return torch.log_softmax(self.ctc(x), dim=-1)

# -------------------- GLOSS VOCAB --------------------
class GlossVocabulary:
    def __init__(self):
        self.gloss2idx = {"<blank>": 0, "<pad>": 1}
        self.idx2gloss = {0: "<blank>", 1: "<pad>"}
        self.next = 2

    def add(self, g):
        if g not in self.gloss2idx:
            self.gloss2idx[g] = self.next
            self.idx2gloss[self.next] = g
            self.next += 1

# -------------------- TRAIN FUNCTIONS --------------------
def train_encoder(model, loader, opt, crit, vocab):
    model.train()
    loss_sum = 0

    for lm, texts, lengths in tqdm(loader):
        lm, lengths = lm.to(DEVICE), lengths.to(DEVICE)

        targets, tgt_len = [], []
        for t in texts:
            words = t.lower().split()[:20]
            for w in words:
                vocab.add(w)
            ids = [vocab.gloss2idx[w] for w in words]
            targets.extend(ids)
            tgt_len.append(len(ids))

        targets = torch.LongTensor(targets).to(DEVICE)
        tgt_len = torch.LongTensor(tgt_len).to(DEVICE)

        logp = model(lm, lengths)
        loss = crit(logp.permute(1, 0, 2), targets, lengths, tgt_len)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        loss_sum += loss.item()

    return loss_sum / len(loader)

def train_decoder(encoder, decoder, loader, opt, tokenizer):
    encoder.eval()
    decoder.train()
    loss_sum = 0

    for lm, texts, lengths in tqdm(loader):
        lm, lengths = lm.to(DEVICE), lengths.to(DEVICE)

        with torch.no_grad():
            preds = torch.argmax(encoder(lm, lengths), dim=-1)

        gloss_texts = [
            " ".join([f"G{i}" for i in p[:l].cpu().numpy()])
            for p, l in zip(preds, lengths)
        ]

        inputs = tokenizer(gloss_texts, padding=True, return_tensors="pt").to(DEVICE)
        labels = tokenizer(texts, padding=True, return_tensors="pt").input_ids.to(DEVICE)
        labels[labels == tokenizer.pad_token_id] = -100

        loss = decoder(**inputs, labels=labels).loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_sum += loss.item()

    return loss_sum / len(loader)

# -------------------- MAIN --------------------
dataset = SignLanguageDataset(
    "/content/filtered_training_data.csv",
    "/content/landmarks"
)
loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

input_dim = dataset[0]["landmarks"].shape[1]
encoder = SignLanguageEncoder(input_dim).to(DEVICE)
vocab = GlossVocabulary()

tokenizer = T5Tokenizer.from_pretrained("t5-small")
decoder = T5ForConditionalGeneration.from_pretrained("t5-small")

decoder = get_peft_model(
    decoder,
    LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q", "v"]
    )
).to(DEVICE)

enc_opt = optim.AdamW(encoder.parameters(), lr=1e-4)
dec_opt = optim.AdamW(decoder.parameters(), lr=1e-4)
ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

training_history = {
    "encoder_loss": [],
    "decoder_loss": [],
    "average_loss": []
}


num_epochs = 60

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")

    enc_loss = train_encoder(encoder, loader, enc_opt, ctc_loss, vocab)
    dec_loss = train_decoder(encoder, decoder, loader, dec_opt, tokenizer)

    avg_loss = (enc_loss + dec_loss) / 2

    # 🔹 Save losses
    training_history["encoder_loss"].append(enc_loss)
    training_history["decoder_loss"].append(dec_loss)
    training_history["average_loss"].append(avg_loss)

    print(f"Encoder Loss : {enc_loss:.4f}")
    print(f"Decoder Loss : {dec_loss:.4f}")
    print(f"Average Loss : {avg_loss:.4f}")

print("✅ Training Completed")

history_path = "/content/training_history.json"

with open(history_path, "w") as f:
    json.dump(training_history, f, indent=4)

print(f"✅ Training history saved to {history_path}")

import matplotlib.pyplot as plt
import json

# Load history
with open("/content/training_history.json", "r") as f:
    history = json.load(f)

epochs = range(1, len(history["encoder_loss"]) + 1)

plt.figure(figsize=(12, 5))

# Encoder & Decoder loss
plt.subplot(1, 2, 1)
plt.plot(epochs, history["encoder_loss"], label="Encoder Loss")
plt.plot(epochs, history["decoder_loss"], label="Decoder Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Encoder & Decoder Loss")
plt.legend()
plt.grid(True)

# Average loss
plt.subplot(1, 2, 2)
plt.plot(epochs, history["average_loss"], label="Average Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Average Training Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()