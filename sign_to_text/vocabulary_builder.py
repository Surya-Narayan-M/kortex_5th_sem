"""
Vocabulary Builder for Sign Language Dataset
Builds word-level and character-level vocabularies from CSV data
"""

import pandas as pd
import json
from collections import Counter
from pathlib import Path
from tqdm import tqdm

class VocabularyBuilder:
    def __init__(self, csv_path, max_words=5000):
        self.csv_path = csv_path
        self.max_words = max_words
        
        # Special tokens
        self.BLANK = "<blank>"
        self.SPACE = "<space>"
        self.UNK = "<unk>"
        self.PAD = "<pad>"
        
        # Vocabularies
        self.word2idx = {}
        self.idx2word = {}
        self.char2idx = {}
        self.idx2char = {}
        
        self.word_freq = Counter()
        
    def build_from_csv(self):
        """Build vocabulary from CSV file"""
        print(f"Loading data from {self.csv_path}...")
        df = pd.read_csv(self.csv_path)
        print(f"Total samples: {len(df)}")
        
        # Count word frequencies
        print("\nCounting word frequencies...")
        for text in tqdm(df['text']):
            if pd.isna(text):
                continue
            words = str(text).lower().split()
            self.word_freq.update(words)
        
        print(f"Unique words found: {len(self.word_freq)}")
        
        # Build word vocabulary (top N most frequent)
        self._build_word_vocab()
        
        # Build character vocabulary
        self._build_char_vocab()
        
        return self
    
    def _build_word_vocab(self):
        """Build word-level vocabulary"""
        print(f"\nBuilding word vocabulary (top {self.max_words})...")
        
        # Add special tokens first
        self.word2idx[self.BLANK] = 0
        self.word2idx[self.SPACE] = 1
        self.word2idx[self.UNK] = 2
        self.word2idx[self.PAD] = 3
        
        self.idx2word[0] = self.BLANK
        self.idx2word[1] = self.SPACE
        self.idx2word[2] = self.UNK
        self.idx2word[3] = self.PAD
        
        # Add most frequent words
        idx = 4
        for word, freq in self.word_freq.most_common(self.max_words):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            idx += 1
        
        print(f"Word vocabulary size: {len(self.word2idx)}")
        
        # Calculate coverage
        total_words = sum(self.word_freq.values())
        covered_words = sum(freq for word, freq in self.word_freq.most_common(self.max_words))
        coverage = (covered_words / total_words) * 100
        print(f"Vocabulary covers {coverage:.2f}% of all word occurrences")
    
    def _build_char_vocab(self):
        """Build character-level vocabulary (for fallback)"""
        print("\nBuilding character vocabulary...")
        
        # Special tokens
        self.char2idx[self.BLANK] = 0
        self.char2idx[self.SPACE] = 1
        
        self.idx2char[0] = self.BLANK
        self.idx2char[1] = self.SPACE
        
        # Add all printable characters
        chars = "abcdefghijklmnopqrstuvwxyz0123456789.,!?'-:;\"()"
        for idx, char in enumerate(chars, start=2):
            self.char2idx[char] = idx
            self.idx2char[idx] = char
        
        print(f"Character vocabulary size: {len(self.char2idx)}")
    
    def text_to_word_indices(self, text, max_length=50):
        """Convert text to word indices"""
        words = str(text).lower().split()[:max_length]
        indices = [self.word2idx.get(w, self.word2idx[self.UNK]) for w in words]
        return indices
    
    def text_to_char_indices(self, text, max_length=200):
        """Convert text to character indices"""
        text = str(text).lower()[:max_length]
        indices = [self.char2idx.get(c, self.char2idx[self.SPACE]) for c in text]
        return indices
    
    def indices_to_text(self, indices, mode='word'):
        """Convert indices back to text"""
        if mode == 'word':
            words = [self.idx2word.get(idx, self.UNK) for idx in indices 
                    if idx not in [0, 3]]  # Skip blank and pad
            return ' '.join(words)
        else:  # char mode
            chars = [self.idx2char.get(idx, '') for idx in indices 
                    if idx != 0]  # Skip blank
            return ''.join(chars)
    
    def save(self, output_path):
        """Save vocabulary to JSON file"""
        vocab_data = {
            'word2idx': self.word2idx,
            'idx2word': {str(k): v for k, v in self.idx2word.items()},
            'char2idx': self.char2idx,
            'idx2char': {str(k): v for k, v in self.idx2char.items()},
            'word_freq': dict(self.word_freq.most_common(100)),  # Save top 100 for reference
            'stats': {
                'word_vocab_size': len(self.word2idx),
                'char_vocab_size': len(self.char2idx),
                'total_unique_words': len(self.word_freq)
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Vocabulary saved to {output_path}")
    
    @classmethod
    def load(cls, vocab_path):
        """Load vocabulary from JSON file"""
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        vocab = cls.__new__(cls)
        vocab.word2idx = vocab_data['word2idx']
        vocab.idx2word = {int(k): v for k, v in vocab_data['idx2word'].items()}
        vocab.char2idx = vocab_data['char2idx']
        vocab.idx2char = {int(k): v for k, v in vocab_data['idx2char'].items()}
        vocab.BLANK = "<blank>"
        vocab.SPACE = "<space>"
        vocab.UNK = "<unk>"
        vocab.PAD = "<pad>"
        
        return vocab
    
    def print_stats(self):
        """Print vocabulary statistics"""
        print("\n" + "="*60)
        print("VOCABULARY STATISTICS")
        print("="*60)
        print(f"Word vocabulary size: {len(self.word2idx)}")
        print(f"Character vocabulary size: {len(self.char2idx)}")
        print(f"Total unique words in dataset: {len(self.word_freq)}")
        print(f"\nTop 20 most frequent words:")
        for word, freq in self.word_freq.most_common(20):
            print(f"  {word:20s} : {freq:6d}")
        print("="*60)


def main():
    """Build vocabulary from iSign dataset"""
    csv_path = "E:/5thsem el/kortex_5th_sem/iSign_v1.1.csv"
    output_path = "E:/5thsem el/kortex_5th_sem/vocabulary.json"
    
    print("🚀 Building Vocabulary for Sign Language Recognition")
    print("="*60)
    
    # Build vocabulary
    builder = VocabularyBuilder(csv_path, max_words=5000)
    builder.build_from_csv()
    builder.print_stats()
    
    # Save vocabulary
    builder.save(output_path)
    
    # Test conversion
    print("\n📝 Testing conversions...")
    test_text = "Make it shorter."
    word_indices = builder.text_to_word_indices(test_text)
    char_indices = builder.text_to_char_indices(test_text)
    
    print(f"Original text: {test_text}")
    print(f"Word indices: {word_indices}")
    print(f"Reconstructed: {builder.indices_to_text(word_indices, mode='word')}")
    print(f"Char indices: {char_indices}")
    print(f"Reconstructed: {builder.indices_to_text(char_indices, mode='char')}")


if __name__ == "__main__":
    main()
