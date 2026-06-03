
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import Dataset, DataLoader

from data.loader import get_all_midi_files, load_midi
from data.tokenizer import notes_to_tokens
from train.config import *


class MaestroDataset(Dataset):
    """
    MIDI files ko load karo aur
    model ke liye chunks banao
    """
    def __init__(self, dataset_path, max_files=50):
        print(f"Loading MIDI files...")
        files = get_all_midi_files(dataset_path)[:max_files]
        print(f"Total files: {len(files)}")

        all_tokens = []
        for f in files:
            notes  = load_midi(f)
            tokens = notes_to_tokens(notes)
            all_tokens.extend(tokens)

        # Tokens ko SEQ_LEN size ke chunks mein kato
        self.chunks = []
        for i in range(0, len(all_tokens) - SEQ_LEN - 1, SEQ_LEN):
            chunk = all_tokens[i : i + SEQ_LEN + 1]
            self.chunks.append(chunk)

        print(f"Total chunks: {len(self.chunks)}")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)  # input
        y = torch.tensor(chunk[1:],  dtype=torch.long)  # target
        return x, y


def get_dataloader(dataset_path, max_files=50):
    ds = MaestroDataset(dataset_path, max_files)
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0       # Windows ke liye 0 zaroori hai
    )