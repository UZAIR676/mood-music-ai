
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from tqdm import tqdm

from model.transformer import MusicTransformer
from model.dataset import get_dataloader
from train.config import *


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data load karo
    loader = get_dataloader(DATASET_PATH, max_files=50)

    if len(loader) == 0:
        print("ERROR: Koi MIDI files nahi mili!")
        print(f"Yeh folder check karo: {DATASET_PATH}")
        return

    # Model banao
    model = MusicTransformer().to(device)
    print(f"Model parameters: {model.count_params():,}")

    # Optimizer aur loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Training loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0

        for x, y in tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = criterion(
                logits.view(-1, VOCAB_SIZE),
                y.view(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            total_loss += loss.item()

        avg = total_loss / len(loader)
        print(f"Epoch {epoch} | Loss: {avg:.4f}")

        # Model save karo
        torch.save(
            model.state_dict(),
            os.path.join(CHECKPOINT_DIR, f"model_epoch{epoch}.pt")
        )
        print(f"Checkpoint saved!")