
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from train.config import *


class MusicTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        # Har token ko vector mein badlo
        self.embed = nn.Embedding(VOCAB_SIZE, D_MODEL)

        # Position — token kaunse number par hai
        self.pos_embed = nn.Embedding(SEQ_LEN, D_MODEL)

        # Transformer layers — yahan AI seekhta hai
        layer = nn.TransformerDecoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=D_FF,
            dropout=DROPOUT,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(layer, num_layers=N_LAYERS)

        self.dropout = nn.Dropout(DROPOUT)

        # Final output — agla token predict karo
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, x):
        B, T = x.shape

        # Token + position embedding
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        out = self.dropout(self.embed(x) + self.pos_embed(positions))

        # Mask — model sirf pichle tokens dekhe, aage nahi
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)

        out = self.transformer(out, out, tgt_mask=mask, memory_mask=mask)

        return self.head(out)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())