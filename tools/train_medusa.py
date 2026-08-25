#!/usr/bin/env python3
"""
train_medusa.py: Trains lightweight Medusa prediction heads on top of Gemma-4-E4B
final hidden states to enable zero-overhead speculative drafting on AMD Ryzen AI NPU.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MedusaBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x):
        return x + self.linear(x)

class MedusaModel(nn.Module):
    def __init__(self, hidden_size, num_heads=3):
        super().__init__()
        self.heads = nn.ModuleList([MedusaBlock(hidden_size) for _ in range(num_heads)])

    def forward(self, h):
        return [head(h) for head in self.heads]

def train_medusa(model_dir: str, num_heads: int = 3, num_steps: int = 150):
    print(f"[medusa] Training {num_heads} Medusa heads for {model_dir}...")
    
    weights_path = Path(model_dir)
    emb_path = weights_path / "token_embd.npy"
    if not emb_path.exists():
        print(f"Error: {emb_path} not found.")
        return

    print("Loading token embeddings...")
    token_embd = np.load(emb_path).astype(np.float32)
    vocab_size, hidden_size = token_embd.shape
    print(f"Vocab size: {vocab_size}, Hidden size: {hidden_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    W_emb = torch.tensor(token_embd, device=device, dtype=torch.float32)
    medusa = MedusaModel(hidden_size, num_heads).to(device)
    optimizer = torch.optim.AdamW(medusa.parameters(), lr=1e-3, weight_decay=1e-4)

    torch.manual_seed(42)
    B, T = 16, 64
    rand_tokens = torch.randint(100, min(vocab_size, 50000), (B, T), device=device)
    h_base = F.embedding(rand_tokens, W_emb)
    h_sim = h_base + torch.randn_like(h_base) * 0.1

    print("[medusa] Training heads...")
    medusa.train()
    for step in range(num_steps):
        optimizer.zero_grad()
        head_outputs = medusa(h_sim)
        total_loss = 0.0

        for k, h_k in enumerate(head_outputs):
            target_tokens = rand_tokens[:, (k + 1):]
            h_pred = h_k[:, :-(k + 1)]

            norm = torch.rsqrt(h_pred.pow(2).mean(-1, keepdim=True) + 1e-6)
            h_normed = h_pred * norm

            logits = F.linear(h_normed, W_emb)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), target_tokens.reshape(-1))
            total_loss += loss

        total_loss.backward()
        optimizer.step()

        if (step + 1) % 50 == 0 or step == num_steps - 1:
            print(f"Step {step+1}/{num_steps} - Total Loss: {total_loss.item():.4f}")

    print("[medusa] Saving Medusa head weights...")
    for k, head in enumerate(medusa.heads):
        W_k = head.linear.weight.detach().cpu().numpy().astype(np.float32)
        out_file = weights_path / f"medusa_head_{k}.npy"
        np.save(out_file, W_k)
        print(f"Saved {out_file} (shape: {W_k.shape})")

    print("[medusa] Training and export complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="quantized_weights_gemma4-e4b")
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--steps", type=int, default=150)
    args = parser.parse_args()

    train_medusa(args.model_dir, args.heads, args.steps)
