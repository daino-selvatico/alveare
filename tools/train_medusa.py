#!/usr/bin/env python3
"""
train_medusa.py: Trains lightweight Medusa prediction heads on real token sequences
for Gemma-4-E4B to achieve high-acceptance (>70%) speculative drafting on AMD Ryzen AI NPU.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

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

def train_medusa(model_dir: str, corpus_file: str = "tools/corpus.jsonl", num_heads: int = 3, num_epochs: int = 40, batch_size: int = 8, lr: float = 3e-3):
    weights_path = Path(model_dir)
    emb_path = weights_path / "token_embd.npy"
    if not emb_path.exists():
        print(f"Error: {emb_path} not found.")
        return

    print(f"[medusa] Loading embeddings and model from {model_dir}...")
    token_embd = np.load(emb_path).astype(np.float32)
    vocab_size, hidden_size = token_embd.shape
    print(f"Vocab size: {vocab_size}, Hidden size: {hidden_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # Load tokenizer
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
    
    # Load corpus
    print(f"[medusa] Loading corpus from {corpus_file}...")
    sequences = []
    max_len = 96
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            text = obj.get("text", "")
            if text:
                toks = tokenizer.encode(text).ids
                # Chunk into fixed max_len chunks
                for i in range(0, len(toks), max_len // 2):
                    chunk = toks[i:i + max_len]
                    if len(chunk) >= num_heads + 8:
                        sequences.append(chunk)

    print(f"[medusa] Prepared {len(sequences)} training sequence chunks.")

    W_emb = torch.tensor(token_embd, device=device, dtype=torch.float32)
    medusa = MedusaModel(hidden_size, num_heads).to(device)
    optimizer = torch.optim.AdamW(medusa.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    print("[medusa] Starting training loop...")
    medusa.train()
    
    num_samples = len(sequences)
    indices = np.arange(num_samples)

    for epoch in range(num_epochs):
        np.random.shuffle(indices)
        total_loss = 0.0
        total_batches = 0
        total_correct = [0] * num_heads
        total_tokens = 0

        for b_start in range(0, num_samples, batch_size):
            batch_idx = indices[b_start:b_start + batch_size]
            batch_seqs = [sequences[idx] for idx in batch_idx]
            min_l = min(len(s) for s in batch_seqs)
            
            # Form tensor
            batch_data = [s[:min_l] for s in batch_seqs]
            t_tensor = torch.tensor(batch_data, dtype=torch.long, device=device) # (B, L)
            
            # Embeddings as base hidden states
            h = F.embedding(t_tensor, W_emb) # (B, L, H)
            
            optimizer.zero_grad()
            head_outputs = medusa(h)
            loss = 0.0

            for k, h_k in enumerate(head_outputs):
                target = t_tensor[:, (k + 1):min_l] # (B, L - 1 - k)
                h_pred = h_k[:, :-(k + 1)]          # (B, L - 1 - k, H)

                # RMSNorm
                norm = torch.rsqrt(h_pred.pow(2).mean(-1, keepdim=True) + 1e-6)
                h_normed = h_pred * norm

                # Logits via tied embedding
                logits = F.linear(h_normed, W_emb) # (B, L - 1 - k, V)
                
                flat_logits = logits.reshape(-1, vocab_size)
                flat_targets = target.reshape(-1)
                
                head_loss = F.cross_entropy(flat_logits, flat_targets)
                loss += head_loss

                # Calculate accuracy
                with torch.no_grad():
                    preds = torch.argmax(flat_logits, dim=-1)
                    correct = (preds == flat_targets).sum().item()
                    total_correct[k] += correct

            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_batches += 1
            total_tokens += target.numel()

        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            avg_loss = total_loss / max(1, total_batches)
            acc_str = ", ".join([f"H{k+1}: {100.0 * total_correct[k] / max(1, total_tokens):.1f}%" for k in range(num_heads)])
            print(f"Epoch {epoch+1:02d}/{num_epochs:02d} | Loss: {avg_loss:.4f} | Accuracy: [{acc_str}]")

    print("[medusa] Saving Medusa head weights to model directory...")
    for k, head in enumerate(medusa.heads):
        W_k = head.linear.weight.detach().cpu().numpy().astype(np.float32)
        out_file = weights_path / f"medusa_head_{k}.npy"
        np.save(out_file, W_k)
        print(f"Saved {out_file} (shape: {W_k.shape})")

    print("[medusa] All heads saved successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="quantized_weights_gemma4-e4b")
    parser.add_argument("--corpus", type=str, default="tools/corpus.jsonl")
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-3)
    args = parser.parse_args()

    train_medusa(args.model_dir, args.corpus, args.heads, args.epochs, args.batch_size, args.lr)
