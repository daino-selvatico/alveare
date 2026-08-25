#!/usr/bin/env python3
"""
download_corpus.py: Downloads and builds a rich, diverse training corpus
for Medusa speculative decoding covering Italian, English, and Code.
"""

import json
import urllib.request
import re
from pathlib import Path

DATA_FILE = Path("tools/corpus.jsonl")

WIKI_TITLES = [
    "Algoritmo",
    "Informatica",
    "Linguaggio_di_programmazione",
    "Struttura_dati",
    "Compilatore",
    "Architettura_dei_calcolatori",
    "Memoria_ad_accesso_casuale",
    "Unità_di_elaborazione_centrale",
    "Calcolo_parallelo",
    "Sistema_operativo",
    "Rete_di_calcolatori",
    "Crittografia",
    "Intelligenza_artificiale",
    "Apprendimento_automatico",
    "Rete_neurale_artificiale",
    "Logica_matematica",
    "Algebra_lineare",
    "Equazione_differenziale",
    "Fisica_teorica",
    "Elettromagnetismo",
    "Termodinamica",
    "Meccanica_razionale"
]

def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def fetch_wiki_articles():
    samples = []
    print("[corpus] Fetching Italian Wikipedia technical articles...")
    for title in WIKI_TITLES:
        url = f"https://it.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext&titles={urllib.parse.quote(title)}&format=json"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'AlveareAIE/1.0 (contact@alveare.ai)'})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                for page_id, page in data.get('query', {}).get('pages', {}).items():
                    extract = page.get('extract', '')
                    if extract:
                        paragraphs = extract.split('\n')
                        for p in paragraphs:
                            p = clean_text(p)
                            if len(p) > 60:
                                samples.append(p)
        except Exception as e:
            print(f"Warning: could not fetch {title}: {e}")
    print(f"[corpus] Extracted {len(samples)} Italian paragraphs from Wikipedia.")
    return samples

def fetch_alpaca_samples():
    samples = []
    url = "https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/main/alpaca_data_cleaned.json"
    print(f"[corpus] Fetching Alpaca dataset from {url}...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'AlveareAIE/1.0'})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            for item in data[:2500]: # 2,500 diverse instructions
                inst = item.get("instruction", "")
                inp = item.get("input", "")
                out = item.get("output", "")
                full = f"{inst}\n{inp}\n{out}".strip()
                if len(full) > 40:
                    samples.append(clean_text(full))
        print(f"[corpus] Extracted {len(samples)} Alpaca instructions.")
    except Exception as e:
        print(f"Warning: could not fetch Alpaca: {e}")
    return samples

def main():
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    all_samples = []

    # 1. Italian Wikipedia
    all_samples.extend(fetch_wiki_articles())

    # 2. Alpaca Instructions (Code / QA / English)
    all_samples.extend(fetch_alpaca_samples())

    # 3. Add explicit code snippets
    code_snippets = [
        """def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)""",
        """def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        L = arr[:mid]
        R = arr[mid:]
        merge_sort(L)
        merge_sort(R)
        i = j = k = 0
        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1
        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1
    return arr""",
        """import numpy as np
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight""",
        """def fibonacci(n):
    a, b = 0, 1
    result = []
    for _ in range(n):
        result.append(a)
        a, b = b, a + b
    return result"""
    ]
    for c in code_snippets:
        all_samples.append(c)

    print(f"[corpus] Total assembled corpus: {len(all_samples)} samples.")
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps({"text": s}, ensure_ascii=False) + "\n")

    print(f"[corpus] Written to {DATA_FILE} successfully.")

if __name__ == "__main__":
    main()
