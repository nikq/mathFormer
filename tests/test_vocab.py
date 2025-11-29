#!/usr/bin/env python3
# Test script to verify vocabulary includes 'carry' token

import sys
sys.path.insert(0, 'src')

from src.prepare_data import build_vocab

vocab = build_vocab()

print("Vocabulary size:", len(vocab))
print("\nChecking for 'carry' characters:")
for ch in 'carry':
    if ch in vocab:
        print(f"  '{ch}': ✓ (index {vocab[ch]})")
    else:
        print(f"  '{ch}': ✗ MISSING")

print("\nAll lowercase letters in vocab:")
letters = [ch for ch in vocab if ch.isalpha() and ch.islower()]
print("  " + "".join(sorted(letters)))

print("\nSpecial tokens:")
for token in ['<pad>', '<sos>', '<eos>', '<big>', '<little>', '<scratchpad>', '<answer>']:
    if token in vocab:
        print(f"  {token}: ✓ (index {vocab[token]})")
