# mathFormer
simple math expression trained transformer.

A project to train a transformer model on randomly generated math expressions.

use sample:
 uv run python -m src.train

viewer (requires pretrained models in checkpoints folder):
 uv run python -m uvicorn viewer.server:app --port 8000