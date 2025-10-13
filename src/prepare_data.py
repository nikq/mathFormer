import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from src.generate_data import generate_expression


class MathExprDataset(Dataset):
    def __init__(self, vocab, num_examples=10000, depth=10, min_digits=1, max_digits=1):
        self.vocab = vocab
        self.num_examples = num_examples
        self.depth = depth
        self.min_digits = min_digits
        self.max_digits = max_digits

        print("MathExprDataset initialized.")
        print(f"vocab size: {len(self.vocab)}")  # デバッグ用出力
        print(f"num_examples: {self.num_examples}")  # デバッグ用出力
        print(f"depth: {self.depth}")  # デバッグ用出力
        print(f"min_digits: {self.min_digits}")  # デバッグ用出力
        print(f"max_digits: {self.max_digits}")  # デバッグ用出力

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        expr, result = generate_expression(
            max_depth=self.depth,
            min_digits=self.min_digits,
            max_digits=self.max_digits
        )
        # print(f"Generated: {expr}={result}")  # デバッグ用出力
        src = [self.vocab['<sos>']] + [self.vocab[char] for char in expr] + [self.vocab['<eos>']]
        tgt = [self.vocab['<sos>']] + [self.vocab[char] for char in str(result)] + [self.vocab['<eos>']]
        return torch.tensor(src), torch.tensor(tgt)


def build_vocab():
    chars = sorted(list("0123456789.+-*/()"))
    vocab = {char: i for i, char in enumerate(chars)}
    vocab['<pad>'] = len(vocab)
    vocab['<sos>'] = len(vocab)
    vocab['<eos>'] = len(vocab)
    return vocab


def collate_fn(batch, pad_value):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)

    src_batch = pad_sequence(src_batch, padding_value=pad_value, batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=pad_value, batch_first=True)
    return src_batch, tgt_batch