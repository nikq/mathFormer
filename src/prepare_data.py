import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from src.generate_data import generate_expression


class MathExprDataset(Dataset):
    def __init__(self, vocab, num_examples=10000, depth=10, min_digits=1, max_digits=1, with_process=True, autoregressive=False):
        self.vocab = vocab
        self.num_examples = num_examples
        self.depth = depth
        self.min_digits = min_digits
        self.max_digits = max_digits
        self.with_process = with_process
        self.autoregressive = autoregressive

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        expr, process, answer = generate_expression(
            max_depth=self.depth,
            min_digits=self.min_digits,
            max_digits=self.max_digits,
            with_process=self.with_process if self.depth > 1 else False
        )
        if self.autoregressive:
            # Build single sequence: expr=process=answer (where process may be empty)
            if process:
                full = f"{expr}={process}={answer}"
            else:
                full = f"{expr}={answer}"
            seq = [self.vocab['<sos>']] + [self.vocab[c] for c in full] + [self.vocab['<eos>']]
            return torch.tensor(seq)
        else:
            if process:
                result = f"{process}={answer}"
            else:
                result = answer
            src = [self.vocab['<sos>']] + [self.vocab[char] for char in expr] + [self.vocab['<eos>']]
            tgt = [self.vocab['<sos>']] + [self.vocab[char] for char in str(result)] + [self.vocab['<eos>']]
            return torch.tensor(src), torch.tensor(tgt)


def build_vocab():
    chars = sorted(list("0123456789.+-*/()="))
    vocab = {char: i for i, char in enumerate(chars)}
    vocab['<pad>'] = len(vocab)
    vocab['<sos>'] = len(vocab)
    vocab['<eos>'] = len(vocab)
    return vocab


def collate_fn(batch, pad_value):
    # Legacy seq2seq collate
    src_batch, tgt_batch = [], []
    for item in batch:
        src_sample, tgt_sample = item
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)
    src_batch = pad_sequence(src_batch, padding_value=pad_value, batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=pad_value, batch_first=True)
    return src_batch, tgt_batch


def collate_fn_autoregressive(batch, pad_value):
    # Batch is list of 1D tensors
    seqs = pad_sequence(batch, padding_value=pad_value, batch_first=True)
    return seqs