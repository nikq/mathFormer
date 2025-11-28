import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from src.generate_data import generate_sample, GenConfig, stream_samples


class MathExprDataset(Dataset):
    """Autoregressive-only dataset producing single sequences.

    Each sample is built as: <sos>{input}={target}<eos>
    Where generate_sample may include scratchpad sections:
      input: possibly denoised expression
      target: either result or <scratch>... </scratch><final>=RESULT</final>
    We collapse to a linear form: input=target_text
    Example:
      input: 14*3
      target: <scratch>10*3=30, 30+4*3=42</scratch><final>=42</final>
      sequence tokens model sees: <sos>14*3=<scratch>10*3=30, 30+4*3=42</scratch><final>=42</final><eos>
    """

    def __init__(self, vocab: dict, num_examples: int = 10000, depth: int = 3, max_digits: int = 2, prob_little_endian: float = 0.5):
        self.vocab = vocab
        self.num_examples = num_examples
        self.depth = depth
        self.max_digits = max_digits
        # configure generator
        self.gen_cfg = GenConfig(
            max_depth_cap=depth,
            min_digits=1,
            max_digits=max_digits,
            seed=42,
            prob_scratchpad_full=1.0,
            prob_little_endian=prob_little_endian)
        self.gen = stream_samples(self.gen_cfg)

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        sample = next(self.gen)
        # linearize: input '=' target
        full = f"{sample['input']}{' [' + sample['scratch'] +']' if sample['scratch'] else ''}; {sample['target']}"
        # print(full)
        # Map chars; unknown chars (e.g. '<','>','/','scratch','final') must be in vocab.
        tokens = [self.vocab['<sos>']]
        
        # Add endian token
        if sample.get('endian') == 'little':
            tokens.append(self.vocab['<little>'])
        else:
            tokens.append(self.vocab['<big>'])
            
        for ch in full:
            if ch not in self.vocab:
                # add unknown? raise for visibility (vocab must include all needed chars)
                print(f"Unknown char '{ch}' in sample: {full}")
                raise KeyError(
                    f"Character '{ch}' missing from vocab. Expand vocab for scratchpad tokens.")
            tokens.append(self.vocab[ch])
        tokens.append(self.vocab['<eos>'])
        return torch.tensor(tokens, dtype=torch.long)


def build_vocab():
    # Extend vocab with scratchpad markup symbols
    base_chars = list("0123456789.+-*/()=^")
    markup_chars = list("[], ;")
    chars = sorted(set(base_chars + markup_chars))
    vocab = {char: i for i, char in enumerate(chars)}
    vocab['<pad>'] = len(vocab)
    vocab['<sos>'] = len(vocab)
    vocab['<eos>'] = len(vocab)
    vocab['<big>'] = len(vocab)
    vocab['<little>'] = len(vocab)
    return vocab


def collate_fn_autoregressive(batch, pad_value):
    seqs = pad_sequence(batch, padding_value=pad_value, batch_first=True)
    return seqs
