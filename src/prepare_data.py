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

    def __init__(self, vocab: dict, 
                num_examples: int = 10000,
                depth: int = 3,
                max_digits: int = 2,
                prob_little_endian: float = 0.5,
                prob_scratchpad_full: float = 0.3,
                prob_scratchpad_carry: float = 0.3,
                prob_scratchpad_muldiv: float = 0.3,
                seed = 42):
        self.vocab = vocab
        self.num_examples = num_examples
        self.depth = depth
        self.max_digits = max_digits
        # configure generator
        self.gen_cfg = GenConfig(
            max_depth_cap=depth,
            min_digits=1,
            max_digits=max_digits,
            seed=seed,
            prob_little_endian=prob_little_endian,
            prob_scratchpad_full=prob_scratchpad_full,
            prob_scratchpad_carry=prob_scratchpad_carry,
            prob_scratchpad_muldiv=prob_scratchpad_muldiv)
        self.gen = stream_samples(self.gen_cfg)

    def __len__(self):
        return self.num_examples

    def str_to_token(self,str):
        tokens = []
        for ch in str:
            if ch not in self.vocab:
                # add unknown? raise for visibility (vocab must include all needed chars)
                print(f"Unknown char '{ch}' in sample: {full}")
                raise KeyError(
                    f"Character '{ch}' missing from vocab. Expand vocab for scratchpad tokens.")
            tokens.append(self.vocab[ch])
        return tokens

    def __getitem__(self, idx):
        sample = next(self.gen)
        tokens = [self.vocab['<sos>']]
        if sample.context.endian == 'little':
            tokens.append(self.vocab['<little>'])
        else:
            tokens.append(self.vocab['<big>'])
        tokens_expr = self.str_to_token(sample.expr)
        tokens_scratchpad = self.str_to_token(sample.scratch)
        tokens_result = self.str_to_token(sample.result)
        tokens.extend(tokens_expr)
        tokens.append(self.vocab['<scratchpad>'])
        tokens.extend(tokens_scratchpad)
        tokens.append(self.vocab['<answer>'])
        tokens.extend(tokens_result)
        tokens.append(self.vocab['<eos>'])
        return torch.tensor(tokens, dtype=torch.long)


def build_vocab():
    # Extend vocab with scratchpad markup symbols and lowercase letters
    base_chars = list("0123456789.+-*/()=^")
    markup_chars = list("[], ;")
    # Add lowercase letters for scratchpad text (e.g., "carry")
    alpha_chars = list("abcdefghijklmnopqrstuvwxyz")
    chars = sorted(set(base_chars + markup_chars + alpha_chars))
    vocab = {char: i for i, char in enumerate(chars)}
    vocab['<scratchpad>'] = len(vocab)
    vocab['<sos>'] = len(vocab)
    vocab['<eos>'] = len(vocab)
    vocab['<big>'] = len(vocab)
    vocab['<little>'] = len(vocab)
    vocab['<answer>'] = len(vocab)
    return vocab


def collate_fn_autoregressive(batch, pad_value):
    seqs = pad_sequence(batch, padding_value=pad_value, batch_first=True)
    return seqs
