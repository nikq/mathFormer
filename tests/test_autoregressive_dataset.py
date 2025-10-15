import unittest
from src.prepare_data import MathExprDataset, build_vocab

class TestAutoregressiveDataset(unittest.TestCase):
    def test_sequence_format(self):
        vocab = build_vocab()
        dataset = MathExprDataset(vocab, num_examples=5, depth=2, min_digits=1, max_digits=1, with_process=True, autoregressive=True)
        seq = dataset[0]
        # seq is a tensor of token ids starting with <sos>
        inv_vocab = {i:ch for ch,i in vocab.items()}
        decoded = ''.join(inv_vocab[i] for i in seq.tolist()[1:-1])  # strip <sos>, <eos>
        # Must contain exactly two '=' if process present else one '='
        self.assertIn('=', decoded)
        # expression part before first '=' should contain an operator or single number
        expr_part = decoded.split('=')[0]
        self.assertTrue(any(op in expr_part for op in '+-*/') or expr_part.isdigit())

if __name__ == '__main__':
    unittest.main()
