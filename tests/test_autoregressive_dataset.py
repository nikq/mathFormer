import unittest
from src.prepare_data import MathExprDataset, build_vocab
from src.evaluate import split_scratchpad_and_result

class TestAutoregressiveDataset(unittest.TestCase):
    def test_sequence_format(self):
        vocab = build_vocab()
        dataset = MathExprDataset(vocab, num_examples=5, depth=3, max_digits=3, prob_scratchpad=1.0)
        seq = dataset[0]
        # seq is a tensor of token ids starting with <sos>
        inv_vocab = {i:ch for ch,i in vocab.items()}
        decoded = ''.join(inv_vocab[i] for i in seq.tolist()[1:-1])  # strip <sos>, <eos>
        # Must contain exactly two '=' if process present else one '='
        self.assertIn('<scratchpad>', decoded)
        self.assertIn('<answer>', decoded)
        # expression part before first '=' should contain an operator or single number
        scratchpad_part, result_part = split_scratchpad_and_result(decoded)
        print(f"Decoded sequence: {decoded} -> scratchpad: {scratchpad_part}, result: {result_part}")
        self.assertTrue(any(op in scratchpad_part for op in '+-*/') or scratchpad_part.isdigit() or scratchpad_part == "")

if __name__ == '__main__':
    unittest.main()
