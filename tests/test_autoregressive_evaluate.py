import unittest
import torch
from src.model import AutoRegressiveTransformerModel
from src.prepare_data import build_vocab
from src.evaluate import evaluateModel

class TestAutoRegressiveEvaluate(unittest.TestCase):
    def test_evaluate_runs(self):
        vocab = build_vocab()
        ntokens = len(vocab)
        model = AutoRegressiveTransformerModel(ntokens, 32, 8, 128, 2, 0.1)
        model.eval()
        # Tiny synthetic check: ensure no exception
        ok = evaluateModel(model, "1+2", max_len=10, print_result=False, print_correct=False)
        self.assertIsInstance(ok, bool)

if __name__ == '__main__':
    unittest.main()
