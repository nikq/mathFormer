import unittest
from src.generate_data import generate_sample, GenConfig

class TestGenerateData(unittest.TestCase):
    def test_generate_sample(self):
        cfg = GenConfig(max_depth_cap=4, min_digits=1, max_digits=2, seed=123)
        sample = generate_sample(cfg)
        expr = sample['expr']
        result = sample['result']
        target = sample['target']
        self.assertIsInstance(expr, str)
        self.assertIsInstance(result, str)
        self.assertIsInstance(target, str)
        # Expression should contain at least one digit
        self.assertRegex(expr, r"[0-9]")
        # Result should be numeric or fraction
        self.assertRegex(result, r"^[0-9/-]+$")
        # If scratchpad used, target contains tags
        if '<scratch>' in target or '<final>' in target:
            self.assertIn('<final>=', target)

if __name__ == '__main__':
    unittest.main()
