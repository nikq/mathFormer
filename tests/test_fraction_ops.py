import unittest
from src.generate_data import GenerationContext, scratchpad, Leaf, OpNode, GenConfig

class TestFractionOps(unittest.TestCase):
    def setUp(self):
        self.ctx = GenerationContext(endian='big')
        self.cfg = GenConfig(prob_scratchpad=1.0)

    def test_fraction_addition(self):
        # 1/2 + 1/3
        n1 = OpNode('/', Leaf(1), Leaf(2))
        n2 = OpNode('/', Leaf(1), Leaf(3))
        root = OpNode('+', n1, n2)
        s = scratchpad(root, self.cfg, self.ctx)
        self.assertIn("1/2+(1/3)", s)

    def test_fraction_subtraction_negative(self):
        # -1/2 + 1/3
        n1 = OpNode('/', Leaf(-1), Leaf(2))
        n2 = OpNode('/', Leaf(1), Leaf(3))
        root = OpNode('+', n1, n2)
        s = scratchpad(root, self.cfg, self.ctx)
        self.assertIn("-1/2+(1/3)", s)

    def test_fraction_subtraction(self):
        # 3/4 - 1/6
        n1 = OpNode('/', Leaf(3), Leaf(4))
        n2 = OpNode('/', Leaf(1), Leaf(6))
        root = OpNode('-', n1, n2)
        s = scratchpad(root, self.cfg, self.ctx)
        self.assertIn("3/4-(1/6)", s)

if __name__ == '__main__':
    unittest.main()
