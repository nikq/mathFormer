
import unittest
from fractions import Fraction
from src.generate_data import (
    evaluate, to_string, scratchpad, 
    OpNode, Leaf, UnaryOpNode, 
    GenConfig, GenerationContext, generate_tree, generate_sample, canonicalize
)
import random

class TestNewOps(unittest.TestCase):
    def setUp(self):
        self.ctx = GenerationContext(endian='big')
        self.cfg = GenConfig(
            operators={'max', 'min', 'next', 'prev', 'abs', '%', '+', '-', '*', '/'},
            min_digits=1, max_digits=2,
            prob_scratchpad=1.0
        )

    def test_evaluate_unary(self):
        # next(10) -> 11
        n = UnaryOpNode('next', Leaf(10))
        self.assertEqual(evaluate(n), Fraction(11))
        
        # prev(10) -> 9
        n = UnaryOpNode('prev', Leaf(10))
        self.assertEqual(evaluate(n), Fraction(9))

        # abs(-5) -> 5
        n = UnaryOpNode('abs', Leaf(-5))
        self.assertEqual(evaluate(n), Fraction(5))

    def test_evaluate_binary(self):
        # max(3, 5) -> 5
        n = OpNode('max', Leaf(3), Leaf(5))
        self.assertEqual(evaluate(n), Fraction(5))

        # min(3, 5) -> 3
        n = OpNode('min', Leaf(3), Leaf(5))
        self.assertEqual(evaluate(n), Fraction(3))
        
        # 7 % 3 -> 1
        n = OpNode('%', Leaf(7), Leaf(3))
        self.assertEqual(evaluate(n), Fraction(1))

    def test_to_string(self):
        n = UnaryOpNode('next', Leaf(5))
        self.assertEqual(to_string(n, self.ctx), "next(5)")
        
        n = OpNode('max', Leaf(1), Leaf(2))
        self.assertEqual(to_string(n, self.ctx), "max(1, 2)")
        
        n = OpNode('%', Leaf(5), Leaf(2))
        self.assertEqual(to_string(n, self.ctx), "5 % 2") # % behaves like operator

    def test_scratchpad(self):
        # next(4)
        n = UnaryOpNode('next', Leaf(4))
        s = scratchpad(n, self.cfg, self.ctx)
        self.assertIn("next(4)=4+1=5", s)
        
        # max(1,2)
        n = OpNode('max', Leaf(1), Leaf(2))
        s = scratchpad(n, self.cfg, self.ctx)
        self.assertIn("max(1, 2)=2", s)

    def test_generate_sample(self):
        # Ensure we can generate samples without error
        random.seed(42)
        # Force using only new ops if possible or mix
        cfg = GenConfig(operators={'next', 'prev', 'max', 'min'}, max_depth_cap=3)
        for _ in range(10):
            res = generate_sample(cfg)
            # print(res.expression, res.result)
            self.assertTrue(len(res.expr) > 0)
            self.assertTrue(len(res.result) > 0)

    def test_canonicalize_max_min(self):
        # max(2, 1) -> max(1, 2) (sorted by string rep)
        # "1" < "2"
        n = OpNode('max', Leaf(2), Leaf(1))
        norm = canonicalize(n)
        self.assertEqual(norm.op, 'max')
        self.assertEqual(norm.left.value, 1)
        self.assertEqual(norm.right.value, 2)

if __name__ == '__main__':
    unittest.main()
