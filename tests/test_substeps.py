import unittest
import sys
import os
from fractions import Fraction

# make sure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from generate_data import (
    fraction_addition_partial,
    fraction_subtraction_partial,
    fraction_multiplication_partial,
    fraction_division_partial,
    subtraction_with_borrow,
    GenerationContext,
    OpNode,
    Leaf,
    GenConfig,
    scratchpad,
)

class TestSubsteps(unittest.TestCase):
    def setUp(self):
        self.ctx = GenerationContext(endian='big')

    def test_fraction_addition_partial(self):
        steps = fraction_addition_partial(Fraction(1,2), Fraction(1,3), self.ctx)
        self.assertIsInstance(steps, list)
        joined = ", ".join(steps)
        self.assertIn("lcm(2,3)=6", joined)
        self.assertIn("1/2=3/6", joined)
        self.assertIn("1/3=2/6", joined)
        self.assertIn("1/2+1/3=5/6", joined)

    def test_fraction_subtraction_partial(self):
        steps = fraction_subtraction_partial(Fraction(3,4), Fraction(1,6), self.ctx)
        self.assertIsInstance(steps, list)
        joined = ", ".join(steps)
        self.assertIn("lcm(4,6)=12", joined)
        self.assertIn("3/4=9/12", joined)
        self.assertIn("1/6=2/12", joined)
        self.assertIn("3/4-1/6=7/12", joined)

    def test_subtraction_with_borrow_sign_patterns(self):
        cfg = GenConfig()
        # (+) - (+)
        steps = subtraction_with_borrow(42, 17, self.ctx)
        self.assertTrue(any("42-17=25" in s for s in steps))
        self.assertTrue(any("borrow" in s for s in steps))

        # (+) - (-)
        steps = subtraction_with_borrow(42, -17, self.ctx)
        self.assertTrue(any("42-(-17)" in s or "42+17" in s for s in steps))
        self.assertTrue(any("42+17=59" in s for s in steps))

        # (-) - (+)
        steps = subtraction_with_borrow(-42, 17, self.ctx)
        self.assertTrue(any("-42-17=" in s for s in steps))
        self.assertTrue(any("-(42+17)" in s or "-59" in s for s in steps))

        # (-) - (-)
        steps = subtraction_with_borrow(-42, -17, self.ctx)
        self.assertTrue(any("-42-(-17)" in s or "-42+17" in s for s in steps))
        self.assertTrue(any("-25" in s or "-(42-17)" in s for s in steps))

    def test_scratchpad_integration_fraction(self):
        # build (1/2) + (1/3)
        n1 = OpNode('/', Leaf(1), Leaf(2))
        n2 = OpNode('/', Leaf(1), Leaf(3))
        root = OpNode('+', n1, n2)
        cfg = GenConfig(prob_scratchpad=1.0)
        s = scratchpad(root, cfg, self.ctx)
        self.assertIsInstance(s, str)
        self.assertIn('lcm(2,3)=6', s)
        self.assertIn('1/2+1/3=', s)

    def test_fraction_multiplication_partial(self):
        steps = fraction_multiplication_partial(Fraction(1, 2), Fraction(2, 3), self.ctx)
        self.assertIsInstance(steps, list)
        joined = ", ".join(steps)
        # Should show: 1*2=2, 2*3=6, and simplification to 1/3
        self.assertIn("1*2=2", joined)
        self.assertIn("2*3=6", joined)
        self.assertIn("1/2*(2/3)=(1*2)/(2*3)", joined)
        self.assertIn('2/6=1/3', joined)

    def test_fraction_division_partial(self):
        steps = fraction_division_partial(Fraction(1, 2), Fraction(1, 3), self.ctx)
        self.assertIsInstance(steps, list)
        joined = ", ".join(steps)
        # 1/2 ÷ 1/3 = 1/2 * 3/1 = 3/2
        self.assertIn("1/2÷1/3", joined)
        self.assertIn("1*3=3", joined)
        self.assertIn("2*1=2", joined)
        self.assertIn("3/2", joined)

if __name__ == '__main__':
    unittest.main()
