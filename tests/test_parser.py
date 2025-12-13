# test_parse.py
from src.math_ast import parse, Node
import unittest

class TestNewOps(unittest.TestCase):
    def test_parse(self):
        node = parse("1 + 2")
        self.assertEqual(node.left.value, 1)
        self.assertEqual(node.right.value, 2)
        self.assertEqual(node.op, '+')

    def test_unary(self):
        node = parse("next(1)")
        self.assertEqual(node.child.value, 1)
        self.assertEqual(node.op, 'next')

    def test_binary(self):
        node = parse("max(1, 2)")
        self.assertEqual(node.left.value, 1)
        self.assertEqual(node.right.value, 2)
        self.assertEqual(node.op, 'max')

    def test_depth2(self):
        node = parse("max(1, next(2))")
        self.assertEqual(node.left.value, 1)
        self.assertEqual(node.right.child.value, 2)
        self.assertEqual(node.op, 'max')

    def test_depth3(self):
        node = parse("-1--2")
        self.assertEqual(node.left.child.value, 1)
        self.assertEqual(node.right.child.value, 2)
        self.assertEqual(node.op, '-')

if __name__ == '__main__':
    unittest.main()
