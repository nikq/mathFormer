import unittest
from src.generate_data import generate_expression

class TestGenerateData(unittest.TestCase):
    def test_generate_expression(self):
        expression = generate_expression()
        self.assertIsInstance(expression, str)
        self.assertIn(expression[0], [str(i) for i in range(10)])

if __name__ == '__main__':
    unittest.main()
