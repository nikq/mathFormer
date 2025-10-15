import unittest
from src.generate_data import generate_expression

class TestGenerateData(unittest.TestCase):
    def test_generate_expression(self):
        expr, process, result = generate_expression()
        self.assertIsInstance(expr, str)
        self.assertIsInstance(result, str)
        # Expression should contain at least one digit
        self.assertRegex(expr, r"[0-9]")
        # Result should be numeric (may include '/') for fractions
        self.assertRegex(result, r"^[0-9/-]+$")

if __name__ == '__main__':
    unittest.main()
