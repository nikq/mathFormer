#!/usr/bin/env python3
# Test script for addition_with_carry function

import sys
sys.path.insert(0, 'src')

from src.generate_data import addition_with_carry, GenerationContext

# Test cases
test_cases = [
    (156, 237),  # 6+7=13 carry 1, 5+3+1=9, 1+2=3 → 393
    (27, 38),    # 7+8=15 carry 1, 2+3+1=6 → 65
    (99, 1),     # 9+1=10 carry 1, 9+0+1=10 carry 1 → 100
    (123, 456),  # 3+6=9, 2+5=7, 1+4=5 → 579
    (0, 0),      # Edge case: 0+0
]

ctx_big = GenerationContext(endian='big')

print("Testing addition_with_carry function:\n")
for left, right in test_cases:
    result = addition_with_carry(left, right, ctx_big)
    actual_sum = left + right
    print(f"{left}+{right}={actual_sum}")
    print(f"  Steps: {result}")
    print()
