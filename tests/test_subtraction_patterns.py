#!/usr/bin/env python3
import sys
sys.path.insert(0, '/Users/nikq/work/mathFormer/src')

from generate_data import subtraction_with_borrow, GenerationContext

ctx = GenerationContext(endian='big')

test_cases = [
    (42, 17),      # (+) - (+)
    (42, -17),     # (+) - (-)
    (-42, 17),     # (-) - (+)
    (-42, -17),    # (-) - (-)
]

for left, right in test_cases:
    expected = left - right
    print("=" * 60)
    print(f"Test: {left} - ({right}) = {expected}")
    print("=" * 60)
    
    try:
        steps = subtraction_with_borrow(left, right, ctx)
        print(f"Steps ({len(steps)} steps):")
        for i, step in enumerate(steps, 1):
            print(f"  {i}. {step}")
        print(f"✓ Expected: {expected}")
    except Exception as e:
        print(f"✗ Error: {e}")
    print()
