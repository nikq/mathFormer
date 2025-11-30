#!/usr/bin/env python3
import sys
sys.path.insert(0, '/Users/nikq/work/mathFormer/src')

from generate_data import subtraction_with_borrow, GenerationContext

ctx = GenerationContext(endian='big')

# Test case: 42 - 17
print("=" * 50)
print("Test: 42 - 17")
print("=" * 50)
steps = subtraction_with_borrow(42, 17, ctx)
print(f"Result type: {type(steps)}")
print(f"Number of steps: {len(steps) if isinstance(steps, list) else 'N/A'}")
print("\nSteps:")
if isinstance(steps, list):
    for i, step in enumerate(steps, 1):
        print(f"  {i}. {step}")
else:
    print(f"  {steps}")

# Expected result: 42 - 17 = 25
print(f"\nExpected result: 25")
print(f"Actual result: {42 - 17}")
