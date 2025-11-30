#!/usr/bin/env python3
import sys
sys.path.insert(0, '/Users/nikq/work/mathFormer/src')

from generate_data import GenerationContext, full_scratch, Leaf, OpNode

ctx = GenerationContext(endian='big')

# Test: 1/2 + 1/3
n1 = OpNode('/', Leaf(1), Leaf(2))
n2 = OpNode('/', Leaf(1), Leaf(3))
root = OpNode('+', n1, n2)
print('='*60)
print('Test: 1/2 + 1/3')
print('='*60)
print(full_scratch(root, ctx))

# Test: -1/2 + 1/3
n1 = OpNode('/', Leaf(-1), Leaf(2))
n2 = OpNode('/', Leaf(1), Leaf(3))
root = OpNode('+', n1, n2)
print('\n' + '='*60)
print('Test: -1/2 + 1/3')
print('='*60)
print(full_scratch(root, ctx))

# Test: 3/4 - 1/6
n1 = OpNode('/', Leaf(3), Leaf(4))
n2 = OpNode('/', Leaf(1), Leaf(6))
root = OpNode('-', n1, n2)
print('\n' + '='*60)
print('Test: 3/4 - 1/6')
print('='*60)
print(full_scratch(root, ctx))
