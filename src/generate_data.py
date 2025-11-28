
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# generate_data_v3.py
#
# v3 変更点（v2ベース）
# - 部分スクラッチ（Partial Scratch）モードを追加：
#   特定の「部分式」だけを人間的に2〜3手で展開して <scratch>...</scratch><final>=...</final> を生成。
#   例) 14*3 -> <scratch>10*3=30, 30+4*3=42</scratch><final>=42</final>
#       27+38 -> <scratch>20+38=58, 58+7=65</scratch><final>=65</final>
#       42-17 -> <scratch>42-10=32, 32-7=25</scratch><final>=25</final>
# - 既存のフルスクラッチも保持し、確率で切り替え可能。
# - デノイジング、近重複排除、ロングテール深さ分布など v2 の改善はすべて継承。
#
# 依存：標準ライブラリのみ

import random
import itertools
from fractions import Fraction
from dataclasses import dataclass
from typing import Optional, List, Tuple, Iterator, Dict, Any, Set


# ====== AST ======
class Node: ...
class Leaf(Node):
    def __init__(self, value: int):
        self.value = value
class OpNode(Node):
    def __init__(self, op: str, left: Node, right: Node):
        self.op = op
        self.left = left
        self.right = right


# ====== 評価 ======
def evaluate(node: Node) -> Fraction:
    if isinstance(node, Leaf):
        return Fraction(node.value)
    left = evaluate(node.left)
    right = evaluate(node.right)
    if node.op == '+':
        return left + right
    if node.op == '-':
        return left - right
    if node.op == '*':
        return left * right
    if node.op == '/':
        if right == 0:
            raise ZeroDivisionError
        return left / right
    raise ValueError(f"Unknown operator: {node.op}")


# ====== 生成 ======
def generate_tree(max_depth=3, current_depth=0, min_digits=1, max_digits=1) -> Node:
    # 葉を生成
    if current_depth >= max_depth or (current_depth > 0 and random.random() < 0.5):
        num_digits = random.randint(min_digits, max_digits)
        lower_bound = 10**(num_digits - 1) if num_digits > 1 else 1
        upper_bound = 10**num_digits - 1
        sign = random.choice([1, -1])
        if random.random() < 0.005:
            sign = 0
        return Leaf(random.randint(lower_bound, upper_bound) * sign)

    operators = ['+', '-', '*', '/']
    op = random.choice(operators)
    left = generate_tree(max_depth, current_depth + 1, min_digits, max_digits)

    if op == '/':
        for _ in range(10):
            right = generate_tree(max_depth, current_depth + 1, min_digits, max_digits)
            try:
                val = evaluate(right)
            except ZeroDivisionError:
                continue
            if val != 0:
                break
        else:
            right = Leaf(1)
    else:
        right = generate_tree(max_depth, current_depth + 1, min_digits, max_digits)

    return OpNode(op, left, right)


# ====== 正規化 ======
def flatten(node: Node, op: str) -> List[Node]:
    if isinstance(node, OpNode) and node.op == op:
        return flatten(node.left, op) + flatten(node.right, op)
    return [node]

def format_number(val: int | Fraction, endian: str) -> str:
    if isinstance(val, Fraction):
        if val.denominator == 1:
            return format_number(val.numerator, endian)
        num = format_number(val.numerator, endian)
        den = format_number(val.denominator, endian)
        return f"{num}/{den}"
    
    s = str(val)
    if endian == 'big':
        return s
    elif endian == 'little':
        if val < 0:
            return '-' + s[1:][::-1]
        return s[::-1]
    else:
        raise ValueError(f"Unknown endian: {endian}")

def to_string(node: Node, parent_prec: int = 0, is_right: bool = False, strip_paren: bool=False, endian: str = 'big') -> str:
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
    if isinstance(node, Leaf):
        return format_number(node.value, endian)
    prec = precedence.get(node.op, 0)
    left_str = to_string(node.left, prec, is_right=False, strip_paren=strip_paren, endian=endian)
    right_str = to_string(node.right, prec, is_right=True, strip_paren=strip_paren, endian=endian)
    s = f"{left_str}{node.op}{right_str}"
    need_paren = prec < parent_prec or (parent_prec == prec and is_right and node.op in ('-', '/'))
    if not need_paren and not strip_paren and random.random() < 0.15:
        need_paren = True
    return f"({s})" if need_paren else s

def canonicalize(node: Node) -> Node:
    if isinstance(node, Leaf):
        return Leaf(node.value)
    left = canonicalize(node.left)
    right = canonicalize(node.right)
    if isinstance(node, OpNode) and node.op == '-':
        right = OpNode('*', Leaf(-1), right)
        node = OpNode('+', left, right)
        left, right = node.left, node.right
    if isinstance(node, OpNode) and node.op in ('+', '*'):
        items = list(itertools.chain.from_iterable(flatten(x, node.op) for x in (left, right)))
        items = [canonicalize(it) for it in items]
        items_sorted = sorted(items, key=lambda n: to_string(n, strip_paren=True))
        current = items_sorted[0]
        for it in items_sorted[1:]:
            current = OpNode(node.op, current, it)
        return current
    if isinstance(node, OpNode) and node.op == '/':
        return OpNode('/', left, right)
    return OpNode(node.op, left, right)

def canonical_hash(node: Node) -> int:
    norm = canonicalize(node)
    s = to_string(norm, strip_paren=True, endian='big')
    h = 0xcbf29ce484222325
    for ch in s:
        h ^= ord(ch)
        h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


# ====== フルスクラッチ（最下層から簡約） ======
def full_scratch(node: Node, endian: str = 'big') -> str:
    # 一番深いノードから定数にしていく
    steps: List[str] = []
    # print('Generating full scratchpad:' + to_string(node))

    def helper(n: Node) -> Fraction:
        if isinstance(n, Leaf):
            return evaluate(n)
        left_val = helper(n.left)
        right_val = helper(n.right)
        result = None
        if n.op == '+':
            result = left_val + right_val
        elif n.op == '-':
            result = left_val - right_val
        elif n.op == '*':
            result = left_val * right_val
        elif n.op == '/':
            if right_val == 0:
                raise ZeroDivisionError
            result = left_val / right_val
        
        s_left = format_number(left_val, endian)
        s_right = format_number(right_val, endian)
        s_res = format_number(result, endian)
        step_str = f"{s_left}{n.op}{s_right}={s_res}"
        # print(f' . {step_str}')
        steps.append(step_str)
        return result
    helper(node)
    return ", ".join(steps)




# ====== サンプリング分布 ======
def sample_depth(max_depth_cap: int, alpha: float=1.5) -> int:
    r = random.random()
    base = random.randint(1, max_depth_cap)
    if r < 0.2 and base < max_depth_cap:
        base = min(max_depth_cap, base + random.randint(1, max_depth_cap - base))
    return base

def sample_digits(min_digits: int, max_digits: int, p_tail: float=0.15) -> Tuple[int,int]:
    if random.random() < p_tail:
        boost = random.randint(1, 3)
        return (min_digits, max_digits + boost)
    return (min_digits, max_digits)


# ====== コンフィグ ======
@dataclass
class GenConfig:
    max_depth_cap: int = 8
    min_digits: int = 1
    max_digits: int = 2
    prob_scratchpad_full: float = 0.5       # フルスクラッチの確率
    prob_little_endian: float = 0.5         # リトルエンディアンの確率
    verifier_ratio: float = 0.05
    dedup_window: int = 100000
    seed: Optional[int] = None


# ====== サンプル生成 ======
def generate_sample(cfg: GenConfig) -> Dict[str, Any]:
    depth = sample_depth(cfg.max_depth_cap)
    min_d, max_d = sample_digits(cfg.min_digits, cfg.max_digits)
    prob_full = cfg.prob_scratchpad_full
    
    endian = 'little' if random.random() < cfg.prob_little_endian else 'big'

    while True:
        tree = generate_tree(max_depth=depth, min_digits=min_d, max_digits=max_d)
        try:
            result = evaluate(tree)
            break
        except ZeroDivisionError:
            continue

    expr = to_string(tree, endian=endian)
    norm_hash = canonical_hash(tree)

    r = random.random()
    use_full = r < prob_full
    # print(f'Generated sample with depth={depth}, digits=({min_d},{max_d}), r{r} {prob_full} use_full={use_full}')

    scratch = ""
    if use_full:
        scratch = full_scratch(tree, endian=endian)
        target = format_number(result, endian)
    else:
        target = format_number(result, endian)

    model_input = expr

    return {
        "input": model_input,
        "scratch" : scratch,
        "target": target,
        "expr": expr,
        "result": str(result),
        "hash": norm_hash,
        "endian": endian,
        "meta": {
            "depth": depth,
            "digits": (min_d, max_d),
            "scratchpad": ("full" if use_full else "none"),
            "endian": endian,
        },
    }


# ====== ストリーミング（重複排除つき） ======
def stream_samples(cfg: GenConfig) -> Iterator[Dict[str, Any]]:
    if cfg.seed is not None:
        random.seed(cfg.seed)

    recent: List[int] = []
    recent_set: Set[int] = set()

    while True:
        s = generate_sample(cfg)
        h = s["hash"]
        if h in recent_set:
            continue  # 近重複スキップ
        len_expr = len(s['expr'])
        len_scratch = len(s['scratch']) if s['scratch'] else 0
        len_target = len(s['target'])
        if len_expr + len_scratch + len_target > 2000:
            continue
        # print(f'len check: {len(s["expr"])} {len(s["scratch"]) if s["scratch"] else 0} {len(s["target"])}')
        # print(f'Generated sample: expr="{s["expr"]}", scratch="{s["scratch"]}", target="{s["target"]}"')
        recent.append(h)
        recent_set.add(h)
        if len(recent) > cfg.dedup_window:
            old = recent.pop(0)
            recent_set.discard(old)
        yield s


# ====== 使い方デモ ======
def demo(n: int = 1, cfg: GenConfig | None = None) -> None:
    if cfg is None:
        cfg = GenConfig(max_depth_cap=4, min_digits=1, max_digits=3, seed=42, prob_scratchpad_full=1.0)
    gen = stream_samples(cfg)
    for i in range(n):
        s = next(gen)
        print(f'{i}\t{s["expr"]}{" [" + s["scratch"] + "]" if s["scratch"] else ""}; {s["result"]}')


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Emit a finite number of generated math samples.")
    parser.add_argument('--num', type=int, default=1, help='Number of samples to output.')
    parser.add_argument('--max-depth', type=int, default=4)
    parser.add_argument('--min-digits', type=int, default=1)
    parser.add_argument('--max-digits', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--prob-full', type=float, default=0.5)
    parser.add_argument('--prob-little-endian', type=float, default=0.5)
    args = parser.parse_args()
    cfg = GenConfig(max_depth_cap=args.max_depth, min_digits=args.min_digits, max_digits=args.max_digits,
                    seed=args.seed, prob_scratchpad_full=args.prob_full,
                    prob_little_endian=args.prob_little_endian)
    demo(args.num, cfg)


if __name__ == "__main__":
    main()
