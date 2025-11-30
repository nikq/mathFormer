
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
import math
from fractions import Fraction
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Iterator, Set
from unittest import case


# ====== コンフィグ ======
@dataclass
class GenConfig:
    max_depth_cap: int = 8
    min_digits: int = 1
    max_digits: int = 2
    operators: Set[str] = field(default_factory=lambda: {'+', '-', '*', '/'})  # 出現する演算子
    prob_scratchpad: float = 0.3      # スクラッチパッドの確率
    prob_little_endian: float = 0.5         # リトルエンディアンの確率
    dedup_window: int = 100
    seed: Optional[int] = None

@dataclass
class GenerationContext:
    """生成時のコンテキスト情報を保持するクラス。
    
    endianなどの設定を関数の引数で引き回す代わりに、
    このオブジェクトにまとめて管理する。
    """
    endian: str = 'big'

    def f(self, val: int | Fraction) -> str:
        if isinstance(val, Fraction):
            if val.denominator == 1:
                return self.f(val.numerator)
            num = self.f(val.numerator)
            den = self.f(val.denominator)
            return f"{num}/{den}"
        
        s = str(val)
        if self.endian == 'big':
            return s
        elif self.endian == 'little':
            if val < 0:
                return '-' + s[1:][::-1]
            return s[::-1]
        else:
            raise ValueError(f"Unknown endian: {self.endian}")

@dataclass
class GeneratorResult:
    """生成されたサンプルを保持するクラス。
    
    生成結果とその生成時のコンテキストを一緒に管理する。
    """
    input: str               # モデルへの入力式
    scratch: str             # スクラッチパッドのステップ（使用しない場合は空文字列）
    expr: str                # 式の文字列（inputと同じ）
    exprBigEndian: str       # 式の文字列（big-endian）
    result: str              # 数値結果の文字列表現
    hash: int                # 式ツリーの正規ハッシュ
    context: GenerationContext  # 生成時のコンテキスト
    meta: dict               # 追加のメタデータ（depth, digitsなど）



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
def generate_tree(cfg: GenConfig, current_depth = 0) -> Node:
    # 葉を生成
    if current_depth >= cfg.max_depth_cap or (current_depth > 0 and random.random() < 0.5):
        num_digits = random.randint(cfg.min_digits, cfg.max_digits)
        lower_bound = 10**(num_digits - 1) if num_digits > 1 else 1
        upper_bound = 10**num_digits - 1
        sign = random.choice([1, -1])
        if random.random() < 0.005:
            sign = 0
        return Leaf(random.randint(lower_bound, upper_bound) * sign)

    operators = list(cfg.operators)
    op = random.choice(operators)
    left = generate_tree(cfg, current_depth + 1)

    if op == '/':
        for _ in range(10):
            right = generate_tree(cfg, current_depth + 1)
            try:
                val = evaluate(right)
            except ZeroDivisionError:
                continue
            if val != 0:
                break
        else:
            right = Leaf(1)
    else:
        right = generate_tree(cfg, current_depth + 1)

    return OpNode(op, left, right)


# ====== 正規化 ======
def flatten(node: Node, op: str) -> List[Node]:
    if isinstance(node, OpNode) and node.op == op:
        return flatten(node.left, op) + flatten(node.right, op)
    return [node]


def to_string(node: Node, ctx: GenerationContext, parent_prec: int = 0, is_right: bool = False, strip_paren: bool=False) -> str:
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
    if isinstance(node, Leaf):
        return ctx.f(node.value)
    prec = precedence.get(node.op, 0)
    left_str = to_string(node.left, ctx, prec, is_right=False, strip_paren=strip_paren)
    right_str = to_string(node.right, ctx, prec, is_right=True, strip_paren=strip_paren)
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
        items_sorted = sorted(items, key=lambda n: to_string(n, GenerationContext(endian='big'), strip_paren=True))
        current = items_sorted[0]
        for it in items_sorted[1:]:
            current = OpNode(node.op, current, it)
        return current
    if isinstance(node, OpNode) and node.op == '/':
        return OpNode('/', left, right)
    return OpNode(node.op, left, right)

def canonical_hash(node: Node) -> int:
    norm = canonicalize(node)
    s = to_string(norm, GenerationContext(endian='big'), strip_paren=True)
    h = 0xcbf29ce484222325
    for ch in s:
        h ^= ord(ch)
        h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


# ====== フルスクラッチ（最下層から簡約） ======
def full_scratch(node: Node, ctx: GenerationContext) -> str:
    # 一番深いノードから定数にしていく
    steps: List[str] = []
    # print('Generating full scratchpad:' + to_string(node, ctx))

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
        
        step_str = f"{ctx.f(left_val)}{n.op}{ctx.f(right_val)}={ctx.f(result)}"
        # print(f' . {step_str}')
        steps.append(step_str)
        return result
    helper(node)
    return ", ".join(steps)


def number_to_str_digits(val: int, ctx: GenerationContext) -> List[int]:
    digits = []
    # 負の場合
    if val < 0:
        val = -val
    temp = val
    while temp > 0:
        digits.append(temp % 10)
        temp //= 10
    return digits


# ====== 繰り下がり計算付き引き算 ======
def subtraction_with_borrow(left_val: int, right_val: int, ctx: GenerationContext) -> str:
    if left_val == 0 or right_val == 0:
        return None
    
    steps = []
    # 負の数の引き算は足し算に変換
    if right_val < 0:
        # -3 - -4 = -3 + 4
        steps.append(f"{left_val}-({right_val})={left_val}+{-right_val}")
        add_steps = addition_with_carry(left_val, -right_val, ctx)
        if add_steps:
            steps.extend(add_steps)
        return steps

    # 左が負の数の場合
    if left_val < 0:
        # -3 - 4 = -(3+4) = -7
        steps.append(f"{left_val}-{right_val}=-({-left_val}+{right_val})")
        addition_partial = addition_with_carry(-left_val, right_val, ctx)
        if addition_partial:
            steps.extend(addition_partial)
        steps.append(f"-({-left_val}+{right_val})={left_val-right_val}")
        return steps

    if right_val > left_val:
        # 58 - 430 = -(430-58) = -372
        steps.append(f"{left_val}-{right_val}=-({right_val}-{left_val})")
        sub_partial = subtraction_with_borrow(right_val, left_val, ctx)
        if sub_partial:
            steps.extend(sub_partial)
        steps.append(f"-({right_val}-{left_val})={left_val-right_val}")
        return steps

    left_digits = number_to_str_digits(left_val, ctx)
    right_digits = number_to_str_digits(right_val, ctx)
    
    if not left_digits:
        left_digits = [0]
    
    if not right_digits:
        right_digits = [0]
    
    # 桁数を揃える
    max_len = max(len(left_digits), len(right_digits))
    while len(left_digits) < max_len:
        left_digits.append(0)
    while len(right_digits) < max_len:
        right_digits.append(0)
    
    # 下の桁から順に引き算（繰り下がりを追跡）
    steps: List[str] = []
    borrow = 0
    
    for i in range(max_len):
        digit_diff = left_digits[i] - right_digits[i]
        new_borrow = 0
        
        if digit_diff < 0:
            steps.append(f"borrow 1")
            digit_diff += 10
            left_digits[i] += 10
            left_digits[i+1] -= 1
            new_borrow = 1

        step = f"{left_digits[i]}"
        terms = 1
        if right_digits[i] > 0:
            step += f"-{right_digits[i]}"
            terms += 1
        step += f"={digit_diff}"
        if terms > 1:
            steps.append(step)

        borrow = new_borrow
    if max_len > 1:
        steps.append(f"{left_val}-{right_val}={left_val-right_val}")
    return steps
    
    
# ====== 繰り上がり計算付き足し算 ======
def addition_with_carry(left_val: int, right_val: int, ctx: GenerationContext) -> str:
    """足し算を桁ごとに分解して繰り上がりを明示的に示す。
    
    例: 156+237 → "6+7=13 carry 1, 5+3+1=9, 1+2=3"
    """
    # 0を含む場合は特に部分式は不要
    if left_val == 0 or right_val == 0:
        return None

    # 負の数を部分的に含む場合は引き算で返す
    if left_val < 0 and right_val>0:
        step = [f"{left_val}+{right_val}={right_val}-{-left_val}"]
        sub_steps = subtraction_with_borrow(right_val, -left_val,ctx)
        step.extend(sub_steps)
        return step
    if right_val < 0:
        step = [f"{left_val}+{right_val}={left_val}-{-right_val}"]
        sub_steps = subtraction_with_borrow(left_val, -right_val, ctx)
        step.extend(sub_steps)
        return step
    
    # 各桁を取り出す（最下位桁から）
    left_digits = number_to_str_digits(left_val, ctx)
    right_digits = number_to_str_digits(right_val, ctx)
    
    if not left_digits:
        left_digits = [0]
    
    if not right_digits:
        right_digits = [0]
    
    # 桁数を揃える
    max_len = max(len(left_digits), len(right_digits))
    while len(left_digits) < max_len:
        left_digits.append(0)
    while len(right_digits) < max_len:
        right_digits.append(0)
    
    # 下の桁から順に足し算（繰り上がりを追跡）
    steps: List[str] = []
    carry = 0
    
    for i in range(max_len):
        digit_sum = left_digits[i] + right_digits[i] + carry
        new_carry = digit_sum // 10
        
        # ステップを記録
        if carry > 0:
            steps.append(f"carry {carry}")
        
        step = ""
        left_terms = 0
        if left_digits[i] > 0:
            left_terms += 1
            step += f"{left_digits[i]}"
        if right_digits[i] > 0:
            if left_terms > 0:
                step += "+"
            step += f"{right_digits[i]}"
            left_terms += 1
        if carry > 0:
            if left_terms > 0:
                step += "+"
            step += f"{carry}"
            left_terms += 1
        step += f"={digit_sum}"
        
        if left_terms > 1:
            steps.append(step)
        carry = new_carry
    
    # 最後の繰り上がりがあれば追加
    if max_len > 1:
        steps.append(f"{left_val}+{right_val}={left_val+right_val}")
    
    return steps

# 割り算の部分式
def division_partial(left_val: int, right_val: int, ctx: GenerationContext) -> str:
    abs_r = right_val if right_val > 0 else -right_val
    abs_l = left_val if left_val > 0 else -left_val

    steps = []
    if left_val < 0 and right_val > 0:
        steps.append(f"{left_val}/{right_val}=-({abs_l}/{abs_r})")
    if left_val > 0 and right_val < 0:
        steps.append(f"{left_val}/{right_val}=-({abs_l}/{abs_r})")
    if left_val < 0 and right_val < 0:
        steps.append(f"{left_val}/{right_val}={abs_l}/{abs_r}")

    # 最大公約数を求める
    gcd = math.gcd(abs_l, abs_r)
    if gcd == 1:
        steps.append(f"{abs_l} and {abs_r} are coprime")
        return steps
    # 最大公約数で割る
    steps.append(f"{abs_l}={abs_l//gcd}*{gcd}")
    steps.append(f"{abs_r}={abs_r//gcd}*{gcd}")
    steps.append(f"{abs_l}/{abs_r}={abs_l//gcd}/{abs_r//gcd}")
    return steps

# 乗算の部分式
def multiplication_partial(left_val: int, right_val: int, ctx: GenerationContext) -> str:
    abs_r = right_val if right_val > 0 else -right_val
    abs_l = left_val if left_val > 0 else -left_val

    l_digits = number_to_str_digits(abs_l, ctx)
    r_digits = number_to_str_digits(abs_r, ctx)
    
    if not l_digits:
        l_digits = [0]
    
    if not r_digits:
        r_digits = [0]
    
    # 下の桁から順に乗算（繰り上がりを追跡）
    steps: List[str] = []
    carry = 0

    # 負の扱い
    if left_val < 0 and right_val > 0:
        steps.append(f"{left_val}*{right_val}=-({abs_l}*{abs_r})")
    if left_val > 0 and right_val < 0:
        steps.append(f"{left_val}*{right_val}=-({abs_l}*{-abs_r})")
    if left_val < 0 and right_val < 0:
        steps.append(f"{left_val}*{right_val}={abs_l}*{abs_r}")

    intermediate = []
    for i in range(len(l_digits)):
        step = f"{l_digits[i]}*{abs_r}={l_digits[i]*abs_r}"
        intermediate.append(l_digits[i]*abs_r*10**i)
        steps.append(step)

    result = intermediate[0]
    for i in range(1, len(intermediate)):
        step = f"{result}+{intermediate[i]}={result+intermediate[i]}"
        result += intermediate[i]
        steps.append(step)
    return steps 


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



# ====== サンプル生成 ======
def generate_sample(cfg: GenConfig) -> GeneratorResult:
    depth = sample_depth(cfg.max_depth_cap)
    min_d, max_d = sample_digits(cfg.min_digits, cfg.max_digits)
    
    endian = 'little' if random.random() < cfg.prob_little_endian else 'big'
    ctx = GenerationContext(endian=endian)

    while True:
        tree = generate_tree(cfg)
        try:
            result = evaluate(tree)
            break
        except ZeroDivisionError:
            continue

    expr = to_string(tree, ctx)
    exprBigEndian = to_string(tree, GenerationContext(endian='big'))
    norm_hash = canonical_hash(tree)

    # スクラッチパッドモードを選択（3種類: none, full, carry）
    r = random.random()

    scratch = ""
    scratchpad_mode = 'none'
    if random.random() < cfg.prob_scratchpad:
        # 繰り上がりモードは足し算のLeafノード同士の場合のみ適用
        if isinstance(tree, OpNode) and isinstance(tree.left, Leaf) and isinstance(tree.right, Leaf):
            left_val = tree.left.value
            right_val = tree.right.value
            match tree.op:
                case '+':
                    steps = addition_with_carry(left_val, right_val, ctx)
                case '-':  
                    steps = subtraction_with_borrow(left_val, right_val, ctx)
                case '*':
                    steps = multiplication_partial(left_val, right_val, ctx)
                case '/':
                    steps = division_partial(left_val, right_val, ctx)
                case _:
                    pass
            if steps is not None:
                scratch = ", ".join(steps)
                scratchpad_mode = f'partial_{tree.op}'
        else:
            scratch = full_scratch(tree, ctx)
            scratchpad_mode = 'full'
    else:
        scratchpad_mode = 'none'
    
    model_input = expr

    return GeneratorResult(
        input=model_input,
        scratch=scratch,
        expr=expr,
        exprBigEndian=exprBigEndian,
        result=str(result),
        hash=norm_hash,
        context=ctx,
        meta={
            "depth": depth,
            "digits": (min_d, max_d),
            "scratchpad": scratchpad_mode,
            "endian": endian,
        },
    )


# ====== ストリーミング（重複排除つき） ======
def stream_samples(cfg: GenConfig) -> Iterator[GeneratorResult]:
    if cfg.seed is not None:
        random.seed(cfg.seed)

    recent: List[int] = []
    recent_set: Set[int] = set()

    while True:
        s = generate_sample(cfg)
        h = s.hash
        if h in recent_set:
            continue  # 近重複スキップ
        len_expr = len(s.expr)
        len_scratch = len(s.scratch) if s.scratch else 0
        if len_expr + len_scratch > 2000:
            continue
        recent.append(h)
        recent_set.add(h)
        if len(recent) > cfg.dedup_window:
            old = recent.pop(0)
            recent_set.discard(old)
        yield s


# ====== 使い方デモ ======
def demo(n: int = 1, cfg: GenConfig | None = None) -> None:
    if cfg is None:
        cfg = GenConfig(max_depth_cap=4, min_digits=1, max_digits=3, seed=42, prob_scratchpad=1.0)
    gen = stream_samples(cfg)
    for i in range(n):
        s = next(gen)
        print(f'endian {s.context.endian:<10} {i:<10} {s.expr}{" [" + s.scratch + "]" if s.scratch else ""}; {s.result}')


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Emit a finite number of generated math samples.")
    parser.add_argument('--num', type=int, default=1, help='Number of samples to output.')
    parser.add_argument('--max-depth', type=int, default=4)
    parser.add_argument('--min-digits', type=int, default=1)
    parser.add_argument('--max-digits', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--prob-scratchpad', type=float, default=0.3)
    parser.add_argument('--prob-little-endian', type=float, default=0.5)
    parser.add_argument('--operators', type=str, default='+-*/')
    args = parser.parse_args()
    cfg = GenConfig(max_depth_cap=args.max_depth,
                    min_digits=args.min_digits,
                    max_digits=args.max_digits,
                    seed=args.seed,
                    prob_scratchpad=args.prob_scratchpad,
                    prob_little_endian=args.prob_little_endian,
                    operators=set(args.operators))
    demo(args.num, cfg)


if __name__ == "__main__":
    main()
