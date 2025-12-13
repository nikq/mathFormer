
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# generate_data.py

import random
import itertools
import math
from fractions import Fraction
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Iterator, Set
from unittest import case
from src.math_ast import Node, Leaf, OpNode, UnaryOpNode, evaluate, evaluate_binary, ARITY, parse


# ====== コンフィグ ======
@dataclass
class GenConfig:
    max_depth_cap: int = 8
    min_digits: int = 1
    max_digits: int = 2
    operators: Set[str] = field(default_factory=lambda: {'+', '-', '*', '/', '%', 'max', 'min', 'next', 'prev', 'abs'})  # 出現する演算子
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

    def r_space(self, probability=0.25):
        return ' ' if random.random() < probability else ''

    # returns random spaced operator
    def r_op(self, operator,probability=0.25):
        return f'{self.r_space(probability)}{operator}{self.r_space(probability)}'



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
    context: GenerationContext  # 生成時のコンテキスト
    meta: dict               # 追加のメタデータ（depth, digitsなど）


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
    arity = ARITY.get(op, 2)
    
    if arity == 1:
        child = generate_tree(cfg, current_depth + 1)
        return UnaryOpNode(op, child)

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
    elif op == '%':
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
    
    if isinstance(node, UnaryOpNode):
        child_str = to_string(node.child, ctx, 100, strip_paren=True) # 100 is high prec
        return f"{node.op}{ctx.r_space()}({ctx.r_space()}{child_str}{ctx.r_space()})"

    if node.op in ('max', 'min'):
        left_str = to_string(node.left, ctx, -1, strip_paren=True)
        right_str = to_string(node.right, ctx, -1, strip_paren=True)
        return f"{node.op}{ctx.r_space()}({ctx.r_space()}{left_str}{ctx.r_space()},{ctx.r_space()}{right_str}{ctx.r_space()})"
        
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '%': 2}
    prec = precedence.get(node.op, 0)
    left_str = to_string(node.left, ctx, prec, is_right=False, strip_paren=strip_paren)
    right_str = to_string(node.right, ctx, prec, is_right=True, strip_paren=strip_paren)
    s = f"{left_str}{ctx.r_op(node.op)}{right_str}"
    need_paren = prec < parent_prec or (parent_prec == prec and is_right and node.op in ('-', '/', '%'))
    if not need_paren and not strip_paren and random.random() < 0.15:
        need_paren = True
    return f"({ctx.r_op(s)})" if need_paren else ctx.r_op(s)




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
    
    l_str = ctx.f(left_val)
    r_str = ctx.f(right_val)
    l_str_minus = ctx.f(-left_val)
    r_str_minus = ctx.f(-right_val)
    result_str = ctx.f(left_val - right_val)
    
    steps = []
    # 負の数の引き算は足し算に変換
    if right_val < 0:
        # -3 - -4 = -3 + 4
        steps.append(f"{l_str}-({r_str})={l_str}+{r_str_minus}")
        add_steps = addition_with_carry(left_val, -right_val, ctx)
        if add_steps:
            steps.extend(add_steps)
        return steps

    # 左が負の数の場合
    if left_val < 0:
        # -3 - 4 = -(3+4) = -7
        steps.append(f"{l_str}-{r_str}=-({l_str_minus}+{r_str_minus})")
        addition_partial = addition_with_carry(-left_val, right_val, ctx)
        if addition_partial:
            steps.extend(addition_partial)
        steps.append(f"-({l_str_minus}+{r_str_minus})={result_str}")
        return steps

    if right_val > left_val:
        # 58 - 430 = -(430-58) = -372
        steps.append(f"{l_str}-{r_str}=-({r_str_minus}-{l_str_minus})")
        sub_partial = subtraction_with_borrow(right_val, left_val, ctx)
        if sub_partial:
            steps.extend(sub_partial)
        steps.append(f"-({r_str_minus}-{l_str_minus})={result_str}")
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
    
    for i in range(max_len):
        digit_diff = left_digits[i] - right_digits[i]
        
        if digit_diff < 0:
            steps.append(f"borrow 1")
            digit_diff += 10
            left_digits[i] += 10
            left_digits[i+1] -= 1

        step = f"{left_digits[i]}"
        terms = 1
        if right_digits[i] > 0:
            step += f"-{right_digits[i]}"
            terms += 1
        step += f"={ctx.f(digit_diff)}"
        if terms > 1:
            steps.append(step)

    if max_len > 1:
        steps.append(f"{l_str}-{r_str}={result_str}")
    return steps
    
    
# ====== 繰り上がり計算付き足し算 ======
def addition_with_carry(left_val: int, right_val: int, ctx: GenerationContext) -> str:
    """足し算を桁ごとに分解して繰り上がりを明示的に示す。
    
    例: 156+237 → "6+7=13 carry 1, 5+3+1=9, 1+2=3"
    """

    l_str = ctx.f(left_val)
    r_str = ctx.f(right_val)
    l_str_minus = ctx.f(-left_val)
    r_str_minus = ctx.f(-right_val)
    result_str = ctx.f(left_val + right_val)

    # 0を含む場合は特に部分式は不要
    if left_val == 0 or right_val == 0:
        return None

    # 負の数を部分的に含む場合は引き算で返す
    if left_val < 0 and right_val>0:
        step = [f"{l_str}+{r_str}={r_str}-{l_str_minus}"]
        sub_steps = subtraction_with_borrow(right_val, -left_val,ctx)
        step.extend(sub_steps)
        return step
    if right_val < 0:
        step = [f"{l_str}+{r_str}={l_str}-{r_str_minus}"]
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
            step += f"{ctx.f(carry)}"
            left_terms += 1
        step += f"={ctx.f(digit_sum)}"
        
        if left_terms > 1:
            steps.append(step)
        carry = new_carry
    
    # 最後の繰り上がりがあれば追加
    if max_len > 1:
        steps.append(f"{l_str}+{r_str}={result_str}")
    
    return steps

# 割り算の部分式
def division_partial(left_val: int, right_val: int, ctx: GenerationContext) -> str:
    
    abs_r = right_val if right_val > 0 else -right_val
    abs_l = left_val if left_val > 0 else -left_val

    l_str = ctx.f(left_val)
    r_str = ctx.f(right_val)
    abs_l_str = ctx.f(abs_l)
    abs_r_str = ctx.f(abs_r)


    steps = []
    if left_val < 0 and right_val > 0:
        steps.append(f"{l_str}/{r_str}=-({abs_l_str}/{abs_r_str})")
    if left_val > 0 and right_val < 0:
        steps.append(f"{l_str}/{r_str}=-({abs_l_str}/{abs_r_str})")
    if left_val < 0 and right_val < 0:
        steps.append(f"{l_str}/{r_str}={abs_l_str}/{abs_r_str}")
    # 最大公約数を求める
    gcd = math.gcd(abs_l, abs_r)
    gcd_str = ctx.f(gcd)
    steps.append(f"gcd({abs_l_str},{abs_r_str})={gcd_str}")
    if gcd == 1:
        return steps
    # 最大公約数で割る
    steps.append(f"{abs_l_str}/{gcd_str}={ctx.f(abs_l//gcd)}")
    steps.append(f"{abs_r_str}/{gcd_str}={ctx.f(abs_r//gcd)}")
    steps.append(f"{abs_l_str}/{abs_r_str}={ctx.f(abs_l//gcd)}/{ctx.f(abs_r//gcd)}")
    return steps


# ====== 分数の通分（部分式） ======
def _lcm(a: int, b: int) -> int:
    return abs(a * b) // math.gcd(a, b) if a and b else 0


def fraction_addition_partial(left_val: Fraction, right_val: Fraction, ctx: GenerationContext) -> List[str]:
    # a/b + c/d -> 通分して計算、最後に約分
    steps: List[str] = []
    a, b = left_val.numerator, left_val.denominator
    c, d = right_val.numerator, right_val.denominator

    a_str = ctx.f(a)
    b_str = ctx.f(b)
    c_str = ctx.f(c)
    d_str = ctx.f(d)

    l = _lcm(b, d)
    l_str = ctx.f(l)
    if l == 0:
        return [f"Cannot compute LCM for denominators {b} and {d}"]

    a_scaled = a * (l // b)
    c_scaled = c * (l // d)
    steps.append(f"lcm({b_str},{d_str})={l_str}")
    steps.append(f"{a_str}/{b_str}={ctx.f(a_scaled)}/{l_str}")
    steps.append(f"{c_str}/{d_str}={ctx.f(c_scaled)}/{l_str}")
    steps.append(f"{ctx.f(a_scaled)}+{ctx.f(c_scaled)}={ctx.f(a_scaled + c_scaled)}")
    sum_num = a_scaled + c_scaled
    g = math.gcd(sum_num, l)
    if g != 1:
        raw = Fraction(sum_num, l)
        steps.append(f"{sum_num}/{l}={ctx.f(raw)}")
    return steps


def fraction_subtraction_partial(left_val: Fraction, right_val: Fraction, ctx: GenerationContext) -> List[str]:
    # a/b - c/d -> 通分して計算、最後に約分
    steps: List[str] = []
    a, b = left_val.numerator, left_val.denominator
    c, d = right_val.numerator, right_val.denominator

    a_str = ctx.f(a)
    b_str = ctx.f(b)
    c_str = ctx.f(c)
    d_str = ctx.f(d)

    l = _lcm(b, d)
    l_str = ctx.f(l)
    if l == 0:
        return [f"Cannot compute LCM for denominators {b} and {d}"]

    a_scaled = a * (l // b)
    c_scaled = c * (l // d)
    steps.append(f"lcm({b_str},{d_str})={l_str}")
    steps.append(f"{a_str}/{b_str}={ctx.f(a_scaled)}/{l_str}")
    steps.append(f"{c_str}/{d_str}={ctx.f(c_scaled)}/{l_str}")
    steps.append(f"{ctx.f(a_scaled)}-{ctx.f(c_scaled)}={ctx.f(a_scaled - c_scaled)}")
    diff_num = a_scaled - c_scaled
    g = math.gcd(abs(diff_num), l)
    if g != 1:
        raw = Fraction(diff_num, l)
        steps.append(f"{diff_num}/{l}={ctx.f(raw)}")
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

    l_str = ctx.f(left_val)
    r_str = ctx.f(right_val)
    abs_l_str = ctx.f(abs_l)
    abs_r_str = ctx.f(abs_r)

    # 負の扱い
    if left_val < 0 and right_val > 0:
        steps.append(f"{l_str}*{r_str}=-({abs_l_str}*{abs_r_str})")
    if left_val > 0 and right_val < 0:
        steps.append(f"{l_str}*{r_str}=-({abs_l_str}*{abs_r_str})")
    if left_val < 0 and right_val < 0:
        steps.append(f"{l_str}*{r_str}={abs_l_str}*{abs_r_str}")
    intermediate = []
    for i in range(len(l_digits)):
        step = f"{ctx.f(l_digits[i])}*{abs_r_str}={ctx.f(l_digits[i]*abs_r)}"
        intermediate.append(l_digits[i]*abs_r*10**i)
        steps.append(step)

    result = intermediate[0]
    for i in range(1, len(intermediate)):
        step = f"{ctx.f(result)}+{ctx.f(intermediate[i])}={ctx.f(result+intermediate[i])}"
        result += intermediate[i]
        steps.append(step)
    return steps 

def fraction_multiplication_partial(left_val: Fraction, right_val: Fraction, ctx: GenerationContext) -> str:
    # a/b * c/d -> 通分して計算、最後に約分
    steps: List[str] = []
    a, b = left_val.numerator, left_val.denominator
    c, d = right_val.numerator, right_val.denominator
    prod_num = a * c
    prod_den = b * d

    a_str = ctx.f(a)
    b_str = ctx.f(b)
    c_str = ctx.f(c)
    d_str = ctx.f(d)

    steps.append(f"{a_str}/{b_str}*{c_str}/{d_str}=({ctx.f(a)}*{ctx.f(c)})/({ctx.f(b)}*{ctx.f(d)})")
    steps.append(f"{ctx.f(left_val)}*{ctx.f(right_val)}={ctx.f(prod_num)}/{ctx.f(prod_den)}")
    g = math.gcd(prod_num, prod_den)
    if g != 1:
        raw = Fraction(prod_num, prod_den)
        steps.append(f"{prod_num}/{prod_den}={ctx.f(raw)}")
    return steps

def fraction_division_partial(left_val: Fraction, right_val: Fraction, ctx: GenerationContext) -> str:
    # a/b ÷ c/d -> a/b * d/c -> 通分して計算、最後に約分
    steps: List[str] = []
    a, b = left_val.numerator, left_val.denominator
    c, d = right_val.numerator, right_val.denominator
    div_num = a * d
    div_den = b * c

    a_str = ctx.f(a)
    b_str = ctx.f(b)
    c_str = ctx.f(c)
    d_str = ctx.f(d)

    steps.append(f"{a_str}/{b_str}/({c_str}/{d_str})=({a_str}*{d_str})/({b_str}*{c_str})")
    gcd = math.gcd( a*d , b*c)
    if gcd != 1:
        steps.append(f"gcd({ctx.f(a*d)},{ctx.f(b*c)})={ctx.f(gcd)}")
        steps.append(f"{ctx.f(a*d)}/{ctx.f(gcd)}={ctx.f((a*d)//gcd)}")
        steps.append(f"{ctx.f(b*c)}/{ctx.f(gcd)}={ctx.f((b*c)//gcd)}")
        steps.append(f"({a_str}*{d_str})/({b_str}*{c_str})={ctx.f((a*d)//gcd)}/{ctx.f((b*c)//gcd)}")
    else:
        steps.append(f"({a_str}*{d_str})/({b_str}*{c_str})={ctx.f(div_num)}/{ctx.f(div_den)}")
    g = math.gcd(div_num, div_den)
    if g != 1:
        raw = Fraction(div_num, div_den)
        steps.append(f"{div_num}/{div_den}={ctx.f(raw)}")
    return steps



# スクラッチパッドの生成
def scratchpad(node: Node, cfg: GenConfig, ctx: GenerationContext) -> str:
    # 一番深いノードから定数にしていく
    steps: List[str] = []

    def helper(n: Node) -> Fraction:
        if random.random() > cfg.prob_scratchpad:
            return evaluate(n)

        # Unary Operators
        if isinstance(n, UnaryOpNode):
            child_val = helper(n.child)
            result = None
            if n.op == 'next':
                result = child_val + 1
                steps.append(f"next({ctx.f(child_val)})={ctx.f(child_val)}+1={ctx.f(result)}")
            elif n.op == 'prev':
                result = child_val - 1
                steps.append(f"prev({ctx.f(child_val)})={ctx.f(child_val)}-1={ctx.f(result)}")
            elif n.op == 'abs':
                result = abs(child_val)
                steps.append(f"abs({ctx.f(child_val)})={ctx.f(result)}")
            elif n.op == '-':
                result = - child_val
                # steps.append(f'-({ctx.f(child_val)})={ctx.f(result)}')
            else:
                raise ValueError(f"Unknown unary op: {n.op}")
            return result

        if isinstance(n, Leaf):
            return evaluate(n)
            
        # Binary Operators
        if isinstance(n, OpNode):
            # Special case for basic arithmetic with leaves: expand substeps
            if n.op in ('+', '-', '*', '/') and isinstance(n.left, Leaf) and isinstance(n.right, Leaf):
                left_val = n.left.value
                right_val = n.right.value
                substeps = None
                match n.op:
                    case '+':
                        substeps = addition_with_carry(left_val, right_val, ctx)
                    case '-':  
                        substeps = subtraction_with_borrow(left_val, right_val, ctx)
                    case '*':
                        substeps = multiplication_partial(left_val, right_val, ctx)
                    case '/':
                        substeps = division_partial(left_val, right_val, ctx)
                if substeps is not None:
                    steps.extend(substeps)
                return evaluate(n)

            # General recursive case
            left_val = helper(n.left)
            right_val = helper(n.right)
            result = None

            if n.op == 'max':
                result = max(left_val, right_val)
                steps.append(f"max({ctx.f(left_val)}, {ctx.f(right_val)})={ctx.f(result)}")
                return result
            elif n.op == 'min':
                result = min(left_val, right_val)
                steps.append(f"min({ctx.f(left_val)}, {ctx.f(right_val)})={ctx.f(result)}")
                return result
            elif n.op == '%':
                if right_val == 0:
                    raise ZeroDivisionError
                result = left_val % right_val
                steps.append(f"{ctx.f(left_val)}%{ctx.f(right_val)}={ctx.f(result)}")
                return result

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
            
            # Fraction steps or basic step
            if isinstance(left_val, Fraction) or isinstance(right_val, Fraction):
                lv = left_val if isinstance(left_val, Fraction) else Fraction(left_val)
                rv = right_val if isinstance(right_val, Fraction) else Fraction(right_val)
                if left_val.denominator != 1 and right_val.denominator != 1:
                    frac_steps = None
                    if n.op == '+':
                        frac_steps = fraction_addition_partial(lv, rv, ctx)
                    elif n.op == '-':
                        frac_steps = fraction_subtraction_partial(lv, rv, ctx)
                    elif n.op == '*':
                        frac_steps = fraction_multiplication_partial(lv, rv, ctx)
                    elif n.op == '/':
                        frac_steps = fraction_division_partial(lv, rv, ctx)
                    
                    if frac_steps is not None:
                        steps.extend(frac_steps)
                
                step_str = f"{ctx.f(left_val)}{n.op}({ctx.f(right_val)})={ctx.f(result)}"
                steps.append(step_str)
            else:
                step_str = f"{ctx.f(left_val)}{n.op}{ctx.f(right_val)}={ctx.f(result)}"
                steps.append(step_str)
            return result
            
        raise ValueError(f"Unknown node type: {type(n)}")

    helper(node)
    return ", ".join(steps)
            
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
    scratch = scratchpad(tree, cfg, ctx)
    
    model_input = expr

    return GeneratorResult(
        input=model_input,
        scratch=scratch,
        expr=expr,
        exprBigEndian=exprBigEndian,
        result=str(result),
        context=ctx,
        meta={
            "depth": depth,
            "digits": (min_d, max_d),
            "endian": endian,
        },
    )


# ====== ストリーミング（重複排除つき） ======
def stream_samples(cfg: GenConfig) -> Iterator[GeneratorResult]:
    if cfg.seed is not None:
        random.seed(cfg.seed)

    while True:
        s = generate_sample(cfg)
        len_expr = len(s.expr)
        len_scratch = len(s.scratch) if s.scratch else 0
        if len_expr + len_scratch > 2000:
            continue
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
    def parse_operators(op_str: str) -> set[str]:
        return set(op_str.split(','))
    parser.add_argument('--operators', type=parse_operators, default='+,-,*,/,min,max,next,prev,abs',
                        help='Comma-separated list of operators to use (e.g., "+,-,*,/,min,max").')
    parser.add_argument('--expr', type=str, default=None, help='If given, only perform step decomposition on this expression.')
    args = parser.parse_args()
    cfg = GenConfig(max_depth_cap=args.max_depth,
                    min_digits=args.min_digits,
                    max_digits=args.max_digits,
                    seed=args.seed,
                    prob_scratchpad=args.prob_scratchpad,
                    prob_little_endian=args.prob_little_endian,
                    operators=set(args.operators))
    
    # もし式が与えられた場合はステップ分解だけを実行
    if args.expr is not None:
        tree = parse(args.expr)
        print(to_string(tree, GenerationContext(endian='big')))
        ctx = GenerationContext(endian='big')
        scratch = scratchpad(tree, GenConfig(prob_scratchpad=1.0), ctx)
        print(scratch)
        return


    demo(args.num, cfg)


if __name__ == "__main__":
    main()
