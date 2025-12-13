# AST.py

from fractions import Fraction

# ====== AST ======
class Node:
    pass

class Leaf(Node):
    def __init__(self, value: int):
        self.value = value
class OpNode(Node):
    def __init__(self, op: str, left: Node, right: Node):
        self.op = op
        self.left = left
        self.right = right

class UnaryOpNode(Node):
    def __init__(self, op: str, child: Node):
        self.op = op
        self.child = child

ARITY = {
    '+': 2, '-': 2, '*': 2, '/': 2, '%': 2,
    'max': 2, 'min': 2,
    'next': 1, 'prev': 1, 'abs': 1
}


# ====== 評価 ======

    
def evaluate(node: Node) -> Fraction:
    if isinstance(node, Leaf):
        return Fraction(node.value)
    if isinstance(node, UnaryOpNode):
        child = evaluate(node.child)
        if node.op == 'next':
            return child + 1
        if node.op == 'prev':
            return child - 1
        if node.op == 'abs':
            return abs(child)
        if node.op == '-':
            return -child
        raise ValueError(f"Unknown unary operator: {node.op}")

    left = evaluate(node.left)
    right = evaluate(node.right)
    return evaluate_binary(node, left, right)

def evaluate_binary(node: OpNode, left: Fraction, right: Fraction) -> Fraction:
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
    if node.op == '%':
        if right == 0:
            raise ZeroDivisionError
        return left % right
    if node.op == 'max':
        return max(left, right)
    if node.op == 'min':
        return min(left, right)
    raise ValueError(f"Unknown operator: {node.op}")


# parser

def parse(expression: str) -> Node:
    tokens = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        '+', '-', '*', '/', '%',
        '(', ')',
        'next', 'prev', 'abs', 'max', 'min',
    ]

    # split first token, ignore spaces
    def first_token(expression: str) -> str:
        for token in tokens:
            if expression.startswith(token):
                return token
        raise ValueError(f"Unknown token: {expression}")
    
    # remove leading spaces
    expression = expression.strip()
    token = first_token(expression)

    # This is a recursive descent parser implementation
    
    pos = 0
    
    def peek():
        nonlocal pos
        while pos < len(expression) and expression[pos].isspace():
            pos += 1
        if pos == len(expression):
            return None
        
        # Check multi-char tokens first
        remaining = expression[pos:]
        for t in ['next', 'prev', 'abs', 'max', 'min']:
            if remaining.startswith(t):
                return t
        
        # Check single char tokens or digits
        char = expression[pos]
        if char.isdigit():
            # Consume full number
            end = pos
            while end < len(expression) and expression[end].isdigit():
                end += 1
            return expression[pos:end]
        return char

    def consume(token):
        nonlocal pos
        p = peek()
        if p == token:
            pos += len(token)
            return True
        return False

    def parse_expression():
        return parse_term()

    def parse_term():
        left = parse_factor()
        while True:
            op = peek()
            if op in ('+', '-'):
                consume(op)
                right = parse_factor()
                left = OpNode(op, left, right)
            else:
                break
        return left

    def parse_factor():
        left = parse_unary()
        while True:
            op = peek()
            if op in ('*', '/', '%'):
                consume(op)
                right = parse_unary()
                left = OpNode(op, left, right)
            else:
                break
        return left
        
    def parse_unary():
        op = peek()
        if op in ('next', 'prev', 'abs', '-'): # Unary minus is treated here
            consume(op)
            child = parse_unary()
            return UnaryOpNode(op, child)
        
        return parse_primary()
        
        return parse_primary()

    def parse_primary():
        t = peek()
        if t == '(':
            consume('(')
            node = parse_expression()
            if not consume(')'):
                raise ValueError("Expected ')'")
            return node
        elif t in ('max', 'min'):
            # These are binary ops but look like function calls: max(a, b)
            # The prompt implies natural expression like (1+2)*3.
            # If max/min are used like max(1, 2), we need to handle comma.
            # Let's assume standard function syntax for these binary ops.
            op = t
            consume(op)
            if not consume('('):
                raise ValueError(f"Expected '(' after {op}")
            left = parse_expression()
            if not consume(','): # We need to handle comma separator for binary functions
                 # If the parser doesn't support comma, maybe they are infix? 1 max 2?
                 # ARITY says they are binary.
                 # Let's assume infix for simplicity if comma logic isn't requested, 
                 # OR implement comma support inside parens.
                 # Given "natural expression", max(a,b) is most likely.
                 raise ValueError("Expected ',' in function call")
            right = parse_expression()
            if not consume(')'):
                raise ValueError("Expected ')'")
            return OpNode(op, left, right)
        
        elif t and t[0].isdigit():
            consume(t)
            return Leaf(int(t))
        else:
            raise ValueError(f"Unexpected token: {t}")

    # Start parsing
    # We need to handle the comma token for max/min
    # Let's patch peek to handle comma if needed, or just treat it as a char.
    
    result = parse_expression()
    if peek() is not None:
        raise ValueError(f"Unexpected extra characters: {expression[pos:]}")
    return result

