import random
from fractions import Fraction


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


def evaluate(node: Node):
    if isinstance(node, Leaf):
        # return 1 if fraction values are the same

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
        # protect against division by zero
        if right == 0:
            raise ZeroDivisionError
        return left / right
    raise ValueError(f"Unknown operator: {node.op}")




def generate_tree(max_depth=3, current_depth=0, min_digits=1, max_digits=1):
    # 深さに達するか確率で葉を返す
    if current_depth >= max_depth or (current_depth > 0 and random.random() < 0.8):
        num_digits = random.randint(min_digits, max_digits)
        lower_bound = 10**(num_digits - 1) if num_digits > 1 else 0
        upper_bound = 10**num_digits - 1
        sign = random.choice([1, -1]) if lower_bound > 0 else 1
        return Leaf(random.randint(lower_bound, upper_bound) * sign)

    operators = ['+', '-', '*', '/']
    op = random.choice(operators)

    left = generate_tree(max_depth, current_depth + 1, min_digits, max_digits)

    # 右側は除算のとき0にならないように配慮して生成
    if op == '/':
        # 試行回数を制限して右側を再生成
        for _ in range(10):
            right = generate_tree(max_depth, current_depth + 1, min_digits, max_digits)
            try:
                val = evaluate(right)
            except ZeroDivisionError:
                # 右部分木の評価に除算ゼロが含まれている可能性があるので再試行
                continue
            if val != 0:
                break
        else:
            # 安全策として右を1にする
            right = Leaf(1)
    else:
        right = generate_tree(max_depth, current_depth + 1, min_digits, max_digits)

    return OpNode(op, left, right)


def to_string(node: Node, parent_prec: int = 0, is_right: bool = False) -> str:
    """ノードを文字列化。演算子の優先度と右辺での非結合性を考慮して不要な括弧を省く。
    さらにランダム性を上げるため、不要な括弧をわずかな確率で追加する。
    """
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
    if isinstance(node, Leaf):
        return str(node.value)

    prec = precedence.get(node.op, 0)
    left_str = to_string(node.left, prec, is_right=False)
    right_str = to_string(node.right, prec, is_right=True)

    s = f"{left_str}{node.op}{right_str}"

    # 親の優先度より低ければ括弧が必要
    need_paren = prec < parent_prec
    # 同じ優先度でも右辺で - や / は括弧が必要（非結合法則）
    if not need_paren and parent_prec == prec and is_right and node.op in ('-', '/'):
        need_paren = True

    # 不要な括弧をランダムに追加して多様性を上げる（15%の確率）
    if not need_paren and random.random() < 0.15:
        need_paren = True

    return f"({s})" if need_paren else s


def to_string_with_process(node: Node) -> str:
    """数式ツリーを段階的に評価し、途中式を文字列として生成する。"""
    import copy

    def find_deepest_reducible_op(n: Node, path: list):
        if isinstance(n, Leaf):
            return None
        
        if isinstance(n.left, Leaf) and isinstance(n.right, Leaf):
            return path

        if isinstance(n.left, OpNode):
            left_path = find_deepest_reducible_op(n.left, path + ['left'])
            if left_path:
                return left_path
        
        if isinstance(n.right, OpNode):
            right_path = find_deepest_reducible_op(n.right, path + ['right'])
            if right_path:
                return right_path
        
        return None

    current_tree = copy.deepcopy(node)
    steps = []

    while isinstance(current_tree, OpNode):
        path_to_op = find_deepest_reducible_op(current_tree, [])
        
        if not path_to_op:
            # Should not happen in a valid tree with ops
            break

        # Get the node to reduce
        op_parent = current_tree
        for direction in path_to_op[:-1]:
            op_parent = getattr(op_parent, direction)
        
        last_direction = path_to_op[-1] if path_to_op else None
        
        if last_direction:
            op_node = getattr(op_parent, last_direction)
        else: # root is reducible
            op_node = current_tree

        # Evaluate the operation
        result = evaluate(op_node)
        
        # Create a new leaf with the result
        new_leaf = Leaf(result)

        # Replace the op_node with the new leaf in the tree
        if not last_direction:
            current_tree = new_leaf
        else:
            setattr(op_parent, last_direction, new_leaf)
        
        steps.append(to_string(current_tree))
    
    return "=".join(steps)


def generate_expression(max_depth=10, min_digits=1, max_digits=1, with_process=False):
    # max_nodes を深さに変換する単純なロジック
    while True:
        tree = generate_tree(max_depth=max_depth, min_digits=min_digits, max_digits=max_digits)
        try:
            # 評価を試みて、エラーが出ない式ができるまで繰り返す
            result = evaluate(tree)
            expr_str = to_string(tree)
            if with_process:
                process_str = to_string_with_process(tree)
                # If the process is just the final result, no intermediate steps
                if str(result) == process_str.split("=")[-1]:
                    return expr_str, "", str(result) # no process.
                else:
                    return expr_str, process_str, str(result)
            else:
                return expr_str, "", str(result)
        except ZeroDivisionError:
            continue


if __name__ == "__main__":
    depth=2
    for i in range(10):
        expr, process, result = generate_expression(max_depth=depth,min_digits=1, max_digits=2, with_process=True)
        if process:
            print(f"{expr}={process}={result}")
        else:
            print(f"{expr}={result}")
    print("-" * 20)
    for i in range(10):
        expr, _, result = generate_expression(max_depth=depth,min_digits=1, max_digits=2, with_process=False)
        print(f"{expr}={result}")
