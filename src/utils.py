import torch
import math
import os
import csv
from fractions import Fraction
from src.math_ast import evaluate, parse


try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, *a, **k):
        return x

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def build_prompt(expression: str, vocab: dict) -> list[int]:
    """Builds the standard prompt sequence for the model."""
    # Standard format: <sos> <big> expression <scratchpad>
    # Note: Training data might use <little>, but for generation/eval we typically use <big>
    prompt = (
        [vocab['<sos>']] +
        [vocab['<big>']] +
        [vocab[c] for c in expression] +
        [vocab['<scratchpad>']]
    )
    return prompt

def check_correctness(expression: str, predicted_result: str) -> bool:
    """Robust correctness check supporting integer, float, and fraction forms (e.g. '2/3')."""
    if predicted_result == '':
        return False

    try:
        expr_tree = parse(expression)
        eval_result = evaluate(expr_tree)
        # print(f"Expression: {expression}, Expected: {eval_result}, Predicted: {predicted_result}") # Optional logging
        true_str = str(eval_result)

        if '/' in predicted_result:
            numer = predicted_result.split('/')[0]
            denom = predicted_result.split('/')[-1]
            if numer == '' or denom == '':
                return False
            if denom == '0' or denom == '1':
                return False
            
            # Check for properly reduced fraction
            def gcd(a, b):
                while b:
                    a, b = b, a % b
                return a
            if gcd(int(numer), int(denom)) != 1:
                return False
            
            # Numerical comparison
            pred_float = float(Fraction(predicted_result))
            true_float = float(true_str)
            return math.isclose(true_float, pred_float, rel_tol=1e-9)
        else:
            # Numerical comparison for non-fractions
            true_val = float(true_str)
            pred_val = float(predicted_result)
            return math.isclose(true_val, pred_val, rel_tol=1e-9)
    except Exception:
        return False

def score_correctness(expression: str, scratchpad: str, predicted_result: str) -> float:
    # calculate a correctness score between 0 and 1 based on partial correctness of scratchpad steps
    if predicted_result == '':
        return 0.0
    
    if check_correctness(expression, predicted_result):
        return 1.0
    
    # failed, check scratchpad steps
    # Evaluate scratchpad steps for partial credit
    steps = [step.strip() for step in scratchpad.split(',') if '=' in step]
    if not steps:
        return 0.0
        
    correct_steps = 0
    for step in steps:
        try:
            expr, pred = step.split('=')
            if check_correctness(expr, pred):
                correct_steps += 1
        except ValueError:
            continue
            
    return correct_steps / len(steps)

def split_scratchpad_and_result(generated_seq: str) -> tuple[str, str]:
    """Split generated sequence into scratchpad part and result part.

    Expects format like: "<sos>expr<scratchpad>scratchpad(optional)<answer>result<eos>"
    If no scratchpad, returns ("", result)
    """ 
    # Handle new format with <scratchpad> and <answer>
    if '<answer>' in generated_seq:
        parts = generated_seq.split('<answer>')
        before_answer = parts[0]
        result_part = parts[1].split('<eos>')[0].strip()
        
        if '<scratchpad>' in before_answer:
            scratchpad_part = before_answer.split('<scratchpad>')[1].strip()
        else:
            scratchpad_part = ""
        return scratchpad_part, result_part

    else:
        # No answer tag found, treat entire as result (fallback behavior)
        scratchpad_part = ""
        result_part = generated_seq.strip()
    return scratchpad_part, result_part

def write_csv_log(path, row_dict, header=None):
    """Write a dictionary to a CSV file. If header is not provided, infers from keys (only if new file)."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    file_exists = os.path.isfile(path)
    
    if header is None:
        header = list(row_dict.keys())

    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        
        # Filter row_dict to match header keys to avoid errors if extra keys present
        row_to_write = {k: row_dict.get(k, '') for k in header}
        writer.writerow(row_to_write)
