import torch
import math
from fractions import Fraction
from src.model import AutoRegressiveTransformerModel
from src.prepare_data import build_vocab
from src.generate_data import GenConfig, stream_samples
from src.modelparam import NInp, NHead, NHid, NLayers, Dropout  # dynamic inference still uses these as fallback

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def check_correctness(expression: str, predicted_result: str) -> bool:
    """Robust correctness check supporting integer, float, and fraction forms (e.g. '2/3')."""
    if predicted_result == '':
        return False
    true_str = str(eval(expression))

    if '/' in predicted_result:
        numer = predicted_result.split('/')[0]
        denom = predicted_result.split('/')[-1]
        if numer == '' or denom == '':
            return False
        if denom == '0' or denom == '1':
            return False
        
        # 約分しきっているかチェック
        try:
            # greatest common divisor
            def gcd(a, b):
                while b:
                    a, b = b, a % b
                return a
            if gcd(int(numer), int(denom)) != 1:
                return False
        
            # 数値的比較
            pred_float = float(Fraction(predicted_result))
            true_float = float(true_str)
            return math.isclose(true_float, pred_float, rel_tol=1e-9)
        except Exception:
            return False
    else:
        try:
            # 分数を含まない場合は数値比較
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
    correct_steps = 0
    for step in steps:
        expr, pred = step.split('=')
        if check_correctness(expr, pred):
            correct_steps += 1
    return correct_steps / len(steps) if steps else 0.0


def split_scratchpad_and_result(generated_seq: str) -> tuple[str, str]:
    """Split generated sequence into scratchpad part and result part.

    Expects format like: "... [scratchpad]; result"
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
        # No semicolon found, treat entire as result
        scratchpad_part = ""
        result_part = generated_seq.strip()
    return scratchpad_part, result_part

def evaluateModel(model, expression: str, max_len=50, print_result=True, print_correct=True, vocab=None):
    if vocab is None:
        vocab = build_vocab()
    inv_vocab = {i: char for char, i in vocab.items()}

    # Use <scratchpad> if available in vocab, otherwise fallback to [
    prompt_str = f'{expression}'
    prompt = [vocab['<sos>']] + [vocab['<big>']] + [vocab[c] for c in expression] + [vocab['<scratchpad>']] # Assume big endian for eval
    
    prompt_tensor = torch.tensor(prompt, dtype=torch.long).to(device)
    with torch.no_grad():
        generated = model.generate(prompt_tensor, max_new_tokens=max_len, eos_token=vocab['<eos>'])  # (T_total,)
    decoded = [inv_vocab[int(i)] for i in generated[1:]]  # skip <sos>
    full_seq = "".join(decoded)
    scratchpad_part, result_part = split_scratchpad_and_result(full_seq)
    correct = check_correctness(expression, result_part)
    if print_result or (print_correct and correct):
        print(f" {'OK' if correct else 'NG'} : {full_seq}")
    return correct



def _infer_model_hparams(state_dict):
    """Infer ntoken, ninp, nhid, nlayers from an autoregressive checkpoint state_dict."""
    emb_key = 'tok_emb.weight'
    layer_prefix = 'block_stack.layers.'
    if emb_key not in state_dict:
        raise ValueError(f"Embedding key '{emb_key}' not found in checkpoint.")
    ntoken, ninp = state_dict[emb_key].shape
    # infer nhid from first feed-forward layer weight
    linear1_key = None
    for k in state_dict.keys():
        if k.startswith(layer_prefix) and k.endswith('linear1.weight'):
            linear1_key = k
            break
    nhid_infer = state_dict[linear1_key].shape[0] if linear1_key else NHid
    # count layers
    nlayers_infer = len({k.split('.')[2] for k in state_dict if k.startswith(layer_prefix) and k.endswith('self_attn.in_proj_weight')}) or NLayers
    return ntoken, ninp, nhid_infer, nlayers_infer

def load_model(model_path):
    raw = torch.load(model_path, map_location=device)
    # Some checkpoints may wrap state_dict inside a dict
    if 'state_dict' in raw and isinstance(raw['state_dict'], dict):
        state_dict = raw['state_dict']
    else:
        state_dict = raw
    ntoken, ninp_ckpt, nhid_ckpt, nlayers_ckpt = _infer_model_hparams(state_dict)
    model = AutoRegressiveTransformerModel(ntoken, ninp_ckpt, NHead, nhid_ckpt, nlayers_ckpt, Dropout).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[WARN] Missing keys: {missing}, Unexpected keys: {unexpected}")
    model.eval()
    return model

def evaluate(expression: str, max_len=50, model_path='mathformer.pth'):
    vocab = build_vocab()
    model = load_model(model_path)
    evaluateModel(model, expression, max_len, vocab=vocab)



import sys
import argparse

def main():
    args = argparse.ArgumentParser(description="Evaluate a mathematical expression using the trained Transformer model.")
    args.add_argument('expression', type=str, nargs='?', default="2+3", help="The mathematical expression to evaluate (default: '2+3').")
    args.add_argument('--model_path', type=str, default='mathformer.pth', help="Path to the trained model file (default: 'mathformer.pth').")
    # autoregressive flag removed (always on)
    args.add_argument('--num_tests', type=int, default=0, help='Number of random tests to run (default: 10).')
    args.add_argument('--depth', type=int, default=3, help='Max depth of generated expressions for random tests (default: 3).')
    args.add_argument('--digits', type=int, default=2, help='Max digits of numbers in generated expressions for random tests (default: 2).')
    args.add_argument('--seed', type=int, default=42, help='Random seed for generating test expressions (default: 42).')
    args = args.parse_args()

    if args.model_path:
        vocab = build_vocab()
        model = load_model(args.model_path)
        if args.num_tests > 1:
            correct_count = 0
            sampler = stream_samples(GenConfig(max_depth_cap=args.depth, min_digits=1, max_digits=args.digits, seed=args.seed))
            for _ in range(args.num_tests):
                sample = next(sampler)
                expr = sample.expr
                ok = evaluateModel(model, expr, max_len=100, print_result=True, print_correct=False, vocab=vocab)
                if ok:
                    correct_count += 1
            print(f"Total: {args.num_tests}, Correct: {correct_count}, Accuracy: {correct_count/args.num_tests:.2%}")
        else:
            evaluateModel(model, args.expression, max_len=100, vocab=vocab)
    else:
        evaluate(args.expression, max_len=100)

if __name__ == '__main__':
    main()