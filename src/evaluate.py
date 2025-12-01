import torch
import math
from fractions import Fraction
from src.checkpoint_utils import load_checkpoint_payload
from src.model import AutoRegressiveTransformerModel
from src.prepare_data import build_vocab
from src.generate_data import GenConfig, stream_samples
from src.modelparam import ModelParam

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
        # No semicolon found, treat entire as result
        scratchpad_part = ""
        result_part = generated_seq.strip()
    return scratchpad_part, result_part

def evaluateModel(model, expression: str, max_len=256, print_result=True, print_correct=True, vocab=None):
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
        print(f" {'OK' if correct else 'NG'} : expr {expression} scratch {scratchpad_part} answer {result_part}")
    return correct



def _infer_model_hparams(state_dict):
    """Infer ntoken, ninp, nhid, nlayers from an autoregressive checkpoint state_dict."""
    emb_key = 'tok_emb.weight'
    if emb_key not in state_dict:
        raise ValueError(f"Embedding key '{emb_key}' not found in checkpoint.")
    ntoken, ninp = state_dict[emb_key].shape

    block_prefix = 'blocks.'
    block_ids = set()
    linear1_key = None
    for key in state_dict.keys():
        if not key.startswith(block_prefix):
            continue
        parts = key.split('.')
        if len(parts) < 3:
            continue
        block_idx = parts[1]
        if block_idx.isdigit():
            block_ids.add(block_idx)
        if linear1_key is None and parts[-2:] == ['linear1', 'weight']:
            linear1_key = key

    default_param = ModelParam('small', ntoken)
    nhid_infer = state_dict[linear1_key].shape[0] if linear1_key else default_param.NHid
    nlayers_infer = len(block_ids) or default_param.NLayers
    return ntoken, ninp, nhid_infer, nlayers_infer

def load_model(model_path, model_size='small'):
    state_dict, config = load_checkpoint_payload(model_path, map_location=device)
    ntoken, ninp_ckpt, nhid_ckpt, nlayers_ckpt = _infer_model_hparams(state_dict)
    
    # ModelParam でモデルパラメータを取得
    model_param = ModelParam(model_size, ntoken)
    nhead = model_param.NHead
    dropout = model_param.Dropout

    if config:
        ninp_ckpt = config.get('ninp', ninp_ckpt)
        nhid_ckpt = config.get('nhid', nhid_ckpt)
        nlayers_ckpt = config.get('nlayers', nlayers_ckpt)
        nhead = config.get('nhead', nhead)
        dropout = config.get('dropout', dropout)
    
    model = AutoRegressiveTransformerModel(
        ntoken,
        ninp_ckpt,
        nhead,
        nhid_ckpt,
        nlayers_ckpt,
        dropout
    ).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[WARN] Missing keys: {missing}, Unexpected keys: {unexpected}")
    model.eval()
    return model

def evaluate(expression: str, max_len=50, model_path='mathformer.pth', model_size='small'):
    vocab = build_vocab()
    model = load_model(model_path, model_size)
    evaluateModel(model, expression, max_len, vocab=vocab)



import sys
import argparse

def main():
    args = argparse.ArgumentParser(description="Evaluate a mathematical expression using the trained Transformer model.")
    args.add_argument('expression', type=str, nargs='?', default="2+3", help="The mathematical expression to evaluate (default: '2+3').")
    args.add_argument('--checkpoint', type=str, default='mathformer.pth', help="Path to the trained model file (default: 'mathformer.pth').")
    args.add_argument('--modelsize', type=str, default='small', choices=['tiny','small','medium'], help='Model size preset (default: small).')
    args.add_argument('--num_tests', type=int, default=100, help='Number of random tests to run (default: 10).')
    args.add_argument('--depth', type=int, default=3, help='Max depth of generated expressions for random tests (default: 3).')
    args.add_argument('--digits', type=int, default=2, help='Max digits of numbers in generated expressions for random tests (default: 2).')
    args.add_argument('--seed', type=int, default=42, help='Random seed for generating test expressions (default: 42).')
    args = args.parse_args()

    if args.checkpoint:
        vocab = build_vocab()
        model = load_model(args.checkpoint, args.modelsize)
        if args.num_tests > 1:
            correct_count = 0
            sampler = stream_samples(GenConfig(max_depth_cap=args.depth, min_digits=1, max_digits=args.digits, seed=args.seed))
            for _ in range(args.num_tests):
                sample = next(sampler)
                ok = evaluateModel(model, sample.exprBigEndian, max_len=256, print_result=True, print_correct=False, vocab=vocab)
                if ok:
                    correct_count += 1
            print(f"Total: {args.num_tests}, Correct: {correct_count}, Accuracy: {correct_count/args.num_tests:.2%}")
        else:
            evaluateModel(model, args.expression, max_len=256, vocab=vocab)
    else:
        evaluate(args.expression, max_len=256, model_size=args.modelsize)

if __name__ == '__main__':
    main()