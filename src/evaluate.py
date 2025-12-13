import torch
import math
from fractions import Fraction
from src.checkpoint_utils import load_checkpoint_payload
from src.model import AutoRegressiveTransformerModel
from src.prepare_data import build_vocab
from src.generate_data import GenConfig, stream_samples
from src.modelparam import ModelParam
from src.math_ast import evaluate, evaluate_binary, ARITY, parse
from src.utils import check_correctness, score_correctness, split_scratchpad_and_result, get_device
from src.checkpoint_utils import load_checkpoint_payload, infer_model_hparams, load_model_from_checkpoint

device = get_device()




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



def load_model(model_path, model_size='small'):
    return load_model_from_checkpoint(model_path, device, model_size)


import sys
import argparse

def main():
    args = argparse.ArgumentParser(description="Evaluate a mathematical expression using the trained Transformer model.")
    args.add_argument('expression', type=str, nargs='?', default="2+3", help="The mathematical expression to evaluate (default: '2+3').")
    args.add_argument('--checkpoint', type=str, default='mathformer.pth', help="Path to the trained model file (default: 'mathformer.pth').")
    args.add_argument('--modelsize', type=str, default='small', choices=['tiny','small','medium','large'], help='Model size preset (default: small).')
    args.add_argument('--num_tests', type=int, default=100, help='Number of random tests to run (default: 10).')
    args.add_argument('--depth', type=int, default=3, help='Max depth of generated expressions for random tests (default: 3).')
    args.add_argument('--digits', type=int, default=2, help='Max digits of numbers in generated expressions for random tests (default: 2).')
    args.add_argument('--seed', type=int, default=42, help='Random seed for generating test expressions (default: 42).')
    args = args.parse_args()

    print(args.checkpoint)

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

if __name__ == '__main__':
    main()