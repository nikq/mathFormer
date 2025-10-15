import torch
import math
from fractions import Fraction
from src.model import TransformerModel, AutoRegressiveTransformerModel
from src.prepare_data import build_vocab

from src.modelparam import NTokens, NInp, NHead, NHid, NLayers, Dropout

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


def evaluateModel(model, expression: str, max_len=50, print_result=True, print_correct=True, autoregressive=False):
    vocab = build_vocab()
    pad_value = vocab['<pad>']
    NTokens = len(vocab)
    inv_vocab = {i: char for char, i in vocab.items()}

    # output sample
    # no process: 1+5=6 -> result = 6, process = ''
    # with process: 1+5*2=1+10=11 -> result = 11, process = '=1+10'
    result = ""
    process = ""

    if autoregressive and isinstance(model, AutoRegressiveTransformerModel):
        # Build initial prompt sequence: <sos> expression '='
        prompt = [vocab['<sos>']] + [vocab[c] for c in expression] + [vocab['=']]
        prompt_tensor = torch.tensor(prompt).to(device)
        with torch.no_grad():
            generated = model.generate(prompt_tensor, max_new_tokens=max_len, eos_token=vocab['<eos>'])
        decoded = [inv_vocab[int(i)] for i in generated[1:]]  # skip <sos>, ensure tensor->int
        # Stop at eos if present
        # (Note: '<eos>' token string not in vocab keys; we used special tokens only in ids, so it won't appear here.)
        full_seq = "".join(decoded)

        # Extract result and process
        if '=' in full_seq:
            parts = full_seq.split('=')
            result = parts[-1].strip()
            process = '=' + '='.join(parts[1:-1]).strip() if len(parts) > 2 else ''
        else:
            result = full_seq
    else:
        src = [vocab['<sos>']] + [vocab[char] for char in expression] + [vocab['<eos>']]
        src = torch.tensor(src).unsqueeze(1).to(device)
        src_mask = model.generate_square_subsequent_mask(src.size(0), device)
        src_key_padding_mask = torch.zeros(src.shape[1], src.shape[0]).to(device).bool()
        tgt = [vocab['<sos>']]
        for i in range(max_len):
            tgt_tensor = torch.tensor(tgt).unsqueeze(1).to(device)
            tgt_mask = model.generate_square_subsequent_mask(tgt_tensor.size(0), device)
            with torch.no_grad():
                output = model(src, tgt_tensor, src_mask, tgt_mask, src_key_padding_mask)
            next_token = torch.argmax(output[-1, :, :], dim=-1).item()
            tgt.append(next_token)
            if next_token == vocab['<eos>']:
                break
        # extract process and result
        process = "".join([inv_vocab[i] for i in tgt[1:-1]])
        result = "".join([inv_vocab[i] for i in tgt[1:-1]]).split('=')[-1].strip()
    
    correct = check_correctness(expression, result)
    if print_result or (print_correct and correct):
        print(f" {'OK' if correct else 'NG'} : {expression}{ ' ' + process if process else ''} = {result}")
    return correct



def evaluate(expression: str, max_len=50, autoregressive=False, model_path='mathformer.pth'):
    if autoregressive:
        model = AutoRegressiveTransformerModel(NTokens, NInp, NHead, NHid, NLayers, Dropout).to(device)
    else:
        model = TransformerModel(NTokens, NInp, NHead, NHid, NLayers, Dropout).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    evaluateModel(model, expression, max_len, autoregressive=autoregressive)



import sys
import argparse

def main():
    args = argparse.ArgumentParser(description="Evaluate a mathematical expression using the trained Transformer model.")
    args.add_argument('expression', type=str, nargs='?', default="2+3", help="The mathematical expression to evaluate (default: '2+3').")
    args.add_argument('--model_path', type=str, default='mathformer.pth', help="Path to the trained model file (default: 'mathformer.pth').")
    args.add_argument('--autoregressive', action='store_true', help='Use autoregressive decoder-only model for evaluation.')
    args = args.parse_args()

    if args.model_path:
        if args.autoregressive:
            model = AutoRegressiveTransformerModel(NTokens, NInp, NHead, NHid, NLayers, Dropout).to(device)
        else:
            model = TransformerModel(NTokens, NInp, NHead, NHid, NLayers, Dropout).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        evaluateModel(model, args.expression, max_len=100, autoregressive=args.autoregressive)
    else:
        evaluate(args.expression, max_len=100, autoregressive=args.autoregressive)

if __name__ == '__main__':
    main()