import torch
import math
from src.model import TransformerModel
from src.prepare_data import build_vocab

from src.modelparam import NTokens, NInp, NHead, NHid, NLayers, Dropout

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def check_correctness(expression: str, predicted_result: str) -> bool:
    try:
        true_result = str(eval(expression))
        # predicted_result_float = eval(predicted_result)

        # # Handle potential floating point inaccuracies
        # if isinstance(true_result, float):
        #     # Use math.isclose for float comparison
        #     return math.isclose(true_result, predicted_result_float, rel_tol=1e-9, abs_tol=0.0)
        # elif isinstance(true_result, int):
        #     # For integers, direct comparison or float comparison if predicted is float
        #     return true_result == predicted_result_float
        # else:
        #     # Fallback for other types, e.g., fractions if eval returns them
        #     # Convert both to string for comparison if direct numeric comparison is not suitable
        return str(true_result) == predicted_result
    except Exception as e:
        # 正しくない数式表記は全て不正解とみなす.
        # print(f"Error evaluating expression or predicted result: {e}")
        return False


def evaluateModel(model, expression: str, max_len=50, print_result=True, print_correct=True):
    vocab = build_vocab()
    pad_value = vocab['<pad>']
    NTokens = len(vocab)
    inv_vocab = {i: char for char, i in vocab.items()}

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

    process = "".join([inv_vocab[i] for i in tgt[1:-1]])
    
    # extract final term if there are multiple '=' in the result
    if '=' in process:
        result = process.split('=')[-1].strip()
    else:
        result = process

    correct = check_correctness(expression, result)
    if print_result or (print_correct and correct):
        print(f" {"OK" if correct else "NG"} : {expression} = {process} = {result}")
    return correct



def evaluate(expression: str, max_len=50):
    model = TransformerModel(NTokens, NInp, NHead, NHid, NLayers, Dropout).to(device)
    model.load_state_dict(torch.load('mathformer.pth', map_location=device))
    model.eval()
    evaluateModel(model, expression, max_len)



import sys
import argparse

def main():
    args = argparse.ArgumentParser(description="Evaluate a mathematical expression using the trained Transformer model.")
    args.add_argument('expression', type=str, nargs='?', default="2+3", help="The mathematical expression to evaluate (default: '2+3').")
    args.add_argument('--model_path', type=str, default='mathformer.pth', help="Path to the trained model file (default: 'mathformer.pth').")
    args = args.parse_args()

    if args.model_path:
        model = TransformerModel(NTokens, NInp, NHead, NHid, NLayers, Dropout).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        evaluateModel(model, args.expression, max_len=100)
    else:
        evaluate(args.expression, max_len=100)

if __name__ == '__main__':
    main()