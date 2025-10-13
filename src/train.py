import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse

from src.generate_data import generate_expression
from src.model import TransformerModel
from src.prepare_data import MathExprDataset, build_vocab, collate_fn

# Hyperparameters are now managed by argparse

from src.modelparam import NTokens, NInp, NHead, NHid, NLayers, Dropout
from src.evaluate import evaluateModel

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    vocab = build_vocab()
    pad_value = vocab['<pad>']
    NTokens = len(vocab)

    model = TransformerModel(NTokens, NInp, NHead, NHid, NLayers, Dropout).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_value)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.practice:
        train_print = True
        train_num = 1000
        train_batch = 10
    else:
        train_print = args.train_print
        train_num = args.num_examples
        train_batch = args.batch_size

    for depth in range(1, args.max_depth + 1):
        for digits in range(args.min_digits, args.max_digits + 1):
            # if there is a checkpoint for this depth and digit, load it
            checkpoint_path = f'mathformer_depth{depth}_digits{digits}.pth'
            model_loaded = False

            if args.skip_pretrained:
                try:
                    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
                    print(f"Loaded checkpoint from {checkpoint_path}")
                    model_loaded = True
                except FileNotFoundError:
                    print(f"No checkpoint found at {checkpoint_path}, starting fresh.")

            if not model_loaded:
                # no model loaded, train from scratch
                print(f"Training with digits={digits}, depth={depth}, with_process={args.with_process}")
                dataset = MathExprDataset(vocab, num_examples=train_num, depth=depth, min_digits=1, max_digits=digits, with_process=args.with_process)
                dataloader = DataLoader(dataset, batch_size=train_batch, shuffle=True, collate_fn=lambda b: collate_fn(b, pad_value))

                difficulty = depth * digits
                epochs = int(args.epochs * difficulty / args.epoch_scale)

                for epoch in range(epochs):
                    epoch_loss = 0
                    for src, tgt in dataloader:
                        optimizer.zero_grad()

                        src = src.permute(1, 0).to(device)
                        tgt = tgt.permute(1, 0).to(device)

                        # Create masks
                        src_mask = model.generate_square_subsequent_mask(src.size(0), device)
                        tgt_mask = model.generate_square_subsequent_mask(tgt.size(0) - 1, device)
                        src_key_padding_mask = (src == pad_value).permute(1, 0)

                        output = model(src, tgt[:-1, :], src_mask, tgt_mask, src_key_padding_mask)
                        loss = criterion(output.view(-1, NTokens), tgt[1:, :].reshape(-1))
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()

                    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader)}")

                    correct_count = 0
                    total_count = 0
                    for i in range(100):
                        expr, _, result = generate_expression(max_depth=depth,min_digits=1, max_digits=digits, with_process=False)
                        correct = evaluateModel(model, expr, print_result=train_print, print_correct=False)
                        total_count += 1
                        if correct:
                            correct_count += 1
                    print(f"Evaluation accuracy over 100 samples: {correct_count}/{total_count} = {correct_count/total_count:.2%}")

                # Save checkpoint after each digit-depth combination
                torch.save(model.state_dict(), checkpoint_path)

            # Epoch終了後の中間試験
            correct_count = 0
            total_count = 0
            for i in range(100):
                expr, _, result = generate_expression(max_depth=depth,min_digits=1, max_digits=digits, with_process=False)
                correct = evaluateModel(model, expr, print_result=True)
                total_count += 1
                if correct:
                    correct_count += 1
            print(f"Evaluation accuracy over 100 samples: {correct_count}/{total_count} = {correct_count/total_count:.2%}")

            # 模擬試験
            correct_count = 0
            total_count = 0
            for i in range(100):
                # スコアは常にフルスペックの問題に対して計算する.
                expr, _, result = generate_expression(max_depth=args.max_depth,min_digits=args.min_digits, max_digits=args.max_digits, with_process=False)
                correct = evaluateModel(model, expr, print_result=True)
                total_count += 1
                if correct:
                    correct_count += 1
            print(f"Evaluation accuracy over 100 samples: {correct_count}/{total_count} = {correct_count/total_count:.2%}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the MathFormer model.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train for.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--num_examples', type=int, default=100000, help='Number of examples per difficulty.')
    parser.add_argument('--practice', action='store_true', help='Use smaller dataset for practice.')
    parser.add_argument('--skip_pretrained',default=False, action='store_true', help='Skip loading pretrained models and train from scratch.')
    parser.add_argument('--with_process', default=True, action='store_true', help='Train with intermediate steps in the target.')
    parser.add_argument('--min_digits', type=int, default=1, help='Minimum number of digits in numbers.')
    parser.add_argument('--max_digits', type=int, default=3, help='Maximum number of digits in numbers.')
    parser.add_argument('--max_depth', type=int, default=3, help='Maximum depth of expressions.')
    parser.add_argument('--train_print', default=False, action='store_true', help='Print training evaluation results.')
    parser.add_argument('--epoch_scale', type=float, default=1, help='Scaling factor for epochs based on difficulty.')
    args = parser.parse_args()
    train(args)
