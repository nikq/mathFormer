import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.generate_data import generate_expression
from src.model import TransformerModel
from src.prepare_data import MathExprDataset, build_vocab, collate_fn

# Hyperparameters
N_EPOCHS = 100
LEARNING_RATE = 0.001

from src.modelparam import NTokens, NInp, NHead, NHid, NLayers, Dropout
from src.evaluate import evaluateModel

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    vocab = build_vocab()
    pad_value = vocab['<pad>']
    NTokens = len(vocab)

    model = TransformerModel(NTokens, NInp, NHead, NHid, NLayers, Dropout).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_value)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    depth = 1
    min_digits = 1
    max_digits = 1

    dataset = MathExprDataset(vocab, num_examples=1000, depth=depth, min_digits=min_digits, max_digits=max_digits)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=lambda b: collate_fn(b, pad_value))

    for epoch in range(N_EPOCHS):
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

        print(f"Epoch {epoch+1}/{N_EPOCHS}, Loss: {epoch_loss/len(dataloader)}")
        torch.save(model.state_dict(), 'mathformer.pth')

        evaluateModel(model, "2+3")
        evaluateModel(model, "5-2")
        evaluateModel(model, "10/2")

        correct_count = 0
        total_count = 0
        for i in range(100):
            expr, result = generate_expression(max_depth=depth,min_digits=min_digits, max_digits=max_digits)
            correct = evaluateModel(model, expr, print_result=False)
            total_count += 1
            if correct:
                correct_count += 1
        print(f"Evaluation accuracy over 100 samples: {correct_count}/{total_count} = {correct_count/total_count:.2%}")

if __name__ == '__main__':
    train()
