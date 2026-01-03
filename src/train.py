import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import csv
import os
from src.utils import get_device, write_csv_log, tqdm
from src.prepare_data import build_vocab
from src.model import AutoRegressiveTransformerModel


from src.model import AutoRegressiveTransformerModel
from src.prepare_data import MathExprDataset, build_vocab, collate_fn_autoregressive
from src.checkpoint_utils import load_checkpoint_payload, build_model_param, save_checkpoint
from src.evaluate import evaluateModel
from src.generate_data import GenConfig, stream_samples
from torch.nn.utils import clip_grad_norm_
import math

def _maybe_load_checkpoint(path, device):
    if not path:
        return None, None
    try:
        state_dict, config = load_checkpoint_payload(path, map_location=device)
        print(f"Loaded checkpoint metadata from {path}")
        return state_dict, config
    except FileNotFoundError:
        print(f"No checkpoint found at {path}, starting fresh.")
        return None, None




def train(args):
    device = get_device()
    print(f"Using device: {device}")

    vocab = build_vocab()
    pad_value = vocab['<scratchpad>']
    NTokens = len(vocab)

    checkpoint_state_dict, checkpoint_config = _maybe_load_checkpoint(args.checkpoint, device)

    # ModelParam でモデルパラメータをセットアップ
    # ModelParam でモデルパラメータをセットアップ
    model_param = build_model_param(args.modelsize, NTokens, checkpoint_config)
    
    # Override MoE parameters if specified in args
    if hasattr(args, 'num_experts') and args.num_experts > 0:
        model_param.NumExperts = args.num_experts
        model_param.ActiveExperts = args.active_experts
        
    print(f"Model configuration: {model_param}")

    model = AutoRegressiveTransformerModel(
        model_param.NTokens,
        model_param.NInp,
        model_param.NHead,
        model_param.NHid,
        model_param.NLayers,
        model_param.Dropout,
        num_experts=model_param.NumExperts,
        active_experts=model_param.ActiveExperts
    ).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_value)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    depth = args.max_depth
    digits = args.max_digits
    steps = args.steps

    if args.practice:
        train_print = True
        train_print_correct = True
        train_num = 1000
        train_batch = 10
    else:
        train_print = args.train_print
        train_print_correct = args.train_print_correct
        train_num = args.num_examples
        train_batch = args.batch_size

    total_steps = steps * math.ceil(train_num / train_batch)
    warmup_steps = int(total_steps * args.warmup_ratio)
    min_lr = args.min_lr

    def lr_lambda(step_idx: int):
        if step_idx < warmup_steps and warmup_steps > 0:
            # linear warmup from 0 to 1
            return (step_idx + 1) / warmup_steps
        # cosine decay after warmup
        progress = (step_idx - warmup_steps) / max(1, (total_steps - warmup_steps))
        cosine = 0.5 * (1 + math.cos(math.pi * progress))  # in [0,1]
        # scale to min_lr/initial_lr ... 1
        min_scale = min_lr / max(args.lr, 1e-12)
        return cosine * (1 - min_scale) + min_scale

    scheduler = None
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    if checkpoint_state_dict:
        model.load_state_dict(checkpoint_state_dict)
        print(f"Loaded weights from {args.checkpoint}")

    # Wrap epoch iterator with tqdm unless disabled
    step_iter = range(steps)
    step_iter = tqdm(step_iter, desc=f"Steps", leave=True)

    for step in step_iter:

        dataset = MathExprDataset(vocab, num_examples=train_num, depth=depth, max_digits=digits, seed=step)
        dataloader = DataLoader(dataset, batch_size=train_batch, shuffle=True, collate_fn=lambda b: collate_fn_autoregressive(b, pad_value))

        checkpoint_path = f"checkpoints/model{model_param.hash()}_{model_param.type()}_step{step}.pt"

        batch_iter = tqdm(dataloader, desc=f"Train step {step}/{steps}", leave=False)

        model.train()
        step_loss = 0

        for batch_idx, batch in enumerate(batch_iter):
            optimizer.zero_grad()
            # batch: (B, T)
            seq = batch.to(device)
            logits = model(seq[:, :-1])  # (B, T-1, V)
            loss = criterion(logits.reshape(-1, NTokens), seq[:, 1:].reshape(-1))
            # NaN
            if loss.isnan().any():
                print("NaN loss encountered, skipping batch.")
                continue
            loss.backward()
            grad_norm = None
            if args.grad_clip > 0:
                grad_norm = clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            if scheduler:
                scheduler.step()
            step_loss += loss.item()
            if batch_idx % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                postfix = { 'loss': f"{loss.item():.4f}", 'lr': f"{current_lr:.2e}" }
                if grad_norm is not None:
                    postfix['gnorm'] = f"{float(grad_norm):.2f}"
                batch_iter.set_postfix(**postfix)
        
        avg_loss = step_loss/len(dataloader)
    
        correct_count = 0
        total_count = 0
        if args.step_eval:
            # model loaded or training done
            # batchごとに評価
            model.eval()
            eval_config = GenConfig(max_depth_cap=depth, max_digits=digits, seed=123)
            stream = stream_samples(eval_config)
            for i in range(100):
                # Generate a fresh sample and evaluate expression part
                sample = next(stream)
                correct = evaluateModel(model, sample.exprBigEndian, print_result=train_print, print_correct=train_print_correct)
                total_count += 1
                if correct:
                    correct_count += 1

        eval_acc = correct_count/total_count if total_count else 0.0
        current_lr = optimizer.param_groups[0]['lr']
        step_iter.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{current_lr:.2e}", acc=f"{eval_acc:.2%}")

        # CSV logging per epoch
        if args.log_csv:
            write_csv_log(args.log_csv, {
                'phase':'train',
                'depth':depth,
                'digits':digits,
                'step':step,
                'steps':steps,
                'loss':avg_loss,
                'lr':optimizer.param_groups[0]['lr'],
                'eval_100_correct':correct_count,
                'eval_100_total':total_count,
                'eval_100_acc':eval_acc,
                'exam_type':'epoch',
                'grad_clip': args.grad_clip,
                'grad_norm': float(grad_norm) if 'grad_norm' in locals() and grad_norm is not None else ''
            }, header=['phase','depth','digits','step','steps','loss','eval_100_correct','eval_100_total','eval_100_acc','exam_type','grad_clip','grad_norm'])
        
        # Save checkpoint after each digit-depth combination
        checkpoint_path = f"checkpoints/model{model_param.hash()}_{model_param.type()}_step{step}.pt"
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        save_checkpoint(checkpoint_path, model, model_param)

        step += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the MathFormer autoregressive model.')
    parser.add_argument('--steps', type=int, default=10, help='Base number of steps (scaled by difficulty).')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--num_examples', type=int, default=10000, help='Examples per (depth,digits) combination.')
    parser.add_argument('--practice', action='store_true', help='Use smaller dataset for quick smoke test.')
    parser.add_argument('--max_digits', type=int, default=3)
    parser.add_argument('--max_depth', type=int, default=3)
    parser.add_argument('--train_print', default=False, action='store_true')
    parser.add_argument('--train_print_correct', default=False, action='store_true')
    parser.add_argument('--log_csv', type=str, default='')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine','none'], help='LR scheduler type.')
    parser.add_argument('--warmup-ratio', type=float, default=0.05, help='Fraction of total steps used for linear warmup.')
    parser.add_argument('--min-lr', type=float, default=1e-5, help='Final minimum learning rate for cosine decay.')
    parser.add_argument('--step_eval', default=False, action='store_true', help='Evaluate on 100 samples after each epoch.')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Clip gradient norm to this value (0 disables).')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--modelsize', type=str, default='small', choices=['tiny','small','medium','large'], help='Model size preset.')
    # MoE parameters
    parser.add_argument('--num_experts', type=int, default=0,
                        help='Number of experts (0 for dense model)')
    parser.add_argument('--active_experts', type=int, default=0,
                        help='Number of active experts per token')
    args = parser.parse_args()

    train(args)
