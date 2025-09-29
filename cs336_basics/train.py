from datetime import datetime
import os
from time import time
import torch
from torch import nn
import numpy as np
import random

from cs336_basics.checkpoints import load_checkpoint, save_checkpoint
from cs336_basics.common_data import DataConfig, ExperimentConfig, ModelConfig, OptimizerConfig, load_config_from_file
from cs336_basics.data_loader import get_batch
from cs336_basics.logger import setup_logging
from cs336_basics.model import calculator
from cs336_basics.model.common import get_device
from cs336_basics.model.loss import cross_entropy
from cs336_basics.model.optimizer import AdamW, get_lr_cosine_schedule, gradient_clipping
from cs336_basics.model.transformer import TransformerLM
from rich.progress import Progress, TaskID, track

is_main_file = __name__ == "__main__"

# Torch
seed = 0
torch.manual_seed(seed)
# NumPy
np.random.seed(seed)
# Python
random.seed(seed)


def calc_validation_loss(
    llm: TransformerLM,
    data: np.memmap,
    batch_size: int,
    max_seq_len: int,
    device_str: str,
    evl_iters: int = 500,
) -> float:
    llm.eval()
    avg_loss = 0.0
    with torch.no_grad():
        for t in track(range(evl_iters), description="[green]Evaluating..."):
            x, y = get_batch(data, batch_size, max_seq_len, device_str)
            logits = llm(x)
            loss = cross_entropy(logits, y)
            avg_loss += loss.cpu().item() / evl_iters
    llm.train()
    return avg_loss


if is_main_file:
    import argparse
    from logger import print_and_log

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, default="default.yaml", help="Path to config file (json or yaml)")
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    parser.add_argument("--restore", action="store_true", help="Restore from the latest checkpoint if available")
    args = parser.parse_args()
    if not os.path.exists(args.config):
        print_and_log(f"Config file {args.config} does not exist.")
        exit(1)
    config = load_config_from_file(args.config)
    model_config: ModelConfig = ModelConfig(**config.get("model", {}))
    data_config: DataConfig = DataConfig(**config.get("data", {}))
    exp_config: ExperimentConfig = ExperimentConfig(**config.get("experiment", {}))
    opt_config: OptimizerConfig = OptimizerConfig(**config.get("optimizer", {}))
    setup_logging(exp_config)
    print_and_log(f"Model config: {model_config}")
    print_and_log(f"Data config: {data_config}")
    print_and_log(f"Experiment config: {exp_config}")
    print_and_log(f"Optimizer config: {opt_config}")

    calculator.calc_llm_memory(
        model_config.vocab_size,
        model_config.max_seq_len,
        model_config.num_layers,
        model_config.d_model,
        model_config.num_heads,
        exp_config.batch_size,
        ffn_type=model_config.ffn_type,
    )
    t_flops = calculator.calc_flops(
        seq_len=model_config.max_seq_len,
        d_model=model_config.d_model,
        num_layers=model_config.num_layers,
        batch_size=exp_config.batch_size,
        vocab_size=model_config.vocab_size,
        ffn_type=model_config.ffn_type,
        d_ff=model_config.d_ff,
    )
    t_params = calculator.calc_num_params(
        vocab_size=model_config.vocab_size,
        d_model=model_config.d_model,
        num_layers=model_config.num_layers,
        num_heads=model_config.num_heads,
        ffn_type=model_config.ffn_type,
        d_ff=model_config.d_ff,
    )
    print_and_log("-" * 120)
    t_flops *= 3  # for gradiend update
    t_flops_per_token = t_flops / (exp_config.batch_size * model_config.max_seq_len)
    print_and_log(f"FLOPs per step: {t_flops/1e6:,.2f} MFLOPs. FLOPs per token: {t_flops_per_token/1e6:,.2f} MFLOPs")
    estimated_flops = t_params * 6.0
    print_and_log(f"model parameters: {t_params/1e6:.2f} M, estimated FLOPs per token: {estimated_flops/1e6:,.2f} MFLOPs")

    total_training_flops = t_flops * exp_config.steps
    print_and_log(f"Total training: {total_training_flops/1e12:,.2f} TFLOPs")
    if args.profile:
        exit(0)

    print_and_log("-" * 120)
    llm = TransformerLM(
        vocab_size=model_config.vocab_size,
        num_layers=model_config.num_layers,
        d_model=model_config.d_model,
        num_heads=model_config.num_heads,
        d_ff=model_config.d_ff,
        max_seq_len=model_config.max_seq_len,
        theta=model_config.theta,
        ffn_type=model_config.ffn_type,
        dtype=torch.float32 if model_config.dtype == "float32" else torch.bfloat16,
        device=get_device(),
    )

    opt = AdamW(
        llm.parameters(),
        lr=opt_config.learning_rate,
        beta1=opt_config.betas[0],
        beta2=opt_config.betas[1],
        weight_decay=opt_config.weight_decay,
    )
    if not os.path.exists(exp_config.checkpoints_path):
        os.makedirs(exp_config.checkpoints_path)

    data = np.memmap(data_config.train_data, mode="r", dtype=np.int16)
    print_and_log(f"Training data has {data.shape[0]} tokens.")
    valid_data = np.memmap(data_config.valid_data, mode="r", dtype=np.int16)
    print_and_log(f"Validation data has {valid_data.shape[0]} tokens.")

    iteration = 0
    latest_file_name = "latest.pt"
    if exp_config.name:
        latest_file_name = f"{exp_config.name}_{latest_file_name}"
    latest_file_path = os.path.join(exp_config.checkpoints_path, latest_file_name)
    found_latest = os.path.exists(latest_file_path)
    if args.restore and found_latest:
        print_and_log(f"Loading checkpoint from {latest_file_path}")
        iteration = load_checkpoint(latest_file_path, llm, opt)
        print_and_log(f"Resumed from iteration {iteration}")
    elif not args.restore and found_latest:
        # ask the user if they want to overwrite the existing checkpoint
        response = input(
            f"Warning: Found existing checkpoint at {latest_file_path}. Do you want to overwrite it? (y/n): "
        )
        if response.lower() != "y":
            print_and_log("Exiting without overwriting the checkpoint.")
            exit(0)

    device = get_device()
    device_str = str(device)

    last_batch_loss = 0
    last_checkpoint_time = time()
    last_print_time = time()
    last_batch_count = 0

    with Progress() as progress:
        task = progress.add_task(f"[green]Training loss", total=exp_config.steps - iteration)

        warmup_steps: int = int(opt_config.warmup_ratio * exp_config.steps)
        cosine_cycle_steps: int = round(opt_config.cosine_cycle_ratio * exp_config.steps)
        if opt_config.lr_schedule == "cosine":
            print_and_log(f"Using cosine learning rate schedule with min {opt_config.lr_min}, max {opt_config.lr_max}, "
                          f"warmup steps {warmup_steps}, cosine cycle steps {cosine_cycle_steps}")
        for t in range(iteration, exp_config.steps):
            x, y = get_batch(data, exp_config.batch_size, model_config.max_seq_len, 
                             device_str)
            opt.zero_grad(set_to_none=True)  # Reset the gradients for all learnable parameters.
            logits = llm(x)  # Forward pass to get logits.
            loss = cross_entropy(logits, y)  # Compute the cross-entropy loss.
            if opt_config.grad_clip > 0:
                gradient_clipping(llm.parameters(), opt_config.grad_clip)
            

            current_loss = loss.item()
            now = time()
            last_batch_count += 1
            if now - last_print_time >= 2 or t == exp_config.steps - 1:
                last_batch_loss = current_loss
                last_train_tokens = exp_config.batch_size * model_config.max_seq_len * last_batch_count
                tokens_per_sec = int(last_train_tokens / (now - last_print_time))
                last_tflops = (t_flops * last_batch_count) / 1e12
                tflops_per_sec = last_tflops / (now - last_print_time)
                delta_time = now - last_print_time
                print_and_log(f"Step {t}: loss {last_batch_loss:.4f} tokens/sec {tokens_per_sec:,} Tflops/sec {tflops_per_sec:,.2f}")
                last_batch_count = 0
                last_print_time = now


            lr: float = opt_config.learning_rate
            if opt_config.lr_schedule == "cosine":
                lr: float = get_lr_cosine_schedule(t, opt_config.lr_max, opt_config.lr_min,
                                                   warmup_steps, cosine_cycle_steps)
            for param_group in opt.param_groups:
                param_group["lr"] = lr
            # Update progress bar description with current loss
            progress.update(task, description=f"[green]loss {current_loss:.4f} lr {lr:.4f}", advance=1)

            loss.backward()  # Run backward pass, which computes gradients.
            opt.step()  # Update parameters based on computed gradients.

            if now - last_checkpoint_time >= exp_config.checkpoints_interval * 60 or t == exp_config.steps - 1:
                last_checkpoint_time = now
                if exp_config.name:
                    checkpoint_file_name = f"{exp_config.name}_checkpoint_{t+1}.pt"
                else:
                    checkpoint_file_name = f"checkpoint_{t+1}.pt"
                checkpoint_file = os.path.join(exp_config.checkpoints_path, checkpoint_file_name)
                print_and_log(f"Saving checkpoint to {checkpoint_file}")
                model_config_dict = model_config.model_dump()
                save_checkpoint(llm, opt, t + 1, checkpoint_file, model_config=model_config_dict)

                # Also save a latest checkpoint
                latest_file = os.path.join(exp_config.checkpoints_path, latest_file_name)
                save_checkpoint(llm, opt, t + 1, latest_file, model_config=model_config_dict)

                valid_loss = calc_validation_loss(
                    llm, valid_data, exp_config.batch_size, model_config.max_seq_len, device_str,
                    evl_iters=exp_config.eval_steps,
                )
                print_and_log(f"Validation loss. step {t+1}: {valid_loss:.4f}")
