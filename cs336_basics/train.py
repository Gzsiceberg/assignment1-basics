from datetime import datetime
import json
import os
from time import time
import typing
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
from cs336_basics.model.optimizer import AdamW, gradient_clipping
from cs336_basics.model.transformer import TransformerLM
from rich.progress import Progress, TaskID, track
from logger import print


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

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, default="default.yaml", help="Path to config file (json or yaml)")
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    parser.add_argument("--restore", action="store_true", help="Restore from the latest checkpoint if available")
    args = parser.parse_args()
    if not os.path.exists(args.config):
        print(f"Config file {args.config} does not exist.")
        exit(1)
    config = load_config_from_file(args.config)
    model_config: ModelConfig = ModelConfig(**config.get("model", {}))
    data_config: DataConfig = DataConfig(**config.get("data", {}))
    exp_config: ExperimentConfig = ExperimentConfig(**config.get("experiment", {}))
    opt_config: OptimizerConfig = OptimizerConfig(**config.get("optimizer", {}))
    setup_logging(exp_config)
    print(f"Model config: {model_config}")
    print(f"Data config: {data_config}")
    print(f"Experiment config: {exp_config}")
    print(f"Optimizer config: {opt_config}")

    calculator.calc_llm_memory(
        model_config.vocab_size,
        model_config.max_seq_len,
        model_config.num_layers,
        model_config.d_model,
        model_config.num_heads,
        exp_config.batch_size,
        ffn_type=model_config.ffn_type,
    )
    if args.profile:
        exit(0)

    print("-" * 120)
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
    print(f"Training data has {data.shape[0]} tokens.")
    valid_data = np.memmap(data_config.valid_data, mode="r", dtype=np.int16)
    print(f"Validation data has {valid_data.shape[0]} tokens.")

    iteration = 0
    latest_file_name = "latest.pt"
    if exp_config.name:
        latest_file_name = f"{exp_config.name}_{latest_file_name}"
    latest_file_path = os.path.join(exp_config.checkpoints_path, latest_file_name)
    found_latest = os.path.exists(latest_file_path)
    if args.restore and found_latest:
        print(f"Loading checkpoint from {latest_file_path}")
        iteration = load_checkpoint(latest_file_path, llm, opt)
        print(f"Resumed from iteration {iteration}")
    elif not args.restore and found_latest:
        # ask the user if they want to overwrite the existing checkpoint
        response = input(
            f"Warning: Found existing checkpoint at {latest_file_path}. Do you want to overwrite it? (y/n): "
        )
        if response.lower() != "y":
            print("Exiting without overwriting the checkpoint.")
            exit(0)

    device = get_device()
    device_str = str(device)

    last_batch_loss = 0
    last_checkpoint_time = time()
    last_print_time = time()

    with Progress() as progress:
        task = progress.add_task(f"[green]Training loss {last_batch_loss:.4f}", total=exp_config.steps - iteration)

        for t in range(iteration, exp_config.steps):
            x, y = get_batch(data, exp_config.batch_size, model_config.max_seq_len, device_str)
            opt.zero_grad(set_to_none=True)  # Reset the gradients for all learnable parameters.
            logits = llm(x)  # Forward pass to get logits.
            loss = cross_entropy(logits, y)  # Compute the cross-entropy loss.
            if opt_config.grad_clip > 0:
                gradient_clipping(llm.parameters(), opt_config.grad_clip)

            current_loss = loss.cpu().item()
            now = time()
            if now - last_print_time >= 2 or t == exp_config.steps - 1:
                last_print_time = now
                last_batch_loss = current_loss
                print(f"Step {t}: loss {last_batch_loss:.4f}")

            # Update progress bar description with current loss
            progress.update(task, description=f"[green]Training loss {current_loss:.4f}", advance=1)

            loss.backward()  # Run backward pass, which computes gradients.
            opt.step()  # Update parameters based on computed gradients.

            if now - last_checkpoint_time >= exp_config.checkpoints_interval * 60 or t == exp_config.steps - 1:
                last_checkpoint_time = now
                if exp_config.name:
                    checkpoint_file_name = f"{exp_config.name}_checkpoint_{t+1}.pt"
                else:
                    checkpoint_file_name = f"checkpoint_{t+1}.pt"
                checkpoint_file = os.path.join(exp_config.checkpoints_path, checkpoint_file_name)
                print(f"Saving checkpoint to {checkpoint_file}")
                model_config_dict = model_config.model_dump()
                save_checkpoint(llm, opt, t + 1, checkpoint_file, model_config=model_config_dict)

                # Also save a latest checkpoint
                latest_file = os.path.join(exp_config.checkpoints_path, latest_file_name)
                save_checkpoint(llm, opt, t + 1, latest_file, model_config=model_config_dict)

                valid_loss = calc_validation_loss(
                    llm, valid_data, exp_config.batch_size, model_config.max_seq_len, device_str
                )
                print(f"Validation loss after step {t+1}: {valid_loss:.4f}")
