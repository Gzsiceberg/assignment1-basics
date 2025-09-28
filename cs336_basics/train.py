import os
from types import NoneType
from typing import Callable, Iterable, Optional, overload
import typing
import torch
from torch import nn
import numpy as np
import random
from einops import rearrange, einsum
from jaxtyping import Float, Int, jaxtyped
from beartype import beartype as typechecker
from cs336_basics.data_loader import get_batch
from cs336_basics.model.linear import Linear
from cs336_basics.model.common import get_device
from cs336_basics.model.loss import cross_entropy
from cs336_basics.model.optimizer import SGD, AdaGrad, AdamW, calc_llm_memory, gradient_clipping
from cs336_basics.model.transformer import TransformerLM

is_main_file = __name__ == "__main__"

# Torch
seed = 0
torch.manual_seed(seed)
# NumPy
np.random.seed(seed)
# Python
random.seed(seed)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: nn.Module, optimizer: torch.optim.Optimizer
) -> int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]


if is_main_file:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        help="Path to save the checkpoint",
        default="data/TinyStoriesV2-GPT4-train-bpe-merged.npy",
    )
    """
    Optimizer parameters
    """
    parser.add_argument("--lr", type=float, required=False, help="Learning rate", default=1e-3)
    parser.add_argument("--weight_decay", type=float, required=False, help="Weight decay", default=0.01)
    parser.add_argument("--betas", type=float, nargs=2, required=False, help="Betas for AdamW", default=(0.9, 0.999))
    """
    Model parameters
    vocab_size = 50_257 num_layers = 48
    d_model = 1600  num_heads = 24
    d_ff = 6400  max_seq_len = 1024
    theta = 100000.0  ffn_type = "silu"
    """
    parser.add_argument("--vocab_size", type=int, required=False, help="Vocabulary size", default=10000)
    parser.add_argument("--num_layers", type=int, required=False, help="Number of layers", default=2)
    parser.add_argument("--d_model", type=int, required=False, help="Model dimension", default=128)
    parser.add_argument("--num_heads", type=int, required=False, help="Number of attention heads", default=8)
    parser.add_argument("--d_ff", type=int, required=False, help="Feedforward dimension", default=512)
    parser.add_argument("--max_seq_len", type=int, required=False, help="Maximum sequence length", default=64)
    parser.add_argument("--theta", type=float, required=False, help="RoPE parameter", default=100000.0)
    parser.add_argument("--ffn_type", type=str, required=False, help="Feedforward network type", default="silu")
    """
    Gradient parameters
    """
    parser.add_argument("--grad_clip", type=float, required=False, help="Gradient clipping value", default=0)
    parser.add_argument("--batch_size", type=int, required=False, help="Batch size", default=16)
    parser.add_argument("--steps", type=int, required=False, help="Number of training steps", default=1000)
    """
    Checkpoint parameters
    """
    parser.add_argument(
        "--checkpoint_interval", type=float, required=False, help="Checkpoint save interval percent", default=0.05
    )
    parser.add_argument(
        "--checkpoint_path", type=str, required=False, help="Path to save checkpoints", default="checkpoints/"
    )
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    args = parser.parse_args()

    calc_llm_memory(
        args.vocab_size,
        args.max_seq_len,
        args.num_layers,
        args.d_model,
        args.num_heads,
        args.batch_size,
        ffn_type=args.ffn_type,
    )
    if args.profile:
        exit(0)

    llm = TransformerLM(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        theta=args.theta,
        device=get_device(),
        dtype=torch.float32,
        ffn_type=args.ffn_type,
    )
    opt = AdamW(llm.parameters(), lr=args.lr, beta1=args.betas[0], beta2=args.betas[1], weight_decay=args.weight_decay)
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    data = np.memmap(args.file, mode="r", dtype=np.int16)
    iteration = 0
    if os.path.exists(os.path.join(args.checkpoint_path, "latest.pt")):
        print(f"Loading checkpoint from {os.path.join(args.checkpoint_path, 'latest.pt')}")
        iteration = load_checkpoint(os.path.join(args.checkpoint_path, "latest.pt"), llm, opt)
        print(f"Resumed from iteration {iteration}")

    device = get_device()
    device_str = str(device)
    from rich.progress import track

    checkpoint_interval = max(1, int(args.steps * args.checkpoint_interval))
    checkpoint_step = checkpoint_interval
    while checkpoint_step <= iteration:
        checkpoint_step += checkpoint_interval

    for t in track(range(iteration, args.steps)):
        x, y = get_batch(data, args.batch_size, args.max_seq_len, device_str)
        opt.zero_grad(set_to_none=True)  # Reset the gradients for all learnable parameters.
        logits = llm(x)  # Forward pass to get logits.
        loss = cross_entropy(logits, y)  # Compute the cross-entropy loss.
        if args.grad_clip > 0:
            gradient_clipping(llm.parameters(), args.grad_clip)
        if t % 100 == 0 or t == args.steps - 1:
            print(f"Step {t}: loss {loss.cpu().item()}")
        loss.backward()  # Run backward pass, which computes gradients.
        opt.step()  # Update parameters based on computed gradients.

        if t + 1 == checkpoint_step or t == args.steps - 1:
            checkpoint_file = os.path.join(args.checkpoint_path, f"checkpoint_{t+1}.pt")
            print(f"Saving checkpoint to {checkpoint_file}")
            save_checkpoint(llm, opt, t + 1, checkpoint_file)

            # Also save a latest checkpoint
            latest_file = os.path.join(args.checkpoint_path, "latest.pt")
            save_checkpoint(llm, opt, t + 1, latest_file)
            checkpoint_step += checkpoint_interval
