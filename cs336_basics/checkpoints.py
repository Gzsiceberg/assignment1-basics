import os
import typing
import torch
from torch import nn
from cs336_basics.model.transformer import TransformerLM


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model_config: dict | None = None,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    if model_config is not None:
        checkpoint["model_config"] = model_config
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: nn.Module, optimizer: torch.optim.Optimizer
) -> int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]


def load_model_from(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    device: str | torch.device = "cpu",
) -> tuple[TransformerLM, dict]:
    checkpoint = torch.load(src)
    model_config = checkpoint.get("model_config", None)
    if model_config is None:
        raise ValueError("No model_config found in the checkpoint.")
    llm = TransformerLM(
        vocab_size=model_config["vocab_size"],
        num_layers=model_config["num_layers"],
        d_model=model_config["d_model"],
        num_heads=model_config["num_heads"],
        d_ff=model_config["d_ff"],
        max_seq_len=model_config["max_seq_len"],
        theta=model_config["theta"],
        dtype=torch.float32,
        ffn_type=model_config["ffn_type"],
        device=torch.device(device),
    )
    llm.load_state_dict(checkpoint["model_state_dict"])
    return llm, model_config
