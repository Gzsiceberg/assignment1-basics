import math
from typing import Iterable
import torch
from torch import nn, Tensor, softmax
from einops import rearrange, einsum
import numpy as np
from jaxtyping import Float, Int, jaxtyped, Bool
from beartype import beartype as typechecker
from rich.progress import track


@jaxtyped(typechecker=typechecker)
def get_batch(
    data: Int[np.ndarray, "num_samples"], batch_size: int, context_length: int, device: str,
    specifed_start: int | None = None, all_random: bool = True
) -> tuple[Int[torch.Tensor, "batch seq_len"], Int[torch.Tensor, "batch seq_len"]]:
    """
    TODO: the instance will have data in two sequences, depending on the context length.
    do we need to handle this case? do we need to ensure that the two sequences are not mixed?
    one solution is to provide token position information, but this is not implemented yet.
    For now, we assume that the data is a single sequence.
    """
    shape = data.shape
    num_samples = shape[0]
    num_samples -= context_length

    if specifed_start is not None:
        if specifed_start < 0 or specifed_start + batch_size > num_samples:
            raise ValueError("Specified start index is out of bounds.")
        start_idx = np.arange(specifed_start, specifed_start + batch_size)
    elif all_random:
        start_idx = np.random.randint(0, num_samples, batch_size)
    else:
        if num_samples - batch_size <= 0:
            raise ValueError("Not enough samples to create a batch with the given context length.")
        start = np.random.randint(0, num_samples - batch_size)
        start_idx = np.arange(start, start + batch_size)

    assert start_idx.shape == (batch_size,), f"Start index shape {start_idx.shape} does not match expected shape {(batch_size,)}"

    start_idx = rearrange(start_idx, "batch -> batch 1")
    idx_range = np.arange(context_length)
    idx_range = rearrange(idx_range, "seq_len -> 1 seq_len")
    batch_indices = start_idx + idx_range

    x = torch.from_numpy(data[batch_indices])
    y = torch.from_numpy(data[batch_indices + 1])
    assert x.shape == (
        batch_size,
        context_length,
    ), f"Input shape {x.shape} does not match expected shape {(batch_size, context_length)}"
    assert y.shape == (
        batch_size,
        context_length,
    ), f"Label shape {y.shape} does not match expected shape {(batch_size, context_length)}"

    if device.startswith("cuda"):
        x = x.pin_memory()
        y = y.pin_memory()
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y
