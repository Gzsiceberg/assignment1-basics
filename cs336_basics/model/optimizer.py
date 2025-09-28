import math
from typing import Iterable
import torch
from torch import nn, Tensor, softmax
from einops import rearrange, einsum
import numpy as np
from jaxtyping import Float, Int, jaxtyped, Bool
from beartype import beartype as typechecker


class SGD(torch.optim.Optimizer):
    def __init__(self, params: Iterable[nn.Parameter], lr=1e-3):
        assert lr > 0, f"Learning rate must be positive, got {lr}"
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    def step(self) -> None:  # type: ignore
        for group in self.param_groups:
            lr = group["lr"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                t = state.get("t", 0)
                grad = param.grad.data
                param.data -= (lr / math.sqrt(t + 1)) * grad
                state["t"] = t + 1
        return None


class AdaGrad(torch.optim.Optimizer):
    def __init__(self, params: Iterable[nn.Parameter], lr=1e-3):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    def step(self) -> None:  # type: ignore
        for group in self.param_groups:
            lr = group["lr"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                g2 = state.get("g2", torch.zeros_like(param))
                g2 += torch.square(param.grad)
                state["g2"] = g2

                grad = param.grad.data
                param.data -= lr * grad / (torch.sqrt(g2) + 1e-8)
        return None


class AdamW(torch.optim.Optimizer):
    def __init__(self, params: Iterable[nn.Parameter], lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self) -> None:  # type: ignore
        """
        Parameters state: 
        - t: time step per param group
        - m: first moment per param
        - v: second moment per param
        """
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                t = state.get("t", 0) + 1
                m = state.get("m", torch.zeros_like(param))
                v = state.get("v", torch.zeros_like(param))

                m = beta1 * m + (1 - beta1) * param.grad
                v = beta2 * v + (1 - beta2) * torch.square(param.grad)

                lr_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                param.data -= lr_t * m / (torch.sqrt(v) + eps)
                param.data -= lr * weight_decay * param.data

def calc_llm_memory(vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, batch_size: int) -> None:
    d_ff = 4 * d_model
    embedding_params = vocab_size * d_model
    embedding_gradients = 2 * embedding_params

    rmsnorm_params = 2 * d_model * num_layers
    rmsnorm_params_gradients = 2 * rmsnorm_params
    rmsnorm_activation = 2 * context_length * d_model * num_layers

    mha_params = 4 * d_model * d_model * num_layers
    mha_gradients = 2 * mha_params
    mha_qkv_activations = 3 * context_length * d_model * num_layers
    mha_attention_scores_activations = num_heads * context_length * context_length * num_layers
    mha_softmax_activations = num_heads * context_length * context_length * num_layers
    mha_o_activations = context_length * d_model * num_layers
    d_v = d_model // num_heads
    mha_attention_values_activations = num_heads * context_length * d_v * num_layers
    mha_activations = mha_qkv_activations + mha_o_activations + mha_attention_scores_activations + mha_softmax_activations + mha_attention_values_activations

    ffn_params = 3 * d_model * d_ff * num_layers
    ffn_params_gradients = 2 * ffn_params
    ffn_silu_activation = context_length * d_ff * num_layers
    ffn_fc1_activation = context_length * d_ff * num_layers
    ffn_fc2_activation = context_length * d_model * num_layers
    ffn_activation = ffn_silu_activation + ffn_fc1_activation + ffn_fc2_activation


    block_params = mha_params + ffn_params + rmsnorm_params

    lmfinal_rmsnorm_params = d_model
    lmfinal_rmsnorm_params_gradients = 2 * lmfinal_rmsnorm_params
    lmfinal_rmsnorm_activation = context_length * d_model

    lmhead_params = d_model * vocab_size
    lmhead_params_gradients = 2 * lmhead_params
    lmhead_params_activations = context_length * vocab_size

    cross_entropy_loss_activation = context_length

    total_params = embedding_params + block_params + lmfinal_rmsnorm_params + lmhead_params
    total_gradients = embedding_gradients + mha_gradients + ffn_params_gradients + \
        rmsnorm_params_gradients + lmfinal_rmsnorm_params_gradients + lmhead_params_gradients
    total_activations = mha_activations + ffn_activation + rmsnorm_activation + \
        lmfinal_rmsnorm_activation + lmhead_params_activations + cross_entropy_loss_activation
    
    print(f"Total parameters: {total_params:,}")
    params_memory = total_params * 4 / (1024**3)
    gradients_memory = total_gradients * 4 / (1024**3)
    print(f"constants per batch size: {(total_params + total_gradients) * 4:,}")
    print(f"activations per batch size: {total_activations * 4:,}")
    activations_memory = batch_size * total_activations * 4 / (1024**3)
    total_memory = (total_params + total_gradients + total_activations) * 4 / (1024**3)
    print(f"Parameters memory (GB): {params_memory:.2f}")
    print(f"Gradients memory (GB): {gradients_memory:.2f}")
    print(f"Activations memory (GB): {activations_memory:.2f}")
    print(f"Total memory (GB): {total_memory:.2f}")

if __name__ == "__main__":
    vocab_size = 50_257
    num_layers = 48
    d_model = 1600
    num_heads = 24
    d_ff = 6400
    max_seq_len = 1024
    theta = 100000.0
    batch_size = 1

    P = 2 * vocab_size * d_model + num_layers * (12 * d_model * d_model + 2 * d_model) + d_model
    print(f"GPT-2 XL model parameters: {P:,}")

    print("-" * 80)
    print(f"GPT-2 XL model memory usage for batch size {batch_size}:")
    calc_llm_memory(vocab_size, max_seq_len, num_layers, d_model, num_heads, batch_size)

    print("-" * 80)
    print(f"GPT-2 XL model memory usage for batch size {batch_size} and 16K context:")
    max_seq_len = 16_384
    calc_llm_memory(vocab_size, max_seq_len, num_layers, d_model, num_heads, batch_size)
