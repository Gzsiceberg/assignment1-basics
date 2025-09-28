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
