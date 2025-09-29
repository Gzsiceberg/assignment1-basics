import math
from typing import Iterable
import torch
from torch import nn

def get_lr_cosine_schedule(step: int, a_max: float, a_min: float, t_warmup: int, t_cooldown: int) -> float:
    if step < t_warmup:
        return a_max * step / t_warmup
    elif step > t_cooldown:
        return a_min
    a = a_min + 0.5 * (a_max - a_min) * (1 + math.cos(math.pi * (step - t_warmup) / (t_cooldown - t_warmup)))
    return a


def gradient_clipping(parameters: Iterable[nn.Parameter], max_l2_norm: float) -> None:
    sum_norm_squared = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        param_norm = p.grad.data.square().sum()
        sum_norm_squared += param_norm.item()
    total_norm = math.sqrt(sum_norm_squared)
    if total_norm < max_l2_norm:
        return
    clip_coef = max_l2_norm / (total_norm + 1e-6)
    for p in parameters:
        if p.grad is None:
            continue
        p.grad.data.mul_(clip_coef)


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
    def __init__(self, params: Iterable[nn.Parameter], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
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
            (beta1, beta2) = group["betas"]
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

                state["t"] = t
                state["m"] = m
                state["v"] = v

