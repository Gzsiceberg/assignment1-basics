import math
from typing import Iterable
import torch
from torch import nn
torch.optim.SGD

def get_lr_cosine_schedule(step: int, a_max: float, a_min: float, t_warmup: int, t_cooldown: int) -> float:
    if step < t_warmup:
        return a_max * step / t_warmup
    elif step > t_cooldown:
        return a_min
    a = a_min + 0.5 * (a_max - a_min) * (1 + math.cos(math.pi * (step - t_warmup) / (t_cooldown - t_warmup)))
    return a


@torch.no_grad()
@torch.compile
def gradient_clipping(parameters: Iterable[nn.Parameter], max_l2_norm: float) -> torch.Tensor:
    if max_l2_norm <= 0:
        return torch.tensor(0.0)
    grads = [p.grad for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return torch.tensor(0.0)

    device = grads[0].device
    sum_sq: torch.Tensor = torch.zeros((), device=device, dtype=torch.float32)
    for g in grads:
        sum_sq += g.to(torch.float32).square().sum()
    total_norm: torch.Tensor = sum_sq.sqrt()
    max_n = torch.as_tensor(max_l2_norm, device=device, dtype=torch.float32)
    clip_coef: torch.Tensor = max_n / (total_norm + 1e-6)
    clip_coef = torch.clamp_max(clip_coef, 1.0)
    for g in grads:
        g.mul_(clip_coef)
    return total_norm


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
    defaults: dict
    def __init__(self, params: Iterable[nn.Parameter], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, self.defaults)
    
    def __str__(self) -> str:
        return f"AdamW(lr={self.defaults['lr']}, betas={self.defaults['betas']}, eps={self.defaults['eps']}, weight_decay={self.defaults['weight_decay']})"

    @torch.no_grad()
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
                t: int = state.get("t", 0) + 1
                m: torch.Tensor = state.get("m", torch.zeros_like(param))
                v: torch.Tensor = state.get("v", torch.zeros_like(param))

                m = beta1 * m + (1 - beta1) * param.grad
                v = beta2 * v + (1 - beta2) * torch.square(param.grad)

                lr_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                param.addcdiv_(m, torch.sqrt(v) + eps, value=-lr_t)
                param.mul_(1 - lr * weight_decay)
                # param.data -= lr_t * m / (torch.sqrt(v) + eps)
                # param.data -= lr * weight_decay * param.data

                state["t"] = t
                state["m"] = m
                state["v"] = v

