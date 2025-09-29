import torch
from einops import rearrange, repeat
from jaxtyping import Float, Int, jaxtyped, Bool
from beartype import beartype as typechecker


@jaxtyped(typechecker=typechecker)
def log_softmax(x: Float[torch.Tensor, "... num_classes"], dim: int = -1) -> Float[torch.Tensor, "... num_classes"]:
    """
    x: (..., num_classes)
    Output: (..., num_classes)
    """
    x_max = x.max(dim=dim, keepdim=True).values
    x = x - x_max
    x_exp_sum = torch.exp(x).sum(dim=dim, keepdim=True)
    log_probs = x - torch.log(x_exp_sum)
    return log_probs


@jaxtyped(typechecker=typechecker)
def cross_entropy(
    logits: Float[torch.Tensor, "... num_classes"],
    targets: Int[torch.Tensor, "..."],
) -> Float[torch.Tensor, ""]:
    """
    logits: (..., num_classes)
    targets: (...,) with values in [0, num_classes-1]
    Output: scalar loss
    """
    log_probs = log_softmax(logits, dim=-1)
    index_log_probs = torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    assert index_log_probs.shape == targets.shape, f"{index_log_probs.shape}, {targets.shape}"
    loss = -index_log_probs.mean()
    return loss


@jaxtyped(typechecker=typechecker)
def perplexity(loss: Float[torch.Tensor, ""]) -> Float[torch.Tensor, ""]:
    """
    loss: scalar loss
    FLOPS: 1
    Output: scalar perplexity
    """
    ppl = torch.exp(loss)
    return ppl

if __name__ == "__main__":
        inputs = torch.tensor(
            [
                [
                    [0.1088, 0.1060, 0.6683, 0.5131, 0.0645],
                    [0.4538, 0.6852, 0.2520, 0.3792, 0.2675],
                    [0.4578, 0.3357, 0.6384, 0.0481, 0.5612],
                    [0.9639, 0.8864, 0.1585, 0.3038, 0.0350],
                ],
                [
                    [0.3356, 0.9013, 0.7052, 0.8294, 0.8334],
                    [0.6333, 0.4434, 0.1428, 0.5739, 0.3810],
                    [0.9476, 0.5917, 0.7037, 0.2987, 0.6208],
                    [0.8541, 0.1803, 0.2054, 0.4775, 0.8199],
                ],
            ]
        )
        targets = torch.tensor([[1, 0, 2, 2], [4, 1, 4, 0]])
        loss = cross_entropy(inputs, targets)
        print(loss)
