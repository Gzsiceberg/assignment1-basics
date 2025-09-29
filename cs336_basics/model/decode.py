from cs336_basics.model.activation import softmax_activation
from cs336_basics.model.transformer import TransformerLM
from torch import Tensor
import torch
import numpy as np
from jaxtyping import Float, Int, jaxtyped, Bool
from beartype import beartype as typechecker


class Decoder:
    temperature = 1.0
    top_p = 1.0
    end_token_id = 0
    vocab_size = 0

    def __init__(self, llm: TransformerLM, device: str, temperature: float = 1.0, top_p: float = 1.0) -> None:
        self.llm = llm
        self.device = device
        self.temperature = temperature
        self.top_p = top_p
        self.vocab_size = self.llm.vocab_size
        self.end_token_id = self.llm.vocab_size - 1  # assuming the last token is <eos>

    def generate(self, prompts: list[int], max_token_count: int = -1) -> list[int]:
        self.llm.eval()
        x = torch.tensor(prompts, dtype=torch.int64, device=self.device)
        assert x.dim() == 1, f"prompts should be a 1D list of token ids, but got {x.shape}"
        assert x.shape[0] == len(prompts), "prompts should be a list of token ids"
        outputs: list[int] = []
        outputs.extend(prompts)
        with torch.no_grad():
            while max_token_count == -1 or len(outputs) < max_token_count:
                logits = self.llm(x)
                probs = softmax_activation(logits, dim=-1, temperature=self.temperature)
                assert probs.shape == (1, self.vocab_size)

                sample_token: int = -1
                if self.top_p < 1.0:
                    sample_token: int = self.top_p_sample(probs)
                else:
                    threshold = np.random.rand()
                    cumulative_probs = torch.cumsum(probs, dim=-1)
                    greater_than_threshold = cumulative_probs > threshold
                    indices = torch.argmax(greater_than_threshold.to(torch.int64), dim=-1)
                    sample_token = int(indices[0].item())

                if sample_token == self.end_token_id:
                    break

                delta_x = torch.tensor([sample_token], device=self.device)
                x = torch.concatenate([x, delta_x])
                outputs.append(sample_token)

        self.llm.train()
        return outputs

    @jaxtyped(typechecker=typechecker)
    def top_p_sample(self, probs: Float[Tensor, "batch vocab_size"]) -> int:
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        assert sorted_probs.shape == probs.shape
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        threshold = np.random.rand() * self.top_p
        greater_than_threshold = cumulative_probs > threshold
        indices = torch.argmax(greater_than_threshold.to(torch.int64), dim=-1)
        assert indices.shape == (probs.shape[0],)
        token_index = int(indices[0].item())
        sample_token = int(sorted_indices[0, token_index].item())
        return sample_token
