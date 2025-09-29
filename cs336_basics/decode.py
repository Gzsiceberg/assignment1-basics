import os
from cs336_basics.checkpoints import load_model_from
from cs336_basics.common_data import DataConfig, ExperimentConfig, ModelConfig, OptimizerConfig, load_config_from_file
from cs336_basics.common_data import load_config_from_file
from cs336_basics.model.activation import softmax_activation
from cs336_basics.model.transformer import TransformerLM
from cs336_basics.model.common import get_device
from torch import Tensor
import torch
import numpy as np
from jaxtyping import Float, Int, jaxtyped, Bool
from beartype import beartype as typechecker
from cs336_basics.tokenizer import BPETokenizer


class Decoder:
    temperature = 1.0
    top_p = 1.0
    end_token_id = 0
    vocab_size = 0
    max_seq_len = 0
    tokenizer: BPETokenizer

    def __init__(
        self, llm: TransformerLM, device: str, tokenizer: BPETokenizer, temperature: float = 1.0, top_p: float = 1.0
    ) -> None:
        self.llm = llm
        self.device = device
        self.tokenizer = tokenizer
        self.temperature = temperature
        assert self.temperature > 0.0, "Temperature must be positive"
        self.top_p = top_p
        assert 0.0 < self.top_p <= 1.0, "top_p must be in (0.0, 1.0]"
        self.vocab_size = self.llm.vocab_size
        self.end_token_id = self.tokenizer.end_token_id
        self.max_seq_len = self.llm.max_seq_len

    def generate(self, prompts: str, max_token_count: int = -1) -> str:
        prompts_is_empty: bool = False
        if not prompts:
            prompts = self.tokenizer.end_token_str
            prompts_is_empty = True
        prompt_tokens = self.tokenizer.encode(prompts)
        output_tokens = self.generate_ids(prompt_tokens, max_token_count=max_token_count)
        print(f"Prompt tokens len: {len(prompt_tokens)}, Output tokens len: {len(output_tokens)}")
        output_text = self.tokenizer.decode(output_tokens if not prompts_is_empty else output_tokens[1:])
        if prompts_is_empty:
            output_text = output_text.lstrip()
        return output_text

    def generate_ids(self, prompts: list[int], max_token_count: int = -1) -> list[int]:
        x = torch.tensor(prompts, dtype=torch.int64, device=self.device)
        x = x.unsqueeze(0)  # add batch dimension
        assert x.shape == (1, len(prompts)), f"x.shape: {x.shape}, prompts: {len(prompts)}"
        outputs: list[int] = []
        outputs.extend(prompts)
        with torch.no_grad():
            while max_token_count == -1 or len(outputs) < max_token_count:
                seq_len = x.shape[1]
                if seq_len > self.max_seq_len:
                    x = x[:, -self.max_seq_len :]
                    assert x.shape == (1, self.max_seq_len), f"x.shape: {x.shape}, seq_len: {seq_len}"
                    seq_len = x.shape[1]
                logits = self.llm(x)
                assert logits.shape == (
                    1,
                    seq_len,
                    self.vocab_size,
                ), f"logits.shape: {logits.shape}, vocab_size: {self.vocab_size}"
                next_logits = logits[0, -1, :]
                assert next_logits.shape == (
                    self.vocab_size,
                ), f"next_logits.shape: {next_logits.shape}, vocab_size: {self.vocab_size}"
                probs = softmax_activation(next_logits, dim=-1, temperature=self.temperature)
                assert probs.shape == (self.vocab_size,), f"probs.shape: {probs.shape}, vocab_size: {self.vocab_size}"

                sample_token: int = -1
                if self.top_p < 1.0:
                    sample_token: int = self.top_p_sample(probs)
                else:
                    indices = torch.multinomial(probs, num_samples=1)
                    assert indices.shape == (1,), f"indices.shape: {indices.shape}, expected: {(1, 1)}"
                    sample_token = int(indices[0].item())

                if sample_token == self.end_token_id:
                    break

                delta_x = torch.tensor([[sample_token]], device=self.device)
                x = torch.concatenate([x, delta_x], dim=-1)
                outputs.append(sample_token)
        return outputs

    @jaxtyped(typechecker=typechecker)
    def top_p_sample(self, probs: Float[Tensor, "vocab_size"]) -> int:
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        assert sorted_probs.shape == probs.shape
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        cutoff_idx = torch.searchsorted(cumulative_probs, self.top_p)
        trimmed_probs = sorted_probs[: cutoff_idx + 1]
        trimmed_idxs = sorted_indices[: cutoff_idx + 1]
        trimmed_probs = trimmed_probs / torch.sum(trimmed_probs)
        next_token = torch.multinomial(trimmed_probs, num_samples=1)
        sample_token_id = trimmed_idxs[next_token]
        return int(sample_token_id.item())


def load_tokenizer(file_prefix: str) -> BPETokenizer:
    vocab_file: str = f"{file_prefix}-vocab.dat"
    merges_file: str = f"{file_prefix}-merges.dat"
    print(f"Loading tokenizer from {vocab_file} and {merges_file}")
    tokenizer = BPETokenizer.from_files(vocab_file, merges_file, special_tokens=["<|endoftext|>"])
    return tokenizer


if __name__ == "__main__":
    import argparse
    from rich import print

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="defaul.yaml", help="Path to config file (json or yaml)")
    parser.add_argument("--steps", type=int, required=False, default=-1, help="Checkpoint step to load, -1 for latest")
    args = parser.parse_args()
    if not os.path.exists(args.config):
        print(f"Config file {args.config} does not exist.")
        exit(1)
    config = load_config_from_file(args.config)
    # model_config: ModelConfig = ModelConfig(**config.get("model", {}))
    data_config: DataConfig = DataConfig(**config.get("data", {}))
    exp_config: ExperimentConfig = ExperimentConfig(**config.get("experiment", {}))
    opt_config: OptimizerConfig = OptimizerConfig(**config.get("optimizer", {}))
    # print(f"Model config: {model_config}")
    print(f"Data config: {data_config}")
    print(f"Experiment config: {exp_config}")
    print(f"Optimizer config: {opt_config}")

    print("-" * 120)
    chp_file_name = "latest.pt" if args.steps == -1 else f"checkpoint_{args.steps}.pt"
    if exp_config.name:
        chp_file_name = f"{exp_config.name}_{chp_file_name}"
    chp_file_path = os.path.join(exp_config.checkpoints_path, chp_file_name)
    found_chp = os.path.exists(chp_file_path)
    if not found_chp:
        print(f"No checkpoint found at {chp_file_path}")
        exit(1)
    device = get_device()
    print(f"Loading checkpoint from {chp_file_path}")
    llm, model_config_dict = load_model_from(chp_file_path, device=device)
    llm.eval()

    # convert TinyStoriesV2-GPT4-train-bpe-merged.npy to TinyStoriesV2-GPT4-train
    # convert TinyStoriesV2-GPT4-train-bpe.npy to TinyStoriesV2-GPT4-train
    # use regex to remove -bpe(-merged)?\.npy$
    import regex as re

    file_prefix = re.sub(r"-bpe(-merged)?\.npy$", "", data_config.train_data)
    tokenizer: BPETokenizer = load_tokenizer(file_prefix)

    decoder = Decoder(llm, device=str(device), tokenizer=tokenizer, temperature=1.0, top_p=1.0)
    output_str = decoder.generate("", max_token_count=1000)
    print(f"Output: {output_str}")
