import os
from cs336_basics.checkpoints import load_checkpoint, load_model_from
from cs336_basics.model.activation import softmax_activation
from cs336_basics.model.transformer import TransformerLM
from cs336_basics.model.common import get_device
from torch import Tensor
import torch
import numpy as np
from jaxtyping import Float, Int, jaxtyped, Bool
from beartype import beartype as typechecker
from cs336_basics.tokenizer import BPETokenizer
from rich import print


class Decoder:
    temperature = 1.0
    top_p = 1.0
    end_token_id = 0
    vocab_size = 0
    tokenizer: BPETokenizer

    def __init__(self, llm: TransformerLM, device: str, tokenizer: BPETokenizer, temperature: float = 1.0, top_p: float = 1.0) -> None:
        self.llm = llm
        self.device = device
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.top_p = top_p
        self.vocab_size = self.llm.vocab_size
        self.end_token_id = self.llm.vocab_size - 1  # assuming the last token is <eos>

    def generate(self, prompts: list[int], max_token_count: int = -1) -> list[int]:
        x = torch.tensor(prompts, dtype=torch.int64, device=self.device)
        x = x.unsqueeze(0)  # add batch dimension
        assert x.shape == (1, len(prompts)), f"x.shape: {x.shape}, prompts: {len(prompts)}"
        outputs: list[int] = []
        outputs.extend(prompts)
        with torch.no_grad():
            while max_token_count == -1 or len(outputs) < max_token_count:
                seq_len = x.shape[1]
                logits = self.llm(x)
                assert logits.shape == (1, seq_len, self.vocab_size), f"logits.shape: {logits.shape}, vocab_size: {self.vocab_size}"
                probs = softmax_activation(logits, dim=-1, temperature=self.temperature)
                assert probs.shape == (1, seq_len, self.vocab_size), f"probs.shape: {probs.shape}, vocab_size: {self.vocab_size}"

                sample_token: int = -1
                if self.top_p < 1.0:
                    sample_token: int = self.top_p_sample(probs)[0]
                else:
                    threshold = np.random.rand()
                    cumulative_probs = torch.cumsum(probs, dim=-1)
                    greater_than_threshold = cumulative_probs > threshold
                    indices = torch.argmax(greater_than_threshold.to(torch.int64), dim=-1)
                    sample_token = int(indices[0, seq_len - 1].item())

                if sample_token == self.end_token_id:
                    break

                delta_x = torch.tensor([[sample_token]], device=self.device)
                x = torch.concatenate([x, delta_x], dim=-1)
                outputs.append(sample_token)
        return outputs

    @jaxtyped(typechecker=typechecker)
    def top_p_sample(self, probs: Float[Tensor, "batch seq_len vocab_size"]) -> list[int]:
        batch_size = probs.shape[0]
        seq_len = probs.shape[1]
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        assert sorted_probs.shape == probs.shape
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        threshold = np.random.rand() * self.top_p
        greater_than_threshold = cumulative_probs > threshold
        indices = torch.argmax(greater_than_threshold.to(torch.int64), dim=-1)
        assert indices.shape == (batch_size, seq_len)
        index = indices[:, seq_len - 1]
        sample_token_ids = sorted_indices[:, seq_len - 1, index]
        assert sample_token_ids.shape == (batch_size, 1), f"sample_token_ids.shape: {sample_token_ids.shape}, expected: {(batch_size, 1)}"
        sample_token_ids = sample_token_ids[:, 0]
        return sample_token_ids.tolist()

def load_tokenizer(file_prefix: str) -> BPETokenizer:
    vocab_file: str = f"{file_prefix}-vocab.dat"
    merges_file: str = f"{file_prefix}-merges.dat"
    print(f"Loading tokenizer from {vocab_file} and {merges_file}")
    tokenizer = BPETokenizer.from_files(vocab_file, merges_file, special_tokens=["<|endoftext|>"])
    return tokenizer

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        default="data/TinyStoriesV2-GPT4-train.txt",
        help="Path to the input text file.",
    )
    parser.add_argument("--exp-name", type=str, required=False, help="Experiment name", default=f"")
    parser.add_argument("--steps", type=int, required=False, help="Number of training steps", default=-1)
    parser.add_argument(
        "--checkpoint_path", type=str, required=False, help="Path to save checkpoints", default="checkpoints/"
    )
    args = parser.parse_args()
    
    chp_file_name = "latest.pt" if args.steps == -1 else f"checkpoint_{args.steps}.pt"
    if args.exp_name:
        chp_file_name = f"{args.exp_name}_{chp_file_name}"
    chp_file_path = os.path.join(args.checkpoint_path, chp_file_name)
    found_chp = os.path.exists(chp_file_path)
    if not found_chp:
        print(f"No checkpoint found at {chp_file_path}")
        exit(1)
    device = get_device()
    print(f"Loading checkpoint from {chp_file_path}")
    llm = load_model_from(chp_file_path, device=device)
    llm.eval()

    file_prefix = args.file.replace("train.txt", "train")
    file_prefix = file_prefix.replace("valid.txt", "train")
    tokenizer = load_tokenizer(file_prefix)

    decoder = Decoder(llm, device=str(device), tokenizer=tokenizer, temperature=1.0, top_p=0.2)
    output = decoder.generate([1], max_token_count=10)
    print(f"Generated output (token ids): {output}")
