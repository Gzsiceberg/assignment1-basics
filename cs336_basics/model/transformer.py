import torch
from torch import nn, Tensor
from einops import rearrange, einsum
import numpy as np
from jaxtyping import Float, Int, jaxtyped, Bool
from beartype import beartype as typechecker
from cs336_basics.model.attention import MultiHeadAttention
from cs336_basics.model.embedding import Embedding
from cs336_basics.model.linear import Linear
from cs336_basics.model.rmsnorm import RMSNorm
from cs336_basics.model.rope import RoPE
from cs336_basics.model.swiglu import SwiGLU


class TransformerBlock(nn.Module):
    multi_head_attention: MultiHeadAttention
    rms_norm1: RMSNorm
    rms_norm2: RMSNorm
    ffn: SwiGLU

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        rope: RoPE | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        """
        Parameters: 4 * d_model^2 (MHA) + 3 * d_model * d_ff (SwiGLU) + 2 * d_model (RMSNorms)
        """
        d_k: int = d_model // num_heads
        if rope is not None:
            assert rope.d_k == d_k, f"RoPE d_k {rope.d_k} must match model d_model/num_heads {d_k}"
        else:
            rope = RoPE(theta=theta, d_k=d_k, max_seq_len=max_seq_len, device=device)
        self.rms_norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads, rope=rope, device=device, dtype=dtype)

        self.rms_norm2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    @jaxtyped(typechecker=typechecker)
    def forward(
        self,
        x: Float[Tensor, "batch_size ... seq_len d_model"],
        token_positions: Int[Tensor, "batch_size ... seq_len"] | None = None,
    ) -> Float[Tensor, " batch_size ... seq_len d_model"]:
        """
        x: (batch_size, ..., seq_len, d_model)
        token_positions: (batch_size, ..., seq_len) or None
        FLOPS_MHA: (4 * seq_len^2 * d_model + 8 * d_model^2 * seq_len) * ... (batch size and other dimensions)
        FLOPS_FFN: 6 * d_model * d_ff * seq_len * ... (batch size and other dimensions)
        FLOPS: FLOPS_MHA + FLOPS_FFN
        """
        norm_x = self.rms_norm1(x)
        atten_x = self.multi_head_attention(norm_x, token_positions=token_positions)
        y0 = x + atten_x
        norm_y0 = self.rms_norm2(y0)
        ffn_y = self.ffn(norm_y0)
        y = y0 + ffn_y
        return y

    def extra_repr(self) -> str:
        return f"rms_norm1={self.rms_norm1}, multi_head_attention={self.multi_head_attention}, rms_norm2={self.rms_norm2}, ffn={self.ffn}"


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        """
        Parameters: 
            - Embedding: (vocab_size, d_model)
            - transformer_blocks: num_layers * (4 * d_model^2 + 3 * d_model * d_ff + 2 * d_model)
            - rms_norm: d_model
            - lm_head: (d_model, vocab_size)
            - Total Parameters: Embedding + transformer_blocks + rms_norm + lm_head
        """
        d_k: int = d_model // num_heads
        self.rope = RoPE(theta=theta, d_k=d_k, max_seq_len=max_seq_len, device=device)
        self.token_embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=max_seq_len,
                    theta=theta,
                    rope=self.rope,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.rms_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    @jaxtyped(typechecker=typechecker)
    def forward(
        self,
        token_ids: Int[Tensor, "batch_size ... seq_len"],
        token_positions: Int[Tensor, "batch_size ... seq_len"] | None = None,
    ) -> Float[Tensor, "batch_size ... seq_len vocab_size"]:
        """
        token_ids: (batch_size, ..., seq_len)
        token_positions: (batch_size, ..., seq_len) or None
        FLOPS_BLOCK: num_layers * (4 * seq_len^2 * d_model + 8 * d_model^2 * seq_len + 6 * d_model * d_ff * seq_len) * ... (batch size and other dimensions)
        FLOPS_OUT: 2 * d_model * vocab_size * seq_len * ... (batch size and other dimensions)
        Total FLOPS: FLOPS_BLOCK + FLOPS_OUT
        """
        x = self.token_embedding(token_ids)
        for layer in self.layers:
            x = layer(x, token_positions=token_positions)
        x = self.rms_norm(x)
        logits = self.lm_head(x)
        return logits

    def extra_repr(self) -> str:
        return f"token_embedding={self.token_embedding}, layers={self.layers}, rms_norm={self.rms_norm}, output_linear={self.lm_head}, softmax={self.softmax}"


def calc_num_params(vocab_size: int, num_layers: int, d_model: int, num_heads: int, d_ff: int) -> int:
    embedding_params = vocab_size * d_model
    mha_params = 4 * d_model * d_model * num_layers
    ffn_params = 3 * d_model * d_ff * num_layers
    rmsnorm_params = 2 * d_model * num_layers
    block_params = mha_params + ffn_params + rmsnorm_params
    rmsnorm_params = d_model
    lmhead_params = d_model * vocab_size
    total_params = embedding_params + block_params + rmsnorm_params + lmhead_params
    print(f"Embedding params: {embedding_params:,} Percent: {embedding_params/total_params:.2%}")
    print(f"MHA params: {mha_params:,} Percent: {mha_params/total_params:.2%}")
    print(f"FFN params: {ffn_params:,} Percent: {ffn_params/total_params:.2%}")
    print(f"rmsnorm params: {rmsnorm_params:,} Percent: {rmsnorm_params/total_params:.2%}")
    print(f"Block params: {block_params:,} Percent: {block_params/total_params:.2%}")
    print(f"RMSNorm params: {rmsnorm_params:,} Percent: {rmsnorm_params/total_params:.2%}")
    print(f"LM Head params: {lmhead_params:,} Percent: {lmhead_params/total_params:.2%}")
    print(f"Total params: {total_params:,} ({total_params * 4 / (1024**3):.2f} GB)")
    return total_params


def calc_flops(seq_len: int, batch_size: int, num_layers: int, d_model: int, d_ff: int, vocab_size: int) -> int:
    atten_flops = num_layers * (4 * seq_len * seq_len * d_model + 8 * d_model * d_model * seq_len) * batch_size
    ffn_flops = num_layers * (6 * d_model * d_ff * seq_len) * batch_size
    flops_block = atten_flops + ffn_flops
    flops_out = 2 * d_model * vocab_size * seq_len * batch_size
    total_flops = flops_block + flops_out
    print(f"FLOPS for attention: {atten_flops / 1e12:.2f} TFLOPS {atten_flops:,} FLOPS Percent: {atten_flops/total_flops:.2%}")
    print(f"FLOPS for FFN: {ffn_flops / 1e12:.2f} TFLOPS {ffn_flops:,} FLOPS, Percent: {ffn_flops/total_flops:.2%}")
    print(f"FLOPS per Transformer block: {flops_block / 1e12:.2f} TFLOPS {flops_block:,} FLOPS, Percent: {flops_block/total_flops:.2%}")
    print(f"FLOPS for output layer: {flops_out / 1e12:.2f} TFLOPS {flops_out:,} FLOPS, Percent: {flops_out/total_flops:.2%}")
    print(f"Total FLOPS: {total_flops / 1e12:.2f} TFLOPS {total_flops:,} FLOPS")
    return total_flops


if __name__ == "__main__":
    from rich import print

    print("--- GPT-XL-like Model Configuration ---")
    vocab_size = 50_257
    num_layers = 48
    d_model = 1600
    num_heads = 24
    d_ff = 6400
    max_seq_len = 1024
    theta = 100000.0
    # model = TransformerLM(
    #     vocab_size=vocab_size,
    #     num_layers=num_layers,
    #     d_model=d_model,
    #     num_heads=num_heads,
    #     d_ff=d_ff,
    #     max_seq_len=max_seq_len,
    #     theta=theta,
    # )
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Model parameters: {total_params:,}")
    print("--- Estimating Parameters ---")
    estimate_params = calc_num_params(vocab_size, num_layers, d_model, num_heads, d_ff)

    print("\n--- Estimating FLOPS ---")
    flops = calc_flops(
        seq_len=max_seq_len, batch_size=1, num_layers=num_layers, d_model=d_model, d_ff=d_ff, vocab_size=vocab_size
    )
    print("-" * 80)

    print("--- GPT-2 small Model Configuration ---")
    num_layers = 12
    d_model = 768
    num_heads = 12
    print("--- Estimating Parameters ---")
    estimate_params = calc_num_params(vocab_size, num_layers, d_model, num_heads, d_ff)

    print("\n--- Estimating FLOPS ---")
    flops = calc_flops(
        seq_len=max_seq_len, batch_size=1, num_layers=num_layers, d_model=d_model, d_ff=d_ff, vocab_size=vocab_size
    )
    print("-" * 80)

    print("--- GPT-2 medium Model Configuration ---")
    num_layers = 24
    d_model = 1024
    num_heads = 16
    print("--- Estimating Parameters ---")
    estimate_params = calc_num_params(vocab_size, num_layers, d_model, num_heads, d_ff)

    print("\n--- Estimating FLOPS ---")
    flops = calc_flops(
        seq_len=max_seq_len, batch_size=1, num_layers=num_layers, d_model=d_model, d_ff=d_ff, vocab_size=vocab_size
    )
    print("-" * 80)

    print("--- GPT-2 large Model Configuration ---")
    num_layers = 36
    d_model = 1280
    num_heads = 20
    print("--- Estimating Parameters ---")
    estimate_params = calc_num_params(vocab_size, num_layers, d_model, num_heads, d_ff)

    print("\n--- Estimating FLOPS ---")
    flops = calc_flops(
        seq_len=max_seq_len, batch_size=1, num_layers=num_layers, d_model=d_model, d_ff=d_ff, vocab_size=vocab_size
    )
    print("-" * 80)


    max_seq_len = 16_384
    print("--- Estimating FLOPS for 16K context ---")
    flops = calc_flops(
        seq_len=max_seq_len, batch_size=1, num_layers=num_layers, d_model=d_model, d_ff=d_ff, vocab_size=vocab_size
    )
    print("-" * 80)