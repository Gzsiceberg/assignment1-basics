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
from cs336_basics.model.silu import SiLU
from cs336_basics.model.swiglu import SwiGLU


class TransformerBlock(nn.Module):
    multi_head_attention: MultiHeadAttention
    rms_norm1: RMSNorm | None
    rms_norm2: RMSNorm | None
    ffn: nn.Module
    layernorm_type: str

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
        ffn_type: str = "swiglu",
        layernorm_type: str = "prenorm",
    ) -> None:
        super().__init__()
        """
        Parameters: 4 * d_model^2 (MHA) + 3 * d_model * d_ff (SwiGLU) + 2 * d_model (RMSNorms)
        """
        d_k: int = d_model // num_heads
        self.layernorm_type = layernorm_type
        if rope is not None:
            assert rope.d_k == d_k, f"RoPE d_k {rope.d_k} must match model d_model/num_heads {d_k}"
        else:
            rope = RoPE(theta=theta, d_k=d_k, max_seq_len=max_seq_len, device=device)
        if layernorm_type:
            self.rms_norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        else:
            self.rms_norm1 = None
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads, rope=rope, device=device, dtype=dtype)

        if layernorm_type:
            self.rms_norm2 = RMSNorm(d_model, device=device, dtype=dtype)
        else:
            self.rms_norm2 = None
        if ffn_type == "swiglu":
            self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        elif ffn_type == "silu":
            self.ffn = SiLU(d_model, d_ff, device=device, dtype=dtype)

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
        if self.layernorm_type == "prenorm" and self.rms_norm1 is not None:
            norm_x = self.rms_norm1(x)
        else:
            norm_x = x
        atten_x = self.multi_head_attention(norm_x, token_positions=token_positions)
        y0 = x + atten_x
        if self.layernorm_type == "postnorm" and self.rms_norm1 is not None:
            y0 = self.rms_norm1(y0)
        
        if self.layernorm_type == "prenorm" and self.rms_norm2 is not None:
            norm_y0 = self.rms_norm2(y0) if self.rms_norm2 is not None else y0
        else:
            norm_y0 = y0
        ffn_y = self.ffn(norm_y0)
        y = y0 + ffn_y
        if self.layernorm_type == "postnorm" and self.rms_norm2 is not None:
            y = self.rms_norm2(y)
        return y

    def extra_repr(self) -> str:
        return f"rms_norm1={self.rms_norm1}, multi_head_attention={self.multi_head_attention}, rms_norm2={self.rms_norm2}, ffn={self.ffn}"


class TransformerLM(nn.Module):
    vocab_size: int
    max_seq_len: int
    rms_norm: RMSNorm

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
        ffn_type: str = "swiglu",
        layernorm_type: str = "prenorm",
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
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
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
                    ffn_type=ffn_type,
                    layernorm_type=layernorm_type,
                )
                for _ in range(num_layers)
            ]
        )
        self.rms_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    @jaxtyped(typechecker=typechecker)
    def forward(
        self,
        token_ids: Int[Tensor, "batch_size seq_len"],
        token_positions: Int[Tensor, "batch_size seq_len"] | None = None,
    ) -> Float[Tensor, "batch_size seq_len vocab_size"]:
        """
        token_ids: (batch_size, seq_len)
        token_positions: (batch_size, seq_len) or None
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




