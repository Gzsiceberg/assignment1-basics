import torch
from torch import nn, Tensor
from einops import rearrange, einsum
import numpy as np
from jaxtyping import Float, Int, jaxtyped, Bool
from beartype import beartype as typechecker
from cs336_basics.model.linear import Linear
from cs336_basics.model.rope import RoPE


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    @jaxtyped(typechecker=typechecker)
    def forward(
        self,
        query: Float[Tensor, "batch ... n d_k"],
        key: Float[Tensor, "batch ... m d_k"],
        value: Float[Tensor, "batch ... m d_v"],
        mask: Bool[Tensor, "batch ... n m"] | None = None,
    ) -> Float[Tensor, "batch ... n d_v"]:
        """
        query: (batch, ..., n, d_k)
        key: (batch, ..., m, d_k)
        value: (batch, ..., m, d_v)
        mask: (batch, ..., n, m) or None
        FLOPS: 2 * n * m * (d_k + d_v) * ... (batch size and other dimensions)
        Output: (batch, ..., n, d_v)
        """
        d_k = query.shape[-1]
        assert key.shape[-1] == query.shape[-1], f"Key shape {key.shape} must match query shape {query.shape}"
        assert value.shape[-2] == key.shape[-2], f"Value shape {value.shape} must match key shape {key.shape}"

        attention_scores = einsum(query, key, "batch ... n d_k, batch ... m d_k -> batch ... n m")
        attention_scores = attention_scores / np.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))
        attention_weights = self.softmax(attention_scores)
        value = einsum(attention_weights, value, "batch ... n m, batch ... m d_v -> batch ... n d_v")
        return value

    def extra_repr(self) -> str:
        return "Scaled Dot-Product Attention"


class MultiHeadAttention(nn.Module):
    num_heads: int
    d_k: int
    d_v: int
    rope: RoPE | None

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RoPE | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        """
        Parameters: 4 * d_model^2 (Linear layers)
        """
        d_k = d_v = d_model // num_heads
        self.rope = rope
        self.wq = Linear(d_model, num_heads * d_k, device=device, dtype=dtype)
        self.wk = Linear(d_model, num_heads * d_k, device=device, dtype=dtype)
        self.wv = Linear(d_model, num_heads * d_v, device=device, dtype=dtype)
        self.wo = Linear(num_heads * d_v, d_model, device=device, dtype=dtype)
        self.attention = Attention()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v

        self.register_buffer("causal_mask", None, persistent=False)

    @jaxtyped(typechecker=typechecker)
    def _get_causal_mask(self, seq_len: int, device: torch.device) -> Bool[Tensor, "seq_len seq_len"]:
        if self.causal_mask is None or self.causal_mask.shape[0] < seq_len:
            max_seq_len = max(seq_len, 2048)
            self.causal_mask = torch.ones(max_seq_len, max_seq_len, device=device, dtype=torch.bool).tril(0)
        return self.causal_mask[:seq_len, :seq_len]

    @jaxtyped(typechecker=typechecker)
    def _create_mask(
        self, query_shape: tuple, seq_len: int, device: torch.device, token_positions: Int[Tensor, "... seq_len"] | None
    ) -> Bool[Tensor, "... seq_len seq_len"]:
        batch_dims = query_shape[:-2]

        if token_positions is not None:
            position_query = rearrange(token_positions, "... n -> ... n 1")
            position_key = rearrange(token_positions, "... m -> ... 1 m")
            casual_mask = position_query >= position_key
        else:
            casual_mask = self._get_causal_mask(seq_len, device)
        casual_mask = casual_mask.expand(*batch_dims, seq_len, seq_len)
        return casual_mask

    @jaxtyped(typechecker=typechecker)
    def forward(
        self, x: Float[Tensor, "... seq_len d_model"], token_positions: Int[Tensor, "... seq_len"] | None = None
    ) -> Float[Tensor, "... d_model"]:
        """
        x: (..., seq_len, d_model)
        token_positions: (..., seq_len) or None
        FLOPS_ATTN: 4 * seq_len^2 * d_model * ... (batch size and other dimensions)
        FLOPS_wq_wk_wv_wo: 4 * 2 * d_model * d_model * seq_len * ... (batch size and other dimensions)
        Total FLOPS: FLOPS_ATTN + FLOPS_wq_wk_wv_wo
        Parameters: 4 * d_model^2 (Linear layers)
        Output: (..., seq_len, d_model)
        """
        seq_len = x.shape[-2]
        query = self.wq(x)
        key = self.wk(x)
        value = self.wv(x)

        query = rearrange(
            query, "... seq_len (heads d_k) -> ... heads seq_len d_k", heads=self.num_heads, seq_len=seq_len
        )
        if self.rope is not None:
            query = self.rope(query, token_positions)

        key = rearrange(key, "... seq_len (heads d_k) -> ... heads seq_len d_k", heads=self.num_heads, seq_len=seq_len)
        if self.rope is not None:
            key = self.rope(key, token_positions)

        value = rearrange(
            value, "... seq_len (heads d_v) -> ... heads seq_len d_v", heads=self.num_heads, seq_len=seq_len
        )
        mask = self._create_mask(x.shape, seq_len, x.device, token_positions)

        out = self.attention(query, key, value, mask=mask)
        out = rearrange(out, "... heads seq_len d_v -> ... seq_len (heads d_v)")
        out = self.wo(out)
        return out

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, num_heads={self.num_heads}, d_k={self.d_k}, d_v={self.d_v}"


if __name__ == "__main__":
    rope = None
    multiHead = MultiHeadAttention(d_model=512, num_heads=8, rope=rope)
    x = torch.randn(10, 20, 512)
    out = multiHead(x)
    print(out.shape)  # Expected output shape: (10, 20, 512)
