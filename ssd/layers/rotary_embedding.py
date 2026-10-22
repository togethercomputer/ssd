from functools import lru_cache
import torch
from torch import nn


def rope_theta_from_config(config, default: float) -> float:
    """Return rope_theta from an HF config (transformers v4 attr or v5 rope_parameters)."""
    rope_parameters = getattr(config, "rope_parameters", None)
    if isinstance(rope_parameters, dict) and "rope_theta" in rope_parameters:
        return rope_parameters["rope_theta"]
    return getattr(config, "rope_theta", default)


def set_rope_theta_on_config(config, rope_theta: float) -> None:
    """Set rope_theta on an HF config so both v4 and v5 readers see it."""
    config.rope_theta = rope_theta
    rope_parameters = getattr(config, "rope_parameters", None)
    if isinstance(rope_parameters, dict):
        rope_parameters["rope_theta"] = rope_theta


def rope_scaling_from_config(config) -> dict | None:
    """Return rope_scaling, mapping transformers v5's default rope_parameters dict to None."""
    rope_scaling = getattr(config, "rope_scaling", None)
    if isinstance(rope_scaling, dict):
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
        if rope_type == "default":
            return None
    return rope_scaling


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        # print(f"head_size: {head_size}, rotary_dim: {rotary_dim}, max_position_embeddings: {max_position_embeddings}, base: {base}")
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = positions.size(0)
        cos_sin = self.cos_sin_cache[positions]
        # cos_sin = torch.ones_like(cos_sin)
        cos, sin = cos_sin.chunk(2, dim=-1)
        query_shape = query.shape

        num_q_heads = query.shape[-1] // self.head_size
        query = query.view(num_tokens, num_q_heads, self.head_size)
        query = apply_rotary_emb(query, cos, sin).view(query_shape)
        key_shape = key.shape
        num_k_heads = key.shape[-1] // self.head_size
        key = key.view(num_tokens, num_k_heads, self.head_size)
        key = apply_rotary_emb(key, cos, sin).view(key_shape)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    assert rope_scaling is None
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
