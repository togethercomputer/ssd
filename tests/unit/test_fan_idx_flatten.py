"""Tier 0: lock down the fan_idx flattening pattern used in draft_runner.

Three call sites (draft_runner.py:574, :759, :1021) build a per-token
"fan index" tensor from per-seq cache_hits via:

    torch.cat([FAN_IDX_HIT if hit else FAN_IDX_MISS for hit in cache_hits_list])

The Python list-comprehension is a per-call CPU pass that hurts the async
decode_tree hot path. The optimization is to replace it with a vectorized
expression like:

    torch.where(cache_hits[:, None].bool(),
                FAN_IDX_HIT[None, :], FAN_IDX_MISS[None, :]).reshape(-1)

These tests pin equivalence so the rewrite can be mechanical.
"""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.tier0


def _fan_idx_hit_miss(K: int, fan_out: list[int], fan_out_miss: list[int]):
    """Mirrors DraftRunner._init_prealloc_buffers for a given fan-out."""
    device = torch.device("cpu")
    hit = torch.arange(K + 1, device=device, dtype=torch.int64).repeat_interleave(
        torch.tensor(fan_out, dtype=torch.int64)
    )
    miss = torch.arange(K + 1, device=device, dtype=torch.int64).repeat_interleave(
        torch.tensor(fan_out_miss, dtype=torch.int64)
    )
    assert hit.shape == miss.shape  # MQ_LEN = sum(fan_out) = sum(fan_out_miss)
    return hit, miss


def _loop_flatten(fan_hit, fan_miss, cache_hits: torch.Tensor) -> torch.Tensor:
    """The existing pattern — Python comprehension + torch.cat."""
    return torch.cat([fan_hit if bool(h) else fan_miss for h in cache_hits.tolist()])


def _vectorized_flatten(fan_hit, fan_miss, cache_hits: torch.Tensor) -> torch.Tensor:
    """Target vectorized equivalent — no Python-side iteration over cache_hits."""
    B = cache_hits.shape[0]
    hits_bool = cache_hits.to(torch.bool).view(B, 1)
    out = torch.where(hits_bool, fan_hit.view(1, -1), fan_miss.view(1, -1))
    return out.reshape(-1)


@pytest.mark.parametrize(
    "K,fan_out,fan_out_miss",
    [
        (2, [2, 2, 2], [2, 2, 2]),            # uniform
        (2, [1, 3, 3], [7, 0, 0]),            # jit-style mismatch
        (3, [2, 2, 2, 2], [8, 0, 0, 0]),      # K=3
        (1, [1, 4], [1, 4]),                  # small
    ],
)
@pytest.mark.parametrize("B", [1, 2, 4, 8, 16])
@pytest.mark.parametrize(
    "hit_pattern",
    ["all_hit", "all_miss", "alternating", "first_hit_only", "last_miss_only"],
)
def test_vectorized_equals_loop(K, fan_out, fan_out_miss, B, hit_pattern):
    fan_hit, fan_miss = _fan_idx_hit_miss(K, fan_out, fan_out_miss)

    if hit_pattern == "all_hit":
        hits = torch.ones(B, dtype=torch.int64)
    elif hit_pattern == "all_miss":
        hits = torch.zeros(B, dtype=torch.int64)
    elif hit_pattern == "alternating":
        hits = torch.tensor([b % 2 for b in range(B)], dtype=torch.int64)
    elif hit_pattern == "first_hit_only":
        hits = torch.zeros(B, dtype=torch.int64)
        hits[0] = 1
    elif hit_pattern == "last_miss_only":
        hits = torch.ones(B, dtype=torch.int64)
        hits[-1] = 0

    loop = _loop_flatten(fan_hit, fan_miss, hits)
    vec = _vectorized_flatten(fan_hit, fan_miss, hits)

    MQ_LEN = sum(fan_out)
    assert loop.shape == (B * MQ_LEN,)
    assert vec.shape == loop.shape
    assert torch.equal(vec, loop), (
        f"vectorized != loop for K={K} B={B} pattern={hit_pattern}"
    )


def test_accepts_float_cache_hits():
    """cache_hits arrives as float at some call sites — check the vectorized form handles it."""
    K = 2
    fan_hit, fan_miss = _fan_idx_hit_miss(K, [2, 2, 2], [2, 2, 2])
    B = 4
    hits_float = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    hits_int = hits_float.to(torch.int64)
    vec_float = _vectorized_flatten(fan_hit, fan_miss, hits_float)
    loop_int = _loop_flatten(fan_hit, fan_miss, hits_int)
    assert torch.equal(vec_float, loop_int)
