"""Tier 0: fork-token (cache-candidate) selection semantics.

`get_forked_recovery_tokens_from_logits` chooses, per sequence and per
accepted-count position k in [0, K], the fork candidates (future recovery
tokens) whose continuations get precomputed into the speculation cache.
Contract under test:

1. At positions 0..K-1 the already-speculated token at that depth
   (returned_tokens[:, j+1], i.e. spec_j) is EXCLUDED — if the target rejects
   at depth j, the recovery token cannot equal spec_j (greedy: it differs by
   definition of rejection; sampling: a token with residual mass p>q is never
   rejected). At position K (all K accepted, bonus position) nothing is
   excluded.
2. Hit rows take fan_out_list[k] candidates per position; miss rows take
   fan_out_list_miss[k] (fast mode: all MQ_LEN candidates at k=0).
3. Output is depth-major per row ([pos0 candidates..., pos1 candidates...]),
   matching the k-index layout `arange(K+1).repeat_interleave(fan_out)` used
   by _populate_tree_cache — the cache keys are only correct if these two
   orderings agree.
4. Rows are independent (batch permutation equivariance).

Pure tensor function — no GPU needed.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from ssd.utils.async_helpers.async_spec_helpers import (
    get_forked_recovery_tokens_from_logits,
)

pytestmark = pytest.mark.tier0

V = 97  # prime-ish, > any fanout * (K+1)


def _mk_cfg(K: int, F: int, jit: bool = False):
    fan_out_list = [F] * (K + 1)
    mq = sum(fan_out_list)
    fan_out_list_miss = fan_out_list if jit else [mq] + [0] * K
    return SimpleNamespace(
        speculate_k=K,
        fan_out_list=fan_out_list,
        fan_out_list_miss=fan_out_list_miss,
    )


def _distinct_logits(B: int, K: int, seed: int = 0) -> torch.Tensor:
    """Logits with no ties anywhere: a strictly decreasing base ramp over the
    vocab, randomly permuted per (b, position) so top-k sets differ."""
    g = torch.Generator().manual_seed(seed)
    base = torch.linspace(10.0, -10.0, V)
    out = torch.empty(B, K + 1, V)
    for b in range(B):
        for j in range(K + 1):
            perm = torch.randperm(V, generator=g)
            out[b, j] = base[perm]
    return out


def _topk_excluding(row: torch.Tensor, n: int, excluded: int | None) -> list[int]:
    order = torch.argsort(row, descending=True).tolist()
    if excluded is not None:
        order = [t for t in order if t != excluded]
    return order[:n]


def test_exclusion_at_depths_0_to_km1_and_not_at_bonus():
    K, F, B = 3, 2, 2
    cfg = _mk_cfg(K, F)
    logits = _distinct_logits(B, K, seed=1)
    # returned_tokens[:, j+1] := argmax of position j, so exclusion is load-bearing
    returned = torch.zeros(B, K + 1, dtype=torch.int64)
    for b in range(B):
        returned[b, 0] = 11  # recovery token: irrelevant to selection
        for j in range(K):
            returned[b, j + 1] = logits[b, j].argmax()
    cache_hits = torch.ones(B, dtype=torch.int64)

    idxs = get_forked_recovery_tokens_from_logits(cfg, logits.clone(), cache_hits, returned, tokenizer=None)
    assert idxs.shape == (B, sum(cfg.fan_out_list))

    for b in range(B):
        chunks = idxs[b].split([F] * (K + 1))
        for j in range(K):  # exclusion positions
            expected = _topk_excluding(logits[b, j], F, int(returned[b, j + 1]))
            assert chunks[j].tolist() == expected, f"pos {j}: spec token not excluded or wrong top-k"
            assert int(returned[b, j + 1]) not in chunks[j].tolist()
        # bonus position K: no exclusion — the plain top-F, even though it
        # coincides with returned argmax-successor tokens
        expected_k = _topk_excluding(logits[b, K], F, None)
        assert chunks[K].tolist() == expected_k, "bonus position must not exclude anything"


def test_miss_rows_reallocate_all_fanout_to_k0():
    K, F = 2, 3
    cfg = _mk_cfg(K, F)  # miss list = [9, 0, 0]
    mq = sum(cfg.fan_out_list)
    B = 4
    logits = _distinct_logits(B, K, seed=2)
    returned = torch.zeros(B, K + 1, dtype=torch.int64)
    for b in range(B):
        for j in range(K):
            returned[b, j + 1] = logits[b, j].argmax()
    cache_hits = torch.tensor([1, 0, 1, 0], dtype=torch.int64)

    idxs = get_forked_recovery_tokens_from_logits(cfg, logits.clone(), cache_hits, returned, tokenizer=None)
    assert idxs.shape == (B, mq)

    for b in range(B):
        if cache_hits[b]:
            chunks = idxs[b].split([F] * (K + 1))
            for j in range(K + 1):
                excl = int(returned[b, j + 1]) if j < K else None
                assert chunks[j].tolist() == _topk_excluding(logits[b, j], F, excl)
        else:
            # all MQ_LEN candidates come from position 0 (post-exclusion)
            expected = _topk_excluding(logits[b, 0], mq, int(returned[b, 1]))
            assert idxs[b].tolist() == expected, f"miss row {b} should take top-{mq} at k=0"


def test_row_permutation_equivariance():
    K, F, B = 2, 2, 5
    cfg = _mk_cfg(K, F)
    logits = _distinct_logits(B, K, seed=3)
    returned = torch.randint(0, V, (B, K + 1), generator=torch.Generator().manual_seed(4))
    cache_hits = torch.tensor([1, 0, 1, 1, 0], dtype=torch.int64)

    base = get_forked_recovery_tokens_from_logits(cfg, logits.clone(), cache_hits, returned, tokenizer=None)
    perm = torch.tensor([3, 0, 4, 1, 2])
    permuted = get_forked_recovery_tokens_from_logits(
        cfg, logits[perm].clone(), cache_hits[perm], returned[perm], tokenizer=None
    )
    assert torch.equal(permuted, base[perm]), "rows must be independent"


def test_depth_major_ordering_matches_fan_idx_layout():
    """The flat candidate ordering must agree with the k-index pattern
    `arange(K+1).repeat_interleave(fan_out)` used when keys are written into
    the tree cache (draft_runner._populate_tree_cache via _fan_idx_hit/miss).
    We verify by construction: give position j a unique token range so the
    position of origin of every candidate is recoverable from its value."""
    K, F, B = 2, 2, 1
    cfg = _mk_cfg(K, F)
    logits = torch.full((B, K + 1, V), -100.0)
    for j in range(K + 1):
        # position j's top tokens live in [10*j, 10*j + 5)
        for r in range(5):
            logits[0, j, 10 * j + r] = 50.0 - r
    returned = torch.zeros(B, K + 1, dtype=torch.int64)  # token 0 excluded at pos<K (harmless)
    cache_hits = torch.ones(B, dtype=torch.int64)

    idxs = get_forked_recovery_tokens_from_logits(cfg, logits.clone(), cache_hits, returned, tokenizer=None)
    origin = (idxs[0] // 10).tolist()
    expected_k_layout = torch.arange(K + 1).repeat_interleave(torch.tensor(cfg.fan_out_list)).tolist()
    assert origin == expected_k_layout, (
        f"candidate ordering {origin} disagrees with cache-key k layout {expected_k_layout}"
    )
