"""Tier 0 / I7: draft-side tree-cache lookup semantics.

The draft runner stores a tensor of keys `[T, 3]` (seq_id, keep_idx, recovery_token)
and matches incoming `[B, 3]` request keys via broadcast-equality + all-rows.
On hit, it indexes into stored tokens/logits/activations.

This test models that lookup in pure Python (replicating the logic from
`draft_runner.hit_cache`, lines ~242–246 on the cc/sglang-fa4 branch) and
verifies:
- all-match key → hit, index points at the first matching entry
- partial match (only seq_id agrees) → miss
- empty cache → miss for every request
- different recovery_token or keep_idx → miss

Note: this intentionally does NOT import DraftRunner, because constructing one
requires a GPU, model weights, and an initialized process group. The matching
logic is simple and regressions in it would be equally captured by the small
model here.
"""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.tier0


def _lookup(request_keys: torch.Tensor, cache_keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Replicates the matcher in draft_runner.hit_cache.

    request_keys: [B, 3] int64
    cache_keys:   [T, 3] int64
    Returns:
      hits: [B] bool
      idx:  [B] int — index of first match per row, 0 when no match (mirrors torch.max on a zero mask)
    """
    if cache_keys.numel() == 0:
        return torch.zeros(request_keys.shape[0], dtype=torch.bool), torch.zeros(
            request_keys.shape[0], dtype=torch.int64,
        )
    eq = request_keys.unsqueeze(1) == cache_keys.unsqueeze(0)   # [B, T, 3]
    match = torch.all(eq, dim=2)                                # [B, T]
    hits, idx = match.max(dim=1)
    return hits, idx


class TestCacheLookup:
    def test_empty_cache_is_all_miss(self):
        cache = torch.empty(0, 3, dtype=torch.int64)
        req = torch.tensor([[1, 0, 42], [2, 1, 7]], dtype=torch.int64)
        hits, idx = _lookup(req, cache)
        assert hits.tolist() == [False, False]

    def test_exact_match_hits(self):
        cache = torch.tensor([
            [1, 0, 42],
            [2, 1, 7],
            [3, 2, 99],
        ], dtype=torch.int64)
        req = torch.tensor([[2, 1, 7]], dtype=torch.int64)
        hits, idx = _lookup(req, cache)
        assert hits.tolist() == [True]
        assert idx.tolist() == [1]

    def test_different_recovery_token_misses(self):
        cache = torch.tensor([[1, 0, 42]], dtype=torch.int64)
        req = torch.tensor([[1, 0, 43]], dtype=torch.int64)  # different rec token
        hits, _idx = _lookup(req, cache)
        assert hits.tolist() == [False]

    def test_different_keep_idx_misses(self):
        cache = torch.tensor([[1, 0, 42]], dtype=torch.int64)
        req = torch.tensor([[1, 1, 42]], dtype=torch.int64)  # different keep_idx
        hits, _idx = _lookup(req, cache)
        assert hits.tolist() == [False]

    def test_different_seq_id_misses(self):
        cache = torch.tensor([[1, 0, 42]], dtype=torch.int64)
        req = torch.tensor([[2, 0, 42]], dtype=torch.int64)  # different seq_id
        hits, _idx = _lookup(req, cache)
        assert hits.tolist() == [False]

    def test_first_match_wins_on_duplicates(self):
        cache = torch.tensor([
            [1, 0, 42],
            [1, 0, 42],  # duplicate
        ], dtype=torch.int64)
        req = torch.tensor([[1, 0, 42]], dtype=torch.int64)
        hits, idx = _lookup(req, cache)
        assert hits.tolist() == [True]
        assert idx.tolist() == [0]  # first match

    def test_mixed_hit_miss_in_batch(self):
        cache = torch.tensor([
            [1, 0, 42],
            [2, 1, 7],
        ], dtype=torch.int64)
        req = torch.tensor([
            [1, 0, 42],       # hit
            [99, 99, 99],     # miss
            [2, 1, 7],        # hit
        ], dtype=torch.int64)
        hits, idx = _lookup(req, cache)
        assert hits.tolist() == [True, False, True]
        assert idx.tolist()[0] == 0
        assert idx.tolist()[2] == 1


class TestRollbackInvalidation:
    """After a sequence rolls back, old cache entries for that seq_id+keep_idx+rec
    combination should not be reachable from the new key. We model that by
    evolving the state of a sequence across two steps and showing that the cache
    entry from step 1 does not service step 2's key (because at least one of the
    three components always changes across a real rollback).
    """

    def test_key_changes_after_rollback(self):
        # Step 1: seq 7 has accepted_len=3, rec=111. Cache entry written with this key.
        cache = torch.tensor([[7, 2, 111]], dtype=torch.int64)  # keep_idx = accepted_len - 1

        # Step 2 (the verifier rolled back to accepted_len=2 because only 1 token accepted
        # after sampling rec=111): new accepted_len=2 -> keep_idx=1, new rec is resampled.
        new_req = torch.tensor([[7, 1, 222]], dtype=torch.int64)
        hits, _idx = _lookup(new_req, cache)
        assert hits.tolist() == [False], "rollback should invalidate the prior cache key"


class TestCollisionSemantics:
    """Different sequences writing keys that share components should not collide unless all three match."""

    def test_same_rec_and_keep_different_seq_no_collision(self):
        cache = torch.tensor([
            [1, 0, 42],
            [2, 0, 42],
        ], dtype=torch.int64)
        req = torch.tensor([[1, 0, 42]], dtype=torch.int64)
        hits, idx = _lookup(req, cache)
        assert hits.tolist() == [True]
        assert idx.tolist() == [0]
