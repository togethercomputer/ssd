"""Tier 0 / I10: BlockManager semantics.

Exercises allocate / deallocate / prefix caching / refcount / may_append /
draft-vs-target independence.
"""
from __future__ import annotations

import pytest

from ssd.engine.block_manager import Block, BlockManager
from ssd.engine.sequence import Sequence
from ssd.sampling_params import SamplingParams

pytestmark = pytest.mark.tier0


# Block_size is a class-var on Sequence set by the engine; we set it for tests.
BLOCK_SIZE = 4


def _seq(token_ids: list[int]) -> Sequence:
    Sequence.block_size = BLOCK_SIZE
    return Sequence(token_ids, SamplingParams())


def _fresh_bm(num_blocks: int = 16, is_draft: bool = False, max_model_len: int = 4096) -> BlockManager:
    return BlockManager(
        num_blocks=num_blocks,
        block_size=BLOCK_SIZE,
        is_draft=is_draft,
        max_model_len=max_model_len,
    )


# ---------------------------------------------------------------------------
# Allocation invariants
# ---------------------------------------------------------------------------
class TestAllocate:
    def test_allocate_fills_block_table(self):
        bm = _fresh_bm()
        s = _seq([1, 2, 3, 4, 5, 6, 7])  # 7 tokens → 2 blocks (one full, one partial)
        assert s.num_blocks == 2
        bm.allocate(s)
        assert len(s.block_table) == 2
        # Complete block finalized (hash set), incomplete block not finalized
        b0 = bm.blocks[s.block_table[0]]
        b1 = bm.blocks[s.block_table[1]]
        assert b0.hash != -1
        assert b1.hash == -1
        assert b0.ref_count == 1
        assert b1.ref_count == 1
        assert b0.block_id not in bm.free_block_ids
        assert b1.block_id not in bm.free_block_ids

    def test_shared_prefix_hits_cache(self):
        """Second sequence with same first-block prefix reuses the same block."""
        bm = _fresh_bm()
        s1 = _seq([10, 11, 12, 13, 14, 15, 16, 17])  # 2 full blocks
        s2 = _seq([10, 11, 12, 13, 99, 98, 97])       # first block matches s1; second differs
        bm.allocate(s1)
        bm.allocate(s2)

        assert s1.block_table[0] == s2.block_table[0], "shared first block not reused"
        assert s1.block_table[1] != s2.block_table[1], "different second block collided"
        # cached_tokens reflects the reuse on s2
        assert s2.num_cached_tokens == BLOCK_SIZE
        # Shared block has ref_count == 2
        assert bm.blocks[s1.block_table[0]].ref_count == 2

    def test_incomplete_last_block_is_not_hashed(self):
        bm = _fresh_bm()
        s = _seq([1, 2, 3])  # less than a block
        bm.allocate(s)
        assert len(s.block_table) == 1
        assert bm.blocks[s.block_table[0]].hash == -1
        assert not any(h == bm.blocks[s.block_table[0]].hash for h in bm.hash_to_block_id)

    def test_can_allocate_respects_free_pool(self):
        bm = _fresh_bm(num_blocks=2)
        s_small = _seq([1, 2, 3])                        # 1 block
        s_big = _seq([1] * (BLOCK_SIZE * 3))             # 3 blocks
        assert bm.can_allocate(s_small) is True
        assert bm.can_allocate(s_big) is False


# ---------------------------------------------------------------------------
# Deallocation / refcount
# ---------------------------------------------------------------------------
class TestDeallocate:
    def test_deallocate_returns_block_to_free_pool(self):
        bm = _fresh_bm()
        s = _seq([1, 2, 3, 4, 5])
        bm.allocate(s)
        freed_ids = list(s.block_table)
        free_before = len(bm.free_block_ids)
        bm.deallocate(s)
        assert s.block_table == []
        assert len(bm.free_block_ids) == free_before + len(freed_ids)
        assert s.num_cached_tokens == 0
        for bid in freed_ids:
            assert bm.blocks[bid].ref_count == 0

    def test_shared_block_stays_until_refcount_zero(self):
        bm = _fresh_bm()
        s1 = _seq([1, 2, 3, 4, 5])    # 2 blocks, shares first with s2
        s2 = _seq([1, 2, 3, 4, 9])
        bm.allocate(s1)
        bm.allocate(s2)
        shared = s1.block_table[0]
        assert bm.blocks[shared].ref_count == 2

        bm.deallocate(s1)
        assert bm.blocks[shared].ref_count == 1
        assert shared not in bm.free_block_ids  # still held by s2

        bm.deallocate(s2)
        assert bm.blocks[shared].ref_count == 0
        assert shared in bm.free_block_ids

    def test_deallocate_removes_hash_mapping(self):
        bm = _fresh_bm()
        s = _seq([1, 2, 3, 4, 5, 6, 7, 8])  # 2 full blocks, both hashed
        bm.allocate(s)
        hashes = [bm.blocks[b].hash for b in s.block_table]
        assert all(h in bm.hash_to_block_id for h in hashes)
        bm.deallocate(s)
        assert not any(h in bm.hash_to_block_id for h in hashes)


# ---------------------------------------------------------------------------
# may_append / lookahead
# ---------------------------------------------------------------------------
class TestMayAppend:
    def test_may_append_allocates_more_blocks(self):
        bm = _fresh_bm()
        s = _seq([1, 2, 3])  # 1 block
        bm.allocate(s)
        # Simulate appending tokens so num_tokens grows
        s.append_token(4)
        s.append_token(5)  # now 5 tokens → needs 2 blocks
        assert s.num_blocks == 2
        bm.may_append(s, lookahead_num_tokens=0)
        assert len(s.block_table) == 2

    def test_can_append_respects_max_model_len(self):
        bm = _fresh_bm(max_model_len=10)
        s = _seq([1] * 9)
        bm.allocate(s)
        # lookahead that would push past max_model_len
        assert bm.can_append(s, lookahead_num_tokens=2) is False
        assert bm.can_append(s, lookahead_num_tokens=1) is True


# ---------------------------------------------------------------------------
# Draft-vs-target independence
# ---------------------------------------------------------------------------
class TestDraftTargetIndependence:
    def test_draft_bm_uses_draft_block_table(self):
        t_bm = _fresh_bm(is_draft=False)
        d_bm = _fresh_bm(is_draft=True)
        s = _seq([1, 2, 3, 4, 5])
        t_bm.allocate(s)
        d_bm.allocate(s)
        # Separate tables; can share ids because each bm has its own pool
        assert s.block_table and s.draft_block_table
        # Deallocating one does not affect the other bm's state
        t_bm.deallocate(s)
        assert s.block_table == []
        assert s.draft_block_table  # untouched
        d_bm.deallocate(s)
        assert s.draft_block_table == []


# ---------------------------------------------------------------------------
# Hash function sanity
# ---------------------------------------------------------------------------
def test_compute_hash_includes_prefix():
    h_no_prefix = BlockManager.compute_hash([1, 2, 3, 4])
    h_with_prefix = BlockManager.compute_hash([1, 2, 3, 4], prefix=999)
    assert h_no_prefix != h_with_prefix


def test_compute_hash_is_deterministic():
    a = BlockManager.compute_hash([1, 2, 3, 4], prefix=5)
    b = BlockManager.compute_hash([1, 2, 3, 4], prefix=5)
    assert a == b


def test_block_reset_clears_state():
    b = Block(block_id=7)
    b.ref_count = 3
    b.hash = 42
    b.token_ids = [1, 2, 3]
    b.reset()
    assert b.ref_count == 1
    assert b.hash == -1
    assert b.token_ids == []
