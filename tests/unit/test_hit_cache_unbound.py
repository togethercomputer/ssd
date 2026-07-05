"""Tier 0: DraftRunner.hit_cache + _populate_tree_cache on a CPU mock (unbound).

These call the *production* methods (not re-implementations) with a mock `self`
carrying the ~10 attributes they touch. Viable because profile events are
env-gated (SSD_PROFILE off => no CUDA), and the non-jit paths are pure tensor
indexing. Covers, at B>1 with mixed hit/miss rows:

- populate: cache keys are (seq_id, k, rec_token) with the k-layout
  arange(K+1).repeat_interleave(fan_out[_miss]) per row's hit/miss status, and
  values row-aligned with keys.
- lookup: hit rows return exactly the cached (tokens, logits, activations) of
  the matching key; the returned glue input ids are [rec, returned tokens] —
  the internal-consistency contract verify() depends on.
- miss rows: tokens/logits/activations all come from ONE cache row (row 0), so
  they are mutually consistent even though arbitrary.
- empty cache (round 1, fast mode): zero tokens, sentinel logits, and — the B6
  regression — activations must be zero-filled, not uninitialized memory.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from ssd.engine.draft_runner import DraftRunner

pytestmark = pytest.mark.tier0

K, F, V, HID = 2, 2, 32, 8
MQ = F * (K + 1)


def _mock_runner(B_cache_rows: int = 0, eagle: bool = True):
    m = SimpleNamespace()
    m.device = torch.device("cpu")
    m.block_size = 16
    m.hf_config = SimpleNamespace(vocab_size=V, torch_dtype=torch.float32)
    m.hidden_states_dim = HID
    m.tokenizer = None
    fan_hit = torch.tensor([F] * (K + 1), dtype=torch.int64)
    fan_miss = torch.tensor([MQ] + [0] * K, dtype=torch.int64)
    m.config = SimpleNamespace(
        communicate_logits=True,
        use_eagle_or_phoenix=eagle,
        use_eagle=eagle,
        use_phoenix=False,
        jit_speculate=False,
        force_jit_speculate=False,
        speculate_k=K,
        MQ_LEN=MQ,
        fan_out_t=fan_hit,
        fan_out_t_miss=fan_miss,
        verbose=False,
    )
    m._fan_idx_hit = torch.arange(K + 1).repeat_interleave(fan_hit)
    m._fan_idx_miss = torch.arange(K + 1).repeat_interleave(fan_miss)
    m.tree_cache_keys = torch.empty(0, 3, dtype=torch.int64)
    m.tree_cache_tokens = None
    m.tree_cache_logits = None
    m.tree_cache_activations = None
    return m


def _populate(m, seq_ids: list[int], cache_hits: list[int], seed: int = 0):
    """Run the production _populate_tree_cache with a synthetic decode result."""
    g = torch.Generator().manual_seed(seed)
    B = len(seq_ids)
    N = B * MQ
    payload = {
        "seq_ids_expanded": torch.tensor(seq_ids, dtype=torch.int64).repeat_interleave(MQ),
        "rec_flat": torch.randint(0, V, (N,), generator=g),
        "cache_hits_list": cache_hits,
        "block_tables": torch.zeros(B, 4, dtype=torch.int32),
    }
    tokens = torch.randint(0, V, (N, K), generator=g)
    logits = torch.randn(N, K, V, generator=g)
    acts = torch.randn(N, K, HID, generator=g)
    DraftRunner._populate_tree_cache(m, payload, tokens, logits, acts)
    return payload, tokens, logits, acts


def test_populate_key_structure_mixed_hit_miss():
    m = _mock_runner()
    seq_ids = [7, 3]
    payload, tokens, _, _ = _populate(m, seq_ids, cache_hits=[1, 0])

    assert m.tree_cache_keys.shape == (2 * MQ, 3)
    # row 0 (hit): k layout [0,0,1,1,2,2]; row 1 (miss): [0]*MQ
    assert m.tree_cache_keys[:MQ, 1].tolist() == m._fan_idx_hit.tolist()
    assert m.tree_cache_keys[MQ:, 1].tolist() == m._fan_idx_miss.tolist()
    assert (m.tree_cache_keys[:MQ, 0] == 7).all() and (m.tree_cache_keys[MQ:, 0] == 3).all()
    assert torch.equal(m.tree_cache_keys[:, 2], payload["rec_flat"])
    assert torch.equal(m.tree_cache_tokens, tokens)


def test_hit_rows_return_exact_cached_entries_and_consistent_glue():
    m = _mock_runner()
    _, tokens, logits, acts = _populate(m, seq_ids=[7, 3], cache_hits=[1, 1])

    # pick one cached entry per sequence: seq 7 at k=1 (cache row 2), seq 3 at k=2 (row MQ+5)
    picks = [2, MQ + 5]
    request_keys = m.tree_cache_keys[picks].clone()
    B = 2
    out_tokens, out_logits, glue_ids, cache_hits, out_acts = DraftRunner.hit_cache(
        m, request_keys, B, K,
        num_tokens=torch.tensor([10, 20]),
        temperatures=torch.zeros(B),
        draft_block_tables=torch.zeros(B, 4, dtype=torch.int32),
    )
    assert cache_hits.tolist() == [True, True]
    assert torch.equal(out_tokens, tokens[picks])
    assert torch.equal(out_logits, logits[picks])
    assert torch.equal(out_acts, acts[picks])
    # glue input = [rec, returned tokens] per row — the trunk the draft will
    # condition on MUST be the same tokens it just returned
    glue = glue_ids.view(B, K + 1)
    assert torch.equal(glue[:, 0], request_keys[:, 2])
    assert torch.equal(glue[:, 1:], out_tokens)


def test_first_match_wins_on_duplicate_keys():
    m = _mock_runner()
    _populate(m, seq_ids=[7, 3], cache_hits=[1, 1])
    # duplicate an existing key at a later row with different tokens
    dup_of = 1
    m.tree_cache_keys = torch.cat([m.tree_cache_keys, m.tree_cache_keys[dup_of : dup_of + 1]])
    m.tree_cache_tokens = torch.cat([m.tree_cache_tokens, m.tree_cache_tokens[dup_of : dup_of + 1] + 1])
    m.tree_cache_logits = torch.cat([m.tree_cache_logits, m.tree_cache_logits[dup_of : dup_of + 1]])
    m.tree_cache_activations = torch.cat([m.tree_cache_activations, m.tree_cache_activations[dup_of : dup_of + 1]])

    request_keys = m.tree_cache_keys[dup_of : dup_of + 1].clone()
    out_tokens, _, _, cache_hits, _ = DraftRunner.hit_cache(
        m, request_keys, 1, K,
        num_tokens=torch.tensor([10]),
        temperatures=torch.zeros(1),
        draft_block_tables=torch.zeros(1, 4, dtype=torch.int32),
    )
    assert cache_hits.tolist() == [True]
    assert torch.equal(out_tokens[0], m.tree_cache_tokens[dup_of])


def test_miss_rows_return_deterministic_zeros():
    """Miss rows must return ZEROS (token 0 + sentinel logits + zero acts),
    exactly like the empty-cache round — NOT stale cache rows, which leaked
    session history into responses and (via the trunk-token exclusion) into the
    next round's fork sets, and at B>1 leaked one sequence's branch content to
    another. The glue trunk must still be built from the returned tokens, and
    hit rows must be untouched by the miss-row overwrite (gather = copy)."""
    m = _mock_runner()
    _, tokens, logits, acts = _populate(m, seq_ids=[7, 3], cache_hits=[1, 1])

    request_keys = torch.stack([
        m.tree_cache_keys[4],                              # hit (seq 7, k=2)
        torch.tensor([99, 1, V + 5], dtype=torch.int64),   # miss: unknown seq
    ])
    out_tokens, out_logits, glue_ids, cache_hits, out_acts = DraftRunner.hit_cache(
        m, request_keys, 2, K,
        num_tokens=torch.tensor([10, 20]),
        temperatures=torch.zeros(2),
        draft_block_tables=torch.zeros(2, 4, dtype=torch.int32),
    )
    assert cache_hits.tolist() == [True, False]
    assert torch.equal(out_tokens[0], tokens[4])
    assert torch.equal(out_logits[0], logits[4])
    assert torch.equal(out_acts[0], acts[4])
    # miss row: deterministic zeros/sentinel, independent of cache content
    assert (out_tokens[1] == 0).all()
    assert (out_logits[1, :, 0] == 0).all() and torch.isneginf(out_logits[1, :, 1:]).all()
    assert (out_acts[1] == 0).all()
    # the cache itself must NOT have been zeroed by the miss-row overwrite
    assert torch.equal(m.tree_cache_tokens, tokens)
    glue = glue_ids.view(2, K + 1)
    assert torch.equal(glue[:, 1:], out_tokens)


def test_empty_cache_fast_path_zero_filled_no_uninitialized_activations():
    """B6 regression: round 1 in fast mode (empty cache, no jit) must return
    deterministic zero tokens / sentinel logits / ZERO activations. We
    monkeypatch torch.empty to poison any uninitialized allocation with NaN, so
    a regression to torch.empty shows up as NaN in the output."""
    m = _mock_runner()
    real_empty = torch.empty

    def poisoned_empty(*args, **kwargs):
        t = real_empty(*args, **kwargs)
        if t.is_floating_point():
            t.fill_(float("nan"))
        return t

    try:
        torch.empty = poisoned_empty
        out_tokens, out_logits, glue_ids, cache_hits, out_acts = DraftRunner.hit_cache(
            m, torch.tensor([[5, -2, 9]], dtype=torch.int64), 1, K,
            num_tokens=torch.tensor([10]),
            temperatures=torch.zeros(1),
            draft_block_tables=torch.zeros(1, 4, dtype=torch.int32),
        )
    finally:
        torch.empty = real_empty

    assert cache_hits.tolist() == [False]
    assert (out_tokens == 0).all()
    # sentinel logits: -inf except column 0
    assert (out_logits[:, :, 0] == 0).all()
    assert torch.isneginf(out_logits[:, :, 1:]).all()
    assert not torch.isnan(out_acts).any(), (
        "empty-cache fast path leaked uninitialized activations (B6)"
    )
    assert (out_acts == 0).all()
