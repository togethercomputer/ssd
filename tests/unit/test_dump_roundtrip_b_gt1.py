"""Tier 0: dump round-trip at B>1 — real dump writers vs tests/hf/trace_reader.

Synthesizes a 3-sequence, 5-round trace by calling the REAL dump methods
(PrefillRequest/SpeculationRequest/SpeculationResponse.dump) with
SSD_DUMP_TENSORS_DIR pointed at a tmp dir, covering:

- two prefill waves (seqs {A,B} then {C} joining later),
- varying batch composition per round (B=2 -> B=3 -> B=2 after A leaves),
- heterogeneous accept counts (k_prev mix incl. 0 and K),
- round-0 sentinel (-2) rows appearing mid-trace for the late joiner,
- FLAT [B*K] speculations exactly as the draft dumps them.

Then validates the reader end to end: pairing/reshaping, per-seq splitting,
prefill matching by prompt tokens, prefix chaining against dumped num_tokens,
accept-length reconstruction, and the engine-comparable mean.
"""
from __future__ import annotations

import pytest
import torch

from ssd.engine.helpers.runner_helpers import (
    PrefillRequest,
    SpeculationRequest,
    SpeculationResponse,
)

from tests.hf.trace_reader import (
    chain_prefixes,
    engine_comparable_accept_length,
    hash_rid,
    load_prefills,
    load_trace,
    reconstruct_accept_lengths,
    split_per_seq,
)

pytestmark = pytest.mark.tier0

K = 2
V = 40
ACT_DIM = 6
MAX_BLOCKS = 4
DEV = torch.device("cpu")


def _dump_prefill(seqs: list[tuple[int, list[int]]]):
    ids = [t for _, p in seqs for t in p]
    num_tokens = torch.tensor([len(p) for _, p in seqs], dtype=torch.int64)
    req = PrefillRequest.prepare(
        input_ids=torch.tensor(ids, dtype=torch.int64),
        num_tokens=num_tokens,
        draft_block_table=torch.zeros(len(seqs), MAX_BLOCKS, dtype=torch.int32),
        eagle_acts=torch.randn(len(ids), ACT_DIM),
        max_blocks=MAX_BLOCKS,
        device=DEV,
    )
    req.dump()


class _Seq:
    def __init__(self, sid: int, prompt: list[int]):
        self.sid = sid
        self.prompt = prompt
        self.prefix_len: int | None = None  # incl. latest recovery token
        self.last_spec: list[int] | None = None
        self.expected_prefix: list[int] | None = None
        self.accepts: list[int] = []

    def step(self, k_prev: int | None, rec_token: int):
        if k_prev is None:
            assert self.prefix_len is None
            self.expected_prefix = self.prompt + [rec_token]
        else:
            self.accepts.append(k_prev)
            self.expected_prefix = self.expected_prefix + self.last_spec[:k_prev] + [rec_token]
        self.prefix_len = len(self.expected_prefix)


def _dump_round(rows: list[tuple[_Seq, int | None, int, list[int], int]]):
    """rows: (seq, k_prev(None=first), rec_token, spec_tokens[K], cache_hit)"""
    B = len(rows)
    req = SpeculationRequest.prepare(
        batch_size=B, lookahead=K, max_blocks=MAX_BLOCKS, vocab_size=V,
        draft_dtype=torch.float32, device=DEV, eagle=True, eagle_act_dim=ACT_DIM,
    )
    for i, (seq, k_prev, rec, spec, _hit) in enumerate(rows):
        seq.step(k_prev, rec)
        req.cache_keys[i, 0] = seq.sid
        req.cache_keys[i, 1] = -2 if k_prev is None else k_prev
        req.cache_keys[i, 2] = rec
        req.num_tokens[i] = seq.prefix_len
        n = 0 if k_prev is None else k_prev
        req.extend_counts[i] = n
        req.extend_token_ids[i, : n + 1] = torch.tensor(
            ([] if n == 0 else seq.expected_prefix[-n - 1 : -1]) + [rec], dtype=torch.int64
        )
        req.extend_activations[i] = torch.randn(K + 1, ACT_DIM)
    req.dump()

    out_tokens = torch.tensor([spec for (_, _, _, spec, _) in rows], dtype=torch.int64)
    resp = SpeculationResponse(
        speculations=out_tokens.reshape(-1),  # FLAT, as the draft dumps it
        logits_q=torch.randn(B, K, V),
        cache_hits=torch.tensor([h for (*_, h) in rows], dtype=torch.int64),
    )
    resp.dump()
    for (seq, _, _, spec, _) in rows:
        seq.last_spec = list(spec)


@pytest.fixture()
def trace(tmp_path, monkeypatch):
    monkeypatch.delenv("SSD_RUN_NAME", raising=False)  # constant name => dumps collide
    monkeypatch.setenv("SSD_DUMP_TENSORS_DIR", str(tmp_path))
    torch.manual_seed(0)

    A = _Seq(hash_rid("req-A"), [1, 2, 3])
    B_ = _Seq(hash_rid("req-B"), [4, 5, 6, 7])
    C = _Seq(hash_rid("req-C"), [8, 9])

    _dump_prefill([(A.sid, A.prompt), (B_.sid, B_.prompt)])
    _dump_round([  # round 0: A,B first requests
        (A, None, 10, [11, 12], 0),
        (B_, None, 20, [21, 22], 0),
    ])
    _dump_round([  # round 1: A accepts K=2, B accepts 0
        (A, 2, 13, [14, 15], 1),
        (B_, 0, 23, [24, 25], 0),
    ])
    _dump_prefill([(C.sid, C.prompt)])
    _dump_round([  # round 2: C joins with sentinel; heterogeneous accepts
        (A, 1, 16, [17, 18], 1),
        (B_, 2, 26, [27, 28], 1),
        (C, None, 30, [31, 32], 0),
    ])
    _dump_round([  # round 3
        (A, 0, 19, [33, 34], 0),
        (B_, 1, 29, [35, 36], 1),
        (C, 2, 37, [38, 39], 1),
    ])
    _dump_round([  # round 4: A left the batch
        (B_, 0, 6, [3, 2], 0),
        (C, 1, 5, [1, 0], 1),
    ])
    return tmp_path, A, B_, C


def test_reader_pairs_reshapes_and_splits(trace):
    tmp_path, A, B_, C = trace
    rounds = load_trace(tmp_path)
    assert [r.B for r in rounds] == [2, 2, 3, 3, 2]
    assert all(r.speculations.shape == (r.B, K) for r in rounds)

    prompts = {A.sid: A.prompt, B_.sid: B_.prompt, C.sid: C.prompt}
    traces = split_per_seq(rounds, prefills=load_prefills(tmp_path), prompts_by_seq_id=prompts)
    assert set(traces) == {A.sid, B_.sid, C.sid}
    assert [len(traces[s.sid].rows) for s in (A, B_, C)] == [4, 5, 3]

    # per-seq round indices reflect join/leave
    assert [r.round_index for r in traces[A.sid].rows] == [0, 1, 2, 3]
    assert [r.round_index for r in traces[C.sid].rows] == [2, 3, 4]

    # prefill eagle acts attached, with per-seq length == prompt length
    for s in (A, B_, C):
        acts = traces[s.sid].prompt_eagle_acts
        assert acts is not None and len(acts) == len(s.prompt)


def test_prefix_chaining_and_accept_lengths(trace):
    tmp_path, A, B_, C = trace
    rounds = load_trace(tmp_path)
    prompts = {A.sid: A.prompt, B_.sid: B_.prompt, C.sid: C.prompt}
    traces = split_per_seq(rounds, prompts_by_seq_id=prompts)

    for s in (A, B_, C):
        prefixes = chain_prefixes(traces[s.sid], s.prompt)
        assert prefixes[-1] == s.expected_prefix
        assert reconstruct_accept_lengths(traces[s.sid]) == [k + 1 for k in s.accepts]

    # engine-comparable mean: completion consistent with the chain + a 1-token tail
    s = B_
    emitted = len(s.expected_prefix) - len(s.prompt)  # through last rec
    completion = list(range(emitted + 1))  # final round emitted 1 more token
    mean = engine_comparable_accept_length(traces[s.sid], s.prompt, completion, lookahead=K)
    assert mean == pytest.approx(len(completion) / len(traces[s.sid].rows))


def test_reader_rejects_corrupt_chain(trace):
    tmp_path, A, B_, C = trace
    rounds = load_trace(tmp_path)
    traces = split_per_seq(rounds, prompts_by_seq_id={A.sid: A.prompt})
    traces[A.sid].rows[2].num_tokens += 1  # corrupt
    with pytest.raises(AssertionError, match="prefix length"):
        chain_prefixes(traces[A.sid], A.prompt)


def test_reader_rejects_mid_stream_sentinel(trace):
    tmp_path, A, B_, C = trace
    rounds = load_trace(tmp_path)
    # forge a sentinel into a later round for seq A
    rounds[3].request["cache_keys"][0, 1] = -2
    with pytest.raises(AssertionError, match="sentinel"):
        split_per_seq(rounds)


def test_reader_rejects_ambiguous_prefill_match(trace):
    tmp_path, A, B_, C = trace
    rounds = load_trace(tmp_path)
    prompts = {A.sid: A.prompt, B_.sid: A.prompt}  # duplicate prompts
    with pytest.raises(AssertionError, match="ambiguous"):
        split_per_seq(rounds, prefills=load_prefills(tmp_path), prompts_by_seq_id=prompts)
