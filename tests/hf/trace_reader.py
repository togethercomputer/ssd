"""Batch-aware reader for SSD draft-side tensor dumps (SSD_DUMP_TENSORS_DIR).

The draft process (shared by the SSD engine and TGL) dumps, per round:
  prefill_request_<ts>.pt      — one per prefill wave; may hold several seqs
  speculation_request_<ts>.pt  — [B]-row request (cache_keys, extends, ...)
  speculation_response_<ts>.pt — [B]-row response (speculations, logits, hits)

This module turns a dump directory into per-sequence traces:
  rounds -> split_per_seq -> SeqTrace(rows) -> chain_prefixes / accept lengths

Conventions handled here so callers don't have to:
- `speculations` is dumped FLAT [B*K] (draft sends out_tokens.reshape(-1));
  reshaped to [B, K] via the paired request's metadata.
- round-0 sentinel: cache_keys[:,1] is -2 on a request with no prior
  acceptance (both engines send accepted_len-1 with the "no prior" value
  encoded as -1-1 = -2 in TGL and last_spec_step_accepted_len-1 = -2 in the
  SSD engine). Any value < 0 is normalized to None ("no prior round").
- TGL identifies rows by hash_to_int64(rid); use `hash_rid` on client-chosen
  rids to map dump rows to requests. The SSD engine uses integer seq_ids.
- Prefill dumps carry no seq ids; they are matched to sequences by exact
  prompt-token equality (requires radix/prefix caching disabled so the full
  prompt is sent to the draft).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import torch


def hash_rid(rid: str) -> int:
    """Replicates tgl python/sglang/private/speculative/spec_worker.py:hash_to_int64."""
    return int.from_bytes(hashlib.md5(rid.encode()).digest()[:8], "little", signed=True)


@dataclass
class Round:
    """One speculation request/response pair (whole batch)."""

    index: int
    request: dict
    response: dict
    B: int
    K: int
    speculations: torch.Tensor  # [B, K] (reshaped from the flat dump)

    @property
    def seq_ids(self) -> list[int]:
        return self.request["cache_keys"][:, 0].tolist()


@dataclass
class SeqRow:
    """One sequence's slice of one round."""

    round_index: int
    k_prev: int | None  # accepted spec tokens in the previous round; None on round 0
    rec_token: int
    num_tokens: int
    spec_tokens: list[int]
    cache_hit: int | None
    logits: torch.Tensor | None  # [K, V]
    extend_count: int | None
    extend_token_ids: torch.Tensor | None  # [K+1]
    extend_activations: torch.Tensor | None  # [K+1, act_dim]


@dataclass
class SeqTrace:
    seq_id: int
    prompt_tokens: list[int] | None = None
    prompt_eagle_acts: torch.Tensor | None = None
    rows: list[SeqRow] = field(default_factory=list)


def load_trace(trace_dir: Path) -> list[Round]:
    trace_dir = Path(trace_dir)
    req_files = sorted(trace_dir.glob("speculation_request_*.pt"))
    resp_files = sorted(trace_dir.glob("speculation_response_*.pt"))
    assert len(req_files) == len(resp_files), (
        f"unpaired dumps: {len(req_files)} requests vs {len(resp_files)} responses "
        f"(SSD_RUN_NAME set? it makes every round overwrite one file)"
    )
    rounds: list[Round] = []
    for i, (rf, pf) in enumerate(zip(req_files, resp_files)):
        request = torch.load(rf, weights_only=False)
        response = torch.load(pf, weights_only=False)
        B, K = int(request["metadata"][0]), int(request["metadata"][1])
        assert request["cache_keys"].shape == (B, 3), (
            f"round {i}: cache_keys shape {tuple(request['cache_keys'].shape)} != ({B}, 3)"
        )
        assert request["num_tokens"].shape == (B,)
        spec = response["speculations"]
        if spec.dim() == 1:
            assert spec.numel() == B * K, (
                f"round {i}: flat speculations has {spec.numel()} elements, expected B*K={B * K}"
            )
            spec = spec.view(B, K)
        else:
            assert spec.shape == (B, K)
        if response.get("cache_hits") is not None:
            assert response["cache_hits"].shape[0] == B
        if response.get("logits") is not None:
            assert response["logits"].shape[:2] == (B, K)
        if request.get("extend_counts") is not None:
            ec = request["extend_counts"]
            assert ec.shape == (B,) and int(ec.min()) >= 0 and int(ec.max()) <= K
            assert request["extend_token_ids"].shape == (B, K + 1)
            assert request["extend_activations"].shape[:2] == (B, K + 1)
        rounds.append(Round(index=i, request=request, response=response, B=B, K=K, speculations=spec))
    return rounds


def load_prefills(trace_dir: Path) -> list[dict]:
    """Each prefill dump split into per-seq records: prompt token list + eagle acts slice."""
    out = []
    for f in sorted(Path(trace_dir).glob("prefill_request_*.pt")):
        d = torch.load(f, weights_only=False)
        num_tokens = d["num_tokens"].tolist()
        ids = d["input_ids"].tolist()
        acts = d.get("eagle_acts")
        off = 0
        for n in num_tokens:
            rec = {
                "prompt_tokens": ids[off : off + n],
                "eagle_acts": acts[off : off + n] if acts is not None else None,
            }
            out.append(rec)
            off += n
        assert off == len(ids), f"prefill {f.name}: num_tokens does not partition input_ids"
    return out


def split_per_seq(
    rounds: list[Round],
    prefills: list[dict] | None = None,
    prompts_by_seq_id: dict[int, list[int]] | None = None,
) -> dict[int, SeqTrace]:
    """Group rounds into per-sequence traces keyed by cache_keys[:,0].

    prompts_by_seq_id maps seq_id -> prompt token list (for TGL: seq_id =
    hash_rid(rid)); used to attach each prefill record (matched by exact
    prompt tokens) to its sequence.
    """
    traces: dict[int, SeqTrace] = {}
    for rnd in rounds:
        ck = rnd.request["cache_keys"]
        hits = rnd.response.get("cache_hits")
        logits = rnd.response.get("logits")
        ec = rnd.request.get("extend_counts")
        eti = rnd.request.get("extend_token_ids")
        eac = rnd.request.get("extend_activations")
        for i in range(rnd.B):
            sid = int(ck[i, 0])
            trace = traces.setdefault(sid, SeqTrace(seq_id=sid))
            k_prev_raw = int(ck[i, 1])
            expected_first = len(trace.rows) == 0
            if k_prev_raw < 0:
                assert expected_first, (
                    f"seq {sid}: round-0 sentinel {k_prev_raw} appeared mid-stream "
                    f"(row {len(trace.rows)}) — seq id reuse or dropped rounds?"
                )
            else:
                assert not expected_first, (
                    f"seq {sid}: first observed row has k_prev={k_prev_raw}, expected a <0 sentinel"
                )
            trace.rows.append(
                SeqRow(
                    round_index=rnd.index,
                    k_prev=None if k_prev_raw < 0 else k_prev_raw,
                    rec_token=int(ck[i, 2]),
                    num_tokens=int(rnd.request["num_tokens"][i]),
                    spec_tokens=rnd.speculations[i].tolist(),
                    cache_hit=None if hits is None else int(hits[i]),
                    logits=None if logits is None else logits[i],
                    extend_count=None if ec is None else int(ec[i]),
                    extend_token_ids=None if eti is None else eti[i],
                    extend_activations=None if eac is None else eac[i],
                )
            )

    if prompts_by_seq_id is not None:
        for sid, prompt in prompts_by_seq_id.items():
            if sid in traces:
                traces[sid].prompt_tokens = list(prompt)
        if prefills:
            unmatched = []
            for rec in prefills:
                matches = [sid for sid, p in prompts_by_seq_id.items() if p == rec["prompt_tokens"]]
                assert len(matches) <= 1, (
                    "ambiguous prefill->seq match (duplicate prompts); use distinct prompts"
                )
                if matches and matches[0] in traces:
                    tr = traces[matches[0]]
                    tr.prompt_eagle_acts = rec["eagle_acts"]
                else:
                    unmatched.append(rec)
            assert not unmatched, (
                f"{len(unmatched)} prefill record(s) matched no known prompt — "
                f"radix/prefix caching enabled, or prompts_by_seq_id incomplete?"
            )
    return traces


def chain_prefixes(trace: SeqTrace, prompt_tokens: list[int] | None = None) -> list[list[int]]:
    """Rebuild the token prefix the draft saw at each round.

    prefix_0 = prompt + [rec_0]
    prefix_r = prefix_{r-1} + spec_{r-1}[:k_prev_r] + [rec_r]

    Cross-checked against the dumped num_tokens each round.
    """
    prompt = list(prompt_tokens if prompt_tokens is not None else trace.prompt_tokens)
    assert prompt, f"seq {trace.seq_id}: no prompt tokens available"
    prefixes: list[list[int]] = []
    for r, row in enumerate(trace.rows):
        if r == 0:
            assert row.k_prev is None
            prefix = prompt + [row.rec_token]
        else:
            k = row.k_prev
            assert k is not None
            prefix = prefixes[-1] + trace.rows[r - 1].spec_tokens[:k] + [row.rec_token]
        assert len(prefix) == row.num_tokens, (
            f"seq {trace.seq_id} round {r}: chained prefix length {len(prefix)} "
            f"!= dumped num_tokens {row.num_tokens}"
        )
        prefixes.append(prefix)
    return prefixes


def reconstruct_accept_lengths(trace: SeqTrace) -> list[int]:
    """Per-round accepted-suffix lengths (accepted specs + recovery), for
    rounds 1..R-1 — the final round's acceptance is not visible in the dumps
    (no following request); see engine_comparable_accept_length."""
    return [row.k_prev + 1 for row in trace.rows[1:]]


def engine_comparable_accept_length(
    trace: SeqTrace,
    prompt_tokens: list[int],
    completion_tokens: list[int],
    lookahead: int,
) -> float:
    """Mean accept length structurally identical to the engine's
    spec_accept_length = completion_tokens / spec_verify_ct: the between-round
    suffix lengths plus the final round's (possibly max_new_tokens-clipped)
    tail, divided by the number of verify rounds.

    Also validates that the dump chain and the emitted completion agree.
    """
    R = len(trace.rows)
    assert R > 0
    emitted_before_last = (len(prompt_tokens) + 1 + sum(reconstruct_accept_lengths(trace))) - len(prompt_tokens)
    # ^ = len(prefixes[-1]) - len(prompt): tokens emitted up to & incl rec_{R-1}
    tail = len(completion_tokens) - emitted_before_last
    assert -1 <= tail <= lookahead + 1, (
        f"seq {trace.seq_id}: final-round tail {tail} out of range — dump chain "
        f"inconsistent with completion (emitted_before_last={emitted_before_last}, "
        f"completion={len(completion_tokens)})"
    )
    return len(completion_tokens) / R
