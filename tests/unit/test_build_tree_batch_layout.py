"""Tier 0: DraftRunner._build_tree_batch layout semantics on a CPU mock.

_build_tree_batch has two eagle/phoenix layouts for the glue decode:
- a vectorized "uniform" fast path taken when B==1 or all extend_counts equal,
- a varlen fallback for heterogeneous extend_counts (the realistic B>1 case,
  where different sequences accepted different numbers of tokens).

The fallback is the undertested one. These tests call the PRODUCTION method
with a recording, pure-per-token run_model stub (logits/prenorm are functions
of (input_id, position, hidden_state) only), so:

  per-seq outputs of [B=3 heterogeneous batch]  ==  per-seq outputs of [B=1 runs]

is exactly a layout-equivalence check between the two paths. Also pinned:
fused token/hidden placement, positions/slots/cu_seqlens/context_lens, the
K+1 extraction indices, per-row hit/miss fan repeat counts of the tree seeds,
and the Phoenix recovery-activation expansion to (B*MQ_LEN, act_dim) — the
documented zero-padding poisoning regression.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from ssd.engine.draft_runner import DraftRunner
from ssd.utils.context import get_context

pytestmark = pytest.mark.tier0

K, F = 2, 2
MQ = F * (K + 1)
V = 24
HID = 8
ACT = 6
BLOCK = 16
DT = torch.float64  # exact row-wise reproducibility across batch layouts


def _mk_runner(mode: str, recorder: list):
    assert mode in ("eagle", "phoenix")
    m = SimpleNamespace()
    m.device = torch.device("cpu")
    m.block_size = BLOCK
    m.tokenizer = None
    hidden_dim = HID if mode == "eagle" else ACT
    m.hidden_states_dim = hidden_dim
    m.hf_config = SimpleNamespace(vocab_size=V, torch_dtype=DT)
    fan_hit = torch.tensor([F] * (K + 1), dtype=torch.int64)
    fan_miss = torch.tensor([MQ] + [0] * K, dtype=torch.int64)
    m.config = SimpleNamespace(
        verbose=False,
        use_eagle_or_phoenix=True,
        use_eagle=(mode == "eagle"),
        use_phoenix=(mode == "phoenix"),
        speculate_k=K,
        MQ_LEN=MQ,
        async_fan_out=F,
        fan_out_list=[F] * (K + 1),
        fan_out_list_miss=[MQ] + [0] * K,
        fan_out_t=fan_hit,
        fan_out_t_miss=fan_miss,
        communicate_logits=True,
        jit_speculate=False,
        force_jit_speculate=False,
    )
    m._arange_kp1 = torch.arange(K + 1)
    m._arange_mq = torch.arange(MQ)
    m._fan_idx_hit = torch.arange(K + 1).repeat_interleave(fan_hit)
    m._fan_idx_miss = torch.arange(K + 1).repeat_interleave(fan_miss)
    if mode == "eagle":
        g = torch.Generator().manual_seed(7)
        fc = torch.nn.Linear(ACT, HID, bias=False).to(DT)
        with torch.no_grad():
            fc.weight.copy_(torch.randn(HID, ACT, generator=g, dtype=DT))
        m.model = SimpleNamespace(fc=fc)
    else:
        m.model = SimpleNamespace()

    def run_model(input_ids, positions, is_prefill, last_only, hidden_states=None, **kw):
        assert hidden_states is not None and hidden_states.shape[0] == input_ids.shape[0]
        ctx = get_context()
        recorder.append(
            dict(
                input_ids=input_ids.clone(),
                positions=positions.clone(),
                hidden_states=hidden_states.clone(),
                cu_seqlens_q=None if ctx.cu_seqlens_q is None else ctx.cu_seqlens_q.clone(),
                max_seqlen_q=ctx.max_seqlen_q,
                slot_mapping=None if ctx.slot_mapping is None else ctx.slot_mapping.clone(),
                context_lens=None if ctx.context_lens is None else ctx.context_lens.clone(),
            )
        )
        # pure per-token functions: identical rows => identical outputs, in any batch
        base = (
            input_ids.to(DT) * 1.7
            + positions.to(DT) * 0.31
            + hidden_states.to(DT).sum(dim=-1) * 0.011
        )
        logits = torch.sin(base[:, None] * torch.arange(1, V + 1, dtype=DT) * 0.13)
        prenorm = torch.cos(base[:, None] * torch.arange(1, hidden_dim + 1, dtype=DT) * 0.29)
        return logits, prenorm

    m.run_model = run_model
    m.prepare_glue_decode_ctxt_eagle = lambda **kw: DraftRunner.prepare_glue_decode_ctxt_eagle(m, **kw)
    return m


def _mk_inputs(seqs: list[dict], mode: str, seed: int = 0):
    """seqs: per-seq dicts with num_tokens, n_ext, hit, seq_id."""
    g = torch.Generator().manual_seed(seed)
    B = len(seqs)
    act_dim = ACT
    prev_dim = HID if mode == "eagle" else ACT
    partial = {
        "num_tokens": torch.tensor([s["num_tokens"] for s in seqs], dtype=torch.int64),
        "seq_ids": torch.tensor([s["seq_id"] for s in seqs], dtype=torch.int64),
        "temperatures": torch.zeros(B, dtype=torch.float32),
        "dbt": torch.stack([torch.arange(s["seq_id"] * 100, s["seq_id"] * 100 + 8, dtype=torch.int32) for s in seqs]),
        "cache_hits": torch.tensor([s["hit"] for s in seqs], dtype=torch.bool),
        "returned_tokens": None,
        "previous_activations": torch.randn(B, K, prev_dim, generator=g, dtype=DT),
        "extend_counts": torch.tensor([s["n_ext"] for s in seqs], dtype=torch.int64),
        "extend_eagle_acts": torch.randn(B, K + 1, act_dim, generator=g, dtype=DT),
        "extend_token_ids": torch.randint(0, V, (B, K + 1), generator=g),
    }
    glue_ids = torch.randint(0, V, (B * (K + 1),), generator=g)
    return partial, glue_ids


def _slice_partial(partial, glue_ids, b):
    one = {
        k: (v[b : b + 1].clone() if isinstance(v, torch.Tensor) else v)
        for k, v in partial.items()
    }
    return one, glue_ids.view(-1, K + 1)[b : b + 1].reshape(-1).clone()


SEQS = [
    dict(num_tokens=21, n_ext=0, hit=1, seq_id=1),
    dict(num_tokens=35, n_ext=1, hit=0, seq_id=2),
    dict(num_tokens=18, n_ext=K, hit=1, seq_id=3),
]


@pytest.mark.parametrize("mode", ["eagle", "phoenix"])
def test_uniform_vs_varlen_equivalence_per_seq(mode):
    B = len(SEQS)
    rec_batch: list = []
    m = _mk_runner(mode, rec_batch)
    partial, glue_ids = _mk_inputs(SEQS, mode)
    out_batch = DraftRunner._build_tree_batch(m, dict(partial), glue_ids)

    for b, s in enumerate(SEQS):
        rec_solo: list = []
        m1 = _mk_runner(mode, rec_solo)
        partial_1, glue_1 = _slice_partial(partial, glue_ids, b)
        out_solo = DraftRunner._build_tree_batch(m1, partial_1, glue_1)

        sl_mq = slice(b * MQ, (b + 1) * MQ)
        for key in ("input_ids", "rec_flat"):
            assert torch.equal(out_batch[key][sl_mq], out_solo[key]), (
                f"{mode} seq {b}: {key} differs between varlen batch and solo run"
            )
        assert torch.equal(out_batch["positions"][sl_mq], out_solo["positions"])
        assert torch.equal(out_batch["rope_positions"][sl_mq], out_solo["rope_positions"])
        assert torch.equal(out_batch["hidden_states"][sl_mq], out_solo["hidden_states"]), (
            f"{mode} seq {b}: tree seed hidden states differ (glue layout bug)"
        )
        if mode == "phoenix":
            assert torch.equal(
                out_batch["target_recovery_activations"][sl_mq],
                out_solo["target_recovery_activations"],
            )


def test_varlen_fused_placement_eagle():
    rec: list = []
    m = _mk_runner("eagle", rec)
    partial, glue_ids = _mk_inputs(SEQS, "eagle")
    DraftRunner._build_tree_batch(m, dict(partial), glue_ids)

    call = rec[0]  # the glue decode forward
    B = len(SEQS)
    n_ext = partial["extend_counts"]
    num_tokens = partial["num_tokens"]
    seqlens = (n_ext + K + 1).tolist()
    cu = torch.tensor([0] + list(torch.cumsum(torch.tensor(seqlens), 0)), dtype=torch.int32)
    assert torch.equal(call["cu_seqlens_q"], cu.to(call["cu_seqlens_q"].dtype))
    assert torch.equal(call["context_lens"], (num_tokens + K).to(call["context_lens"].dtype))

    gd = glue_ids.view(B, K + 1)
    for b in range(B):
        n = int(n_ext[b])
        seg = slice(int(cu[b]), int(cu[b + 1]))
        ids = call["input_ids"][seg]
        pos = call["positions"][seg]
        hs = call["hidden_states"][seg]
        # tokens: [ext_0..ext_{n-1}, rec, spec_0..spec_{K-1}]
        assert torch.equal(ids[: n + 1], partial["extend_token_ids"][b, : n + 1])
        assert torch.equal(ids[n + 1 :], gd[b, 1:])
        # positions: num_tokens-1-n .. num_tokens-1+K
        exp_pos = torch.arange(int(num_tokens[b]) - 1 - n, int(num_tokens[b]) + K)
        assert torch.equal(pos, exp_pos)
        # slots: dbt[b][pos // BLOCK] * BLOCK + pos % BLOCK
        exp_slots = partial["dbt"][b][(exp_pos // BLOCK)].to(torch.int64) * BLOCK + exp_pos % BLOCK
        assert torch.equal(call["slot_mapping"][seg].to(torch.int64), exp_slots)
        # hidden: extend+rec slots get fc(target acts); spec slots get prev_acts.
        # (tiny rtol: fc is a matmul over different M dims in the two layouts,
        # which legitimately differs in the last float bits)
        exp_ext_hs = m.model.fc(partial["extend_eagle_acts"][b, : n + 1].to(DT))
        assert torch.allclose(hs[: n + 1], exp_ext_hs, rtol=1e-12, atol=0)
        assert torch.equal(hs[n + 1 :], partial["previous_activations"][b])


def test_tree_seed_extraction_and_fan_repeat_counts_eagle():
    rec: list = []
    m = _mk_runner("eagle", rec)
    partial, glue_ids = _mk_inputs(SEQS, "eagle")
    out = DraftRunner._build_tree_batch(m, dict(partial), glue_ids)

    call = rec[0]
    B = len(SEQS)
    n_ext = partial["extend_counts"]
    seqlens = (n_ext + K + 1).tolist()
    cu = [0]
    for s in seqlens:
        cu.append(cu[-1] + s)
    # recompute the stub's prenorm at the rec+spec rows and repeat by fan pattern
    base = (
        call["input_ids"].to(DT) * 1.7
        + call["positions"].to(DT) * 0.31
        + call["hidden_states"].to(DT).sum(dim=-1) * 0.011
    )
    prenorm = torch.cos(base[:, None] * torch.arange(1, HID + 1, dtype=DT) * 0.29)
    for b in range(B):
        rows = torch.arange(cu[b] + int(n_ext[b]), cu[b] + int(n_ext[b]) + K + 1)
        anchors = prenorm[rows]  # [K+1, HID]
        fan = m.config.fan_out_t if partial["cache_hits"][b] else m.config.fan_out_t_miss
        expected = torch.repeat_interleave(anchors, fan, dim=0)  # [MQ, HID]
        assert torch.equal(out["hidden_states"][b * MQ : (b + 1) * MQ], expected), (
            f"seq {b}: tree seeds not anchored/repeated per its own hit/miss fan pattern"
        )
        # rope positions follow the same per-row fan k-layout
        j_idx = torch.arange(K + 1).repeat_interleave(fan)
        exp_rope = (int(partial["num_tokens"][b]) - 1) + j_idx + 1
        assert torch.equal(out["rope_positions"][b * MQ : (b + 1) * MQ], exp_rope)


def test_phoenix_recovery_expansion_shape_and_content():
    rec: list = []
    m = _mk_runner("phoenix", rec)
    partial, glue_ids = _mk_inputs(SEQS, "phoenix")
    out = DraftRunner._build_tree_batch(m, dict(partial), glue_ids)

    tra = out["target_recovery_activations"]
    B = len(SEQS)
    assert tra.shape == (B * MQ, ACT), (
        "phoenix recovery activations must be expanded to (B*MQ_LEN, act_dim); "
        "a (B, act_dim) tensor here gets zero-padded downstream, poisoning branches"
    )
    for b in range(B):
        expected = partial["extend_eagle_acts"][b, int(partial["extend_counts"][b])]
        block = tra[b * MQ : (b + 1) * MQ]
        assert (block == expected).all(), f"seq {b}: recovery activation not broadcast to all branches"
        # tree seeds for phoenix are ALSO the recovery activation (no recurrence)
        assert (out["hidden_states"][b * MQ : (b + 1) * MQ] == expected).all()
