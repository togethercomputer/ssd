"""Scripted-target batch>1 correctness suite for the shared DraftRunner.

A real DraftRunner child process (production code, CUDA graphs ON) is driven by
a scripted fake target over NCCL — no server, no 8B model, 2 GPUs. Sequences
use seeded synthetic target activations consumed identically by an HF mirror
(tests/hf/mirror_replay.EagleMirror), which supplies fork sets (for scripting
guaranteed hits) and reference branch tokens/logits.

What must hold EXACTLY (immune to bf16 kernel drift):
  - cache_hits per row (hits scripted at max-margin candidates, misses far out)
  - response self-consistency: argmax(logits_q[row]) == speculations[row]
  - row-permutation equivariance: an identical script re-run under fresh seq
    ids with batch rows permuted returns row-permuted identical responses
    (same N => same kernels => bit-exact); catches cross-row indexing bugs
    like the TGL extend-pointer misalignment class
  - re-run determinism (also pins the empty-cache zero-fill fix)
  - batch shrink/regrow: absent-then-returning rows are guaranteed misses

What holds STATISTICALLY (the eagle recurrence chain is mildly chaotic, and
the mirror is a different implementation of the same math):
  - hit-row tokens match the mirror branch, or rank low in its logits

Run: pytest tests/draft_runner -x -q   (needs 2 GPUs; ~2 min startup for
graph capture, then seconds per test)
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.draft_runner._harness import DraftSession, build_target_config
from tests.hf.helpers import require_8b_target, require_eagle_llama_8b_draft
from tests.hf.mirror_replay import EagleMirror

pytestmark = [pytest.mark.tier1]

K = 2
FANOUT = 3
MQ = FANOUT * (K + 1)
BLOCK = 64
MAX_MODEL_LEN = 2048
# The DraftRunner's CUDA-graph static block-table buffers are sized
# max_model_len/block_size wide; requests must use that width.
MAX_BLOCKS = MAX_MODEL_LEN // BLOCK
PROMPT_LEN = 33
MAX_SEQS = 4
ACT_DIM = 3 * 4096
MISS_TOKEN = 424242 % 128000  # far outside any top-fanout candidate set

requires_2gpus = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="needs 2 GPUs (target + draft)"
)


class ScriptedSeq:
    """Driver-side state for one sequence, with its HF mirror.

    Conditioning uses REAL Llama-8B target activations, teacher-forced over the
    sequence's actual token chain each round (synthetic random activations give
    off-manifold draft logits whose top-fanout margins are razor thin, making
    hit/miss scripting flaky between kernels)."""

    _next_block = 0

    PROMPT_BANK = [
        "The Golden Gate Bridge connects San Francisco to Marin County and is one of",
        "Photosynthesis is the process by which green plants convert sunlight into",
        "The French Revolution began in 1789 when the people of Paris stormed the",
        "In machine learning, gradient descent is an optimization algorithm that",
        "The Pacific Ocean is the largest and deepest of Earth's five oceans, covering",
        "Ludwig van Beethoven composed his Ninth Symphony while almost completely",
        "The human immune system defends the body against infection through layers of",
        "Mount Everest, the highest mountain on Earth, attracts climbers from all over",
    ]

    def __init__(self, seq_id: int, seq_tag: int, draft, session, target, tokenizer):
        self.seq_id = seq_id
        self.tag = seq_tag
        self.session = session
        self.target = target
        # real text => on-manifold activations => healthy fork margins (random
        # token prompts leave the draft's top-fanout boundaries at coin-flip
        # margins where engine and mirror kernels legitimately disagree)
        text = self.PROMPT_BANK[seq_tag % len(self.PROMPT_BANK)]
        self.prompt = tokenizer(text)["input_ids"][:PROMPT_LEN]
        self.chain: list[int] = list(self.prompt)  # tokens incl. latest recovery
        self._dup_acts: torch.Tensor | None = None  # [len(chain), ACT_DIM]; index g conditions token g
        self.nt = None  # num_tokens incl. latest recovery
        self.last_response_tokens: list[int] | None = None
        self.last_hit: bool = False  # was this seq's previous round a cache hit?
        self._refresh_acts()
        self.mirror = EagleMirror(
            draft, self.prompt, self._dup_acts[: len(self.prompt)],
            K, [FANOUT] * (K + 1), [MQ] + [0] * K, phoenix=False,
        )
        # contiguous private block range
        n_blocks = 12
        self.blocks = torch.arange(
            ScriptedSeq._next_block, ScriptedSeq._next_block + n_blocks, dtype=torch.int32
        )
        ScriptedSeq._next_block += n_blocks
        assert ScriptedSeq._next_block * BLOCK < session.num_kv_blocks * BLOCK

    def _refresh_acts(self):
        """Recompute dup-shifted target activations over the current chain:
        _dup_acts[g] = h_{g-1} (h_{-1} := h_0), the eagle conditioning of token g."""
        from tests.hf.test_ssd_vs_hf_reference import (
            get_hf_target_activations_for_eagle_or_phoenix,
        )

        acts = get_hf_target_activations_for_eagle_or_phoenix(
            self.target, self.chain, eagle=True, phoenix=False
        ).to(torch.bfloat16)
        self._dup_acts = torch.cat([acts[:1], acts]).cpu()

    def block_table(self, max_blocks: int) -> torch.Tensor:
        bt = torch.full((max_blocks,), -1, dtype=torch.int32)
        bt[: len(self.blocks)] = self.blocks
        return bt

    def round_fields(self, k: int | None, rec: int):
        """Build this seq's request row for outcome (k, rec). k=None => round 0."""
        if k is None:
            assert self.nt is None
            self.chain.append(rec)
            self.nt = len(self.prompt) + 1
            k_wire = -2
            n_ext = 0
        else:
            base = self.nt
            self.chain.extend(self.last_response_tokens[:k])
            self.chain.append(rec)
            self.nt = base + k + 1
            k_wire = k
            n_ext = k
        self._refresh_acts()
        ext_ids = torch.zeros(K + 1, dtype=torch.int64)
        ext_acts = torch.zeros(K + 1, ACT_DIM)
        if n_ext > 0:
            base = self.nt - n_ext - 1
            ext_ids[:n_ext] = torch.tensor(self.last_response_tokens[:n_ext])
            ext_acts[:n_ext] = self._dup_acts[base : base + n_ext].float()
        ext_acts[n_ext] = self._dup_acts[self.nt - 1].float()
        ext_ids[n_ext] = rec
        row = SimpleNamespace(
            extend_count=n_ext, extend_token_ids=ext_ids, extend_activations=ext_acts,
            rec_token=rec, num_tokens=self.nt, cache_hit=None,  # filled after response
        )
        return dict(
            cache_keys=torch.tensor([self.seq_id, k_wire, rec], dtype=torch.int64),
            num_tokens=self.nt, extend_counts=n_ext,
            extend_token_ids=ext_ids, extend_activations=ext_acts, row=row,
        )

    MARGIN_TAU = 1.0  # bf16-robust logit lead required to script an exact hit

    def hit_or_miss(self, k: int, miss_token: int) -> tuple[int, bool]:
        """Script a hit at accepted-count k if the mirror's top candidate leads
        the fork-set boundary decisively (kernel-drift-robust); otherwise fall
        back to a scripted miss so hit/miss stays exactly assertable."""
        assert self.mirror.fork_sets is not None, "no cache yet (round 0)"
        if self.mirror.fork_margins.get(k, 0.0) >= self.MARGIN_TAU:
            return self.mirror.fork_sets[k][0], True
        return miss_token, False


def _drive_round(session, seqs: list[ScriptedSeq], outcomes: list[tuple],
                 order: list[int] | None = None):
    """Send one batched spec request for `seqs` with the given (k, rec, ...) per
    seq. Returns (tokens, hits, logits) row-aligned with the given order."""
    order = order or list(range(len(seqs)))
    fields = []
    for i in order:
        k, rec = outcomes[i][0], outcomes[i][1]
        fields.append((seqs[i], seqs[i].round_fields(k, rec)))
    B = len(fields)
    max_blocks = MAX_BLOCKS
    req = dict(
        cache_keys=torch.stack([f["cache_keys"] for _, f in fields]),
        num_tokens=torch.tensor([f["num_tokens"] for _, f in fields], dtype=torch.int64),
        block_tables=torch.stack([s.block_table(max_blocks) for s, _ in fields]),
        extend_counts=torch.tensor([f["extend_counts"] for _, f in fields], dtype=torch.int64),
        extend_token_ids=torch.stack([f["extend_token_ids"] for _, f in fields]),
        extend_activations=torch.stack([f["extend_activations"] for _, f in fields]),
    )
    resp = session.spec_round(**req)
    toks = resp.speculations.view(B, K).cpu()
    hits = resp.cache_hits.cpu().tolist()
    logits = resp.logits_q.cpu().clone()

    # universal self-consistency: served tokens are the argmax of served logits
    finite = torch.isfinite(logits)
    assert finite.any(dim=-1).all(), "response logits row entirely non-finite"
    am = logits.argmax(dim=-1)
    for b in range(B):
        for j in range(K):
            assert int(am[b, j]) == int(toks[b, j]), (
                f"row {b} depth {j}: argmax(logits)={int(am[b, j])} != token {int(toks[b, j])}"
            )

    # advance driver + mirror state
    for pos, (s, f) in enumerate(fields):
        row = f["row"]
        row.cache_hit = hits[pos]
        _, entry = s.mirror.serve(
            None if f["cache_keys"][1] == -2 else int(f["cache_keys"][1]), row.rec_token
        )
        s.last_response_tokens = toks[pos].tolist()
        s.last_hit = bool(hits[pos])
        s.mirror.advance(row, s.last_response_tokens, entry)
    return toks, hits, logits


@pytest.fixture(scope="module")
def eagle_session():
    cfg = build_target_config(
        target_path=require_8b_target(), draft_path=require_eagle_llama_8b_draft(),
        K=K, fanout=FANOUT, mode="eagle", backup="fast",
        max_num_seqs=MAX_SEQS, max_model_len=2048, block_size=BLOCK,
        enforce_eager=False,
    )
    sess = DraftSession(cfg, target_device="cuda:0")
    # HF mirror draft on the target-side GPU
    from transformers import AutoModelForCausalLM
    from tests.hf.eagle3_hf import load_eagle3_specforge

    tgt = AutoModelForCausalLM.from_pretrained(require_8b_target(), torch_dtype=torch.bfloat16)
    draft = load_eagle3_specforge(
        require_eagle_llama_8b_draft(), tgt.model.embed_tokens.weight,
        tgt.config.hidden_size, "cuda:0", dtype=torch.bfloat16,
    ).eval()
    tgt = tgt.to("cuda:0").eval()  # real conditioning acts for the scripted chains
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(require_8b_target())
    yield SimpleNamespace(session=sess, draft=draft, target=tgt, tokenizer=tok)
    sess.close()


_SEQ_COUNTER = [0]


def _fresh_seqs(bank, n: int, tag_base: int):
    out = []
    for i in range(n):
        _SEQ_COUNTER[0] += 1
        out.append(ScriptedSeq(
            10_000 + _SEQ_COUNTER[0], tag_base + i, bank.draft, bank.session,
            bank.target, bank.tokenizer,
        ))
    # Prefill the draft with the group's prompts + dup-shifted target acts —
    # without this the glue decode attends unwritten prompt KV.
    input_ids = torch.tensor([t for s in out for t in s.prompt], dtype=torch.int64)
    num_tokens = torch.tensor([len(s.prompt) for s in out], dtype=torch.int64)
    block_tables = torch.stack([s.block_table(MAX_BLOCKS) for s in out])
    eagle_acts = torch.cat([s._dup_acts[: len(s.prompt)].float() for s in out])
    bank.session.send_prefill(input_ids, num_tokens, block_tables, eagle_acts, MAX_BLOCKS)
    return out


def _scripted_schedule(seqs, r: int):
    """Round r outcomes per seq as (k, rec, expected_hit).

    Hits are scripted ONLY at k=0: the k=0 fork position is chain-free (the
    recovery token is conditioned on a target activation and its glue row
    attends only the target-conditioned prefix), so an independent mirror can
    predict the engine's candidate set robustly (margin-aware). Fork positions
    k>=1 are conditioned on the engine's private recurrence chain and are NOT
    predictable from outside — those ks are exercised via scripted MISSES
    (which still drive heterogeneous extend_counts through the varlen glue
    path); their hit-path content is covered by the bit-exact permutation
    equivariance test and the batched server test."""
    outs = []
    for i, s in enumerate(seqs):
        if r == 0:
            outs.append((None, 777 + i, False))
        elif (r + i) % 3 == 2:
            outs.append((min((r + i) % (K + 1), K), MISS_TOKEN + i, False))  # scripted miss, varied k
        else:
            rec, is_hit = s.hit_or_miss(0, MISS_TOKEN + i)
            outs.append((0, rec, is_hit))
    return outs


@requires_2gpus
@pytest.mark.parametrize(
    "B,tag_base", [(1, 100), (1, 200), (2, 200), (4, 400)],
    ids=["B1", "B1-altprompts", "B2", "B4"],
)
def test_scripted_outcomes_hit_miss_exact(eagle_session, B, tag_base):
    """Heterogeneous per-row (k, hit/miss) schedules: the engine's cache_hits
    must exactly follow the script — hits at max-margin fork candidates, misses
    at far-out tokens — across mixed batches (varlen glue path at B>1).
    B1-altprompts uses the same prompts as B2, so a B2-only failure isolates a
    batch effect from prompt-content margin fragility."""
    seqs = _fresh_seqs(eagle_session, B, tag_base=tag_base)
    n_hits_scripted = 0
    for r in range(8):
        outs = _scripted_schedule(seqs, r)
        toks, hits, logits = _drive_round(eagle_session.session, seqs, outs)
        for i, s in enumerate(seqs):
            k, rec, expected_hit = outs[i]
            n_hits_scripted += int(expected_hit)
            if r == 0:
                assert hits[i] == 0, "round 0 must be a miss (sentinel)"
                continue
            assert hits[i] == int(expected_hit), (
                f"round {r} seq {i}: cache_hit={hits[i]} but scripted "
                f"{'hit' if expected_hit else 'miss'} (k={k}, rec={rec}; mirror "
                f"fork_set[k]={s.mirror.fork_sets.get(k) if s.mirror.fork_sets else None}, "
                f"margin={s.mirror.fork_margins.get(k) if s.mirror.fork_sets else None}, "
                f"engine_toks={toks[i].tolist()})"
            )
    assert n_hits_scripted >= 3 * B, (
        f"schedule degenerated to misses ({n_hits_scripted} hits scripted) — "
        f"margins too thin to test the hit path; loosen MARGIN_TAU or fix prompts"
    )


@requires_2gpus
def test_hits_track_mirror_content(eagle_session):
    """Scripted all-hit run at B=2: engine hit-row tokens should track the
    mirror's branch tokens (exact or low-rank; statistical due to chain)."""
    seqs = _fresh_seqs(eagle_session, 2, tag_base=900)
    stats = SimpleNamespace(tok=0, exact=0, rank_ok=0)
    for r in range(8):
        outs = []
        for i, s in enumerate(seqs):
            if r == 0:
                outs.append((None, 555 + i, False))
            else:
                rec, is_hit = s.hit_or_miss(0, MISS_TOKEN + i)  # k=0: chain-free fork position
                outs.append((0, rec, is_hit))
        # capture mirror entries BEFORE advance rotates them
        pre_entries = []
        for i, s in enumerate(seqs):
            k, rec, is_hit = outs[i]
            pre_entries.append(s.mirror.cache.get((k, rec)) if is_hit else None)
        toks, hits, logits = _drive_round(eagle_session.session, seqs, outs)
        for i, s in enumerate(seqs):
            if pre_entries[i] is None:
                continue
            assert hits[i] == 1
            m_toks, m_logits, _ = pre_entries[i]
            for j in range(K):
                stats.tok += 1
                if int(toks[i, j]) == m_toks[j]:
                    stats.exact += 1
                    stats.rank_ok += 1
                else:
                    lm = m_logits[j]
                    rank = int((lm > lm[int(toks[i, j])]).sum())
                    stats.rank_ok += int(rank <= 8)
    assert stats.tok > 0
    assert stats.exact / stats.tok >= 0.6, f"only {stats.exact}/{stats.tok} exact vs mirror"
    assert stats.rank_ok / stats.tok >= 0.9, f"only {stats.rank_ok}/{stats.tok} rank<=8 vs mirror"


@requires_2gpus
def test_row_permutation_equivariance(eagle_session):
    """Identical scripts under fresh seq ids, rows permuted: responses must be
    the row-permuted originals, bit-exactly (same N, deterministic kernels).
    Catches cross-row indexing/misalignment bugs in glue/tree/extends."""
    perm = [2, 0, 3, 1]
    results = []
    for run, order in enumerate([None, perm]):
        seqs = _fresh_seqs(eagle_session, 4, tag_base=700)  # same tags => same content
        run_rows = []
        for r in range(5):
            outs = _scripted_schedule(seqs, r)
            toks, hits, logits = _drive_round(eagle_session.session, seqs, outs, order=order)
            inv = list(range(4)) if order is None else order
            # map response rows back to canonical seq index
            by_seq = {}
            for pos, i in enumerate(inv):
                by_seq[i] = (toks[pos].tolist(), hits[pos], logits[pos])
            run_rows.append(by_seq)
        results.append(run_rows)

    for r in range(5):
        for i in range(4):
            t0, h0, l0 = results[0][r][i]
            t1, h1, l1 = results[1][r][i]
            assert h0 == h1, f"round {r} seq {i}: hit differs under permutation"
            assert t0 == t1, f"round {r} seq {i}: tokens differ under permutation: {t0} vs {t1}"
            fin = torch.isfinite(l0) & torch.isfinite(l1)
            assert torch.equal(l0[fin], l1[fin]), f"round {r} seq {i}: logits differ under permutation"


@requires_2gpus
def test_rerun_determinism(eagle_session):
    """Same script twice (fresh seq ids): byte-identical responses. Pins the
    empty-cache zero-fill (uninitialized memory would flake here)."""
    snap = []
    for run in range(2):
        seqs = _fresh_seqs(eagle_session, 3, tag_base=800)
        rows = []
        for r in range(4):
            outs = _scripted_schedule(seqs, r)
            toks, hits, logits = _drive_round(eagle_session.session, seqs, outs)
            rows.append((toks.tolist(), hits))
        snap.append(rows)
    assert snap[0] == snap[1], "identical scripts must produce identical responses"


@requires_2gpus
def test_batch_shrink_and_regrow(eagle_session):
    """B=3 -> B=2 (seq 1 absent) -> B=3: the returning seq must be a guaranteed
    miss (per-round cache reset), present seqs keep serving hits."""
    seqs = _fresh_seqs(eagle_session, 3, tag_base=600)
    _drive_round(eagle_session.session, seqs, [(None, 601 + i, False) for i in range(3)])
    _drive_round(
        eagle_session.session, seqs,
        [(0,) + s.hit_or_miss(0, MISS_TOKEN + i) for i, s in enumerate(seqs)],
    )

    # shrink: only seqs 0 and 2 (k=0 hits: the chain-free scriptable position)
    sub = [seqs[0], seqs[2]]
    outs_sub = [(0,) + s.hit_or_miss(0, MISS_TOKEN + i) for i, s in enumerate(sub)]
    toks, hits, _ = _drive_round(eagle_session.session, sub, outs_sub)
    for i, (_, _, expected) in enumerate(outs_sub):
        assert hits[i] == int(expected)

    # regrow: seq 1 returns claiming an outcome whose top fork candidate WOULD
    # have been cached had it stayed — but the per-round cache reset dropped its
    # entries, so it must miss no matter what (a hit here = stale cache leak).
    stale_candidate = seqs[1].mirror.fork_sets[0][0]
    outs = [
        (0,) + seqs[0].hit_or_miss(0, MISS_TOKEN),
        (0, stale_candidate, False),
        (0,) + seqs[2].hit_or_miss(0, MISS_TOKEN + 2),
    ]
    toks, hits, _ = _drive_round(eagle_session.session, seqs, outs)
    assert hits[1] == 0, "returning seq must miss (its cache entries were reset)"
    for i in (0, 2):
        assert hits[i] == int(outs[i][2]), f"seq {i} hit/miss deviated from script"
