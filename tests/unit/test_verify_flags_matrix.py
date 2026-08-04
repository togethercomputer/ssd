"""Tier 0: verify() contract across the (temperature x communicate_* flags) matrix.

The async engine can be configured with communicate_logits / communicate_cache_hits
off, in which case verify() receives logits_q=None / cache_hits=None. At temp=0
that is fine (pure greedy). At temp>0 it used to be a landmine:

- cache_hits=None (jit off): ratio_rows silently became all-False and every row
  fell back to GREEDY acceptance — not the target's sampling distribution, so
  losslessness was silently broken.
- logits_q=None with any ratio row: crashed deep inside with a TypeError.

verify() now raises an explicit ValueError for both misconfigurations; these
tests pin that contract plus row-wise behavior on mixed batches.
"""
from __future__ import annotations

import pytest
import torch

from ssd.utils.verify import verify

pytestmark = pytest.mark.tier0

V = 50


def _one_hot_logits(tokens: list[int], scale: float = 200.0) -> torch.Tensor:
    """[len(tokens), V] logits that are effectively a delta at each token even
    after softmax at temp<=1 (gap 200 => e^-200 rounds to 0 in fp32)."""
    out = torch.zeros(len(tokens), V)
    for i, t in enumerate(tokens):
        out[i, t] = scale
    return out


def _mk_inputs(rows: list[dict], K: int = 2):
    """Each row dict: draft (K tokens), target (K+1 argmax tokens), prev_rec,
    temp_t, temp_q, hit."""
    B = len(rows)
    logits_p = torch.stack([_one_hot_logits(r["target"]) for r in rows])  # [B, K+1, V]
    logits_q = torch.stack([_one_hot_logits(r["draft"]) for r in rows])  # [B, K, V]
    speculations = torch.tensor([[r["prev_rec"]] + r["draft"] for r in rows], dtype=torch.int64)
    temps_t = torch.tensor([r["temp_t"] for r in rows], dtype=torch.float32)
    temps_q = torch.tensor([r["temp_q"] for r in rows], dtype=torch.float32)
    cache_hits = torch.tensor([r["hit"] for r in rows], dtype=torch.int64)
    return logits_p, logits_q, speculations, temps_t, temps_q, cache_hits


def test_temp_gt0_without_cache_hits_raises():
    lp, lq, spec, tt, tq, _ = _mk_inputs(
        [dict(draft=[1, 2], target=[1, 2, 3], prev_rec=9, temp_t=0.7, temp_q=0.7, hit=1)]
    )
    with pytest.raises(ValueError, match="communicate_cache_hits"):
        verify(lp, lq, spec, tt, tq, cache_hits=None, jit_speculate=False)


def test_temp_gt0_ratio_rows_without_logits_q_raises():
    lp, lq, spec, tt, tq, ch = _mk_inputs(
        [dict(draft=[1, 2], target=[1, 2, 3], prev_rec=9, temp_t=0.7, temp_q=0.7, hit=1)]
    )
    with pytest.raises(ValueError, match="communicate_logits"):
        verify(lp, None, spec, tt, tq, cache_hits=ch, jit_speculate=False)
    # jit mode makes every temp>0 row a ratio row — same requirement
    with pytest.raises(ValueError, match="communicate_logits"):
        verify(lp, None, spec, tt, tq, cache_hits=None, jit_speculate=True)


def test_temp_gt0_all_miss_rows_do_not_need_logits_q():
    """Misses fall back to greedy acceptance + recovery sampled from p; the
    draft distribution q is not consulted, so logits_q=None must be fine."""
    lp, _, spec, tt, tq, ch = _mk_inputs(
        [dict(draft=[1, 2], target=[1, 2, 3], prev_rec=9, temp_t=0.7, temp_q=0.0, hit=0)]
    )
    suffixes, recs = verify(lp, None, spec, tt, tq, cache_hits=ch, jit_speculate=False)
    # deterministic despite temp>0: p is (numerically) a delta at each position
    assert suffixes == [[9, 1, 2]]
    assert recs == [3]


def test_greedy_rows_never_need_q_or_hits():
    lp, _, spec, tt, tq, _ = _mk_inputs(
        [
            dict(draft=[1, 2], target=[1, 2, 3], prev_rec=9, temp_t=0.0, temp_q=0.0, hit=0),
            dict(draft=[4, 5], target=[4, 7, 8], prev_rec=9, temp_t=0.0, temp_q=0.0, hit=0),
        ]
    )
    suffixes, recs = verify(lp, None, spec, tt, tq, cache_hits=None, jit_speculate=False)
    assert suffixes == [[9, 1, 2], [9, 4]]
    assert recs == [3, 7]


def test_mixed_temp_rows_rowwise_semantics():
    """4-row batch mixing greedy / sampled-hit / sampled-miss / greedy-hit.
    All distributions are numerical deltas so temp>0 behavior is deterministic:
    p==q on the hit row => accept prob 1 everywhere; miss row uses greedy
    acceptance; recovery always the target's (delta) choice."""
    rows = [
        dict(draft=[1, 2], target=[1, 2, 3], prev_rec=9, temp_t=0.0, temp_q=0.0, hit=0),  # greedy full accept
        dict(draft=[4, 5], target=[4, 5, 6], prev_rec=9, temp_t=0.8, temp_q=0.8, hit=1),  # ratio, p==q, full accept
        dict(draft=[7, 8], target=[7, 3, 2], prev_rec=9, temp_t=0.8, temp_q=0.0, hit=0),  # miss: greedy, reject at 1
        dict(draft=[5, 6], target=[5, 1, 0], prev_rec=9, temp_t=0.0, temp_q=0.0, hit=1),  # greedy, reject at 1
    ]
    lp, lq, spec, tt, tq, ch = _mk_inputs(rows)
    suffixes, recs = verify(lp, lq, spec, tt, tq, cache_hits=ch, jit_speculate=False)
    assert suffixes[0] == [9, 1, 2] and recs[0] == 3
    assert suffixes[1] == [9, 4, 5] and recs[1] == 6
    assert suffixes[2] == [9, 7] and recs[2] == 3
    assert suffixes[3] == [9, 5] and recs[3] == 1


def test_jit_speculate_uses_ratio_regardless_of_cache_hits():
    """Under jit, every row's tokens came from q, so ratio acceptance applies
    even to rows flagged miss; with p==q that means full acceptance (a greedy
    fallback would also accept here, so distinguish via q!=p argmax: token
    accepted by ratio (p==q on the token) but not equal to argmax(p))."""
    # p: token 4 has slightly less mass than token 0 => argmax(p)=0, but
    # q(4)=p(4) so ratio accepts token 4 with prob 1... use exact overlap:
    # p = q = uniform over {4, 0} at position 0 -> accept prob p/q = 1.
    torch.manual_seed(0)  # rand==0.0 exactly would accept a 0-probability token
    lp = torch.zeros(1, 3, V)
    lp[0, 0, 4] = 5.0
    lp[0, 0, 0] = 5.0  # p: 50/50 over {0,4} at pos 0
    lp[0, 1, 6] = 200.0  # delta at 6 (recovery if rejected at 1)
    lp[0, 2, 7] = 200.0
    lq = torch.zeros(1, 2, V)
    lq[0, 0, 4] = 5.0
    lq[0, 0, 0] = 5.0  # q == p at pos 0
    lq[0, 1, 9] = 200.0  # q delta at 9, but draft proposed 8: q(8)=~0 -> ratio rejects
    spec = torch.tensor([[9, 4, 8]], dtype=torch.int64)
    tt = torch.tensor([0.8])
    tq = torch.tensor([0.8])
    suffixes, recs = verify(lp, lq, spec, tt, tq, cache_hits=torch.tensor([0]), jit_speculate=True)
    # token 4 accepted via ratio (p==q), token 8 rejected (q(8)=0 -> wait, p(8)/q(8) clamps to 1)...
    # p(8) is ~0 and q(8) ~0: accept prob = (0/(0+1e-10)) = 0 -> rejected.
    assert suffixes == [[9, 4]]
    # recovery from residual max(p-q, 0) at position 1: p=delta(6), q=delta(9) -> residual=delta(6)
    assert recs == [6]


def test_row_permutation_equivariance_greedy():
    rows = [
        dict(draft=[1, 2], target=[1, 2, 3], prev_rec=9, temp_t=0.0, temp_q=0.0, hit=1),
        dict(draft=[4, 5], target=[4, 7, 8], prev_rec=10, temp_t=0.0, temp_q=0.0, hit=0),
        dict(draft=[6, 6], target=[0, 1, 2], prev_rec=11, temp_t=0.0, temp_q=0.0, hit=1),
    ]
    lp, lq, spec, tt, tq, ch = _mk_inputs(rows)
    base_suf, base_rec = verify(lp, lq, spec, tt, tq, cache_hits=ch, jit_speculate=False)
    perm = [2, 0, 1]
    p_suf, p_rec = verify(
        lp[perm], lq[perm], spec[perm], tt[perm], tq[perm],
        cache_hits=ch[perm], jit_speculate=False,
    )
    assert p_suf == [base_suf[i] for i in perm]
    assert p_rec == [base_rec[i] for i in perm]
