"""Tier 0 / I8: correctness of ssd.utils.verify.verify across branches.

Branches exercised:
- greedy only (temps_t=0, temps_q=0)
- target-sampled, draft-greedy (temp_t>0, temp_q=0) — goes through sampling branch
- both sampled, cache hit (ratio acceptance)
- both sampled, cache miss (falls back to greedy when jit_speculate=False)
- jit_speculate=True uses ratio acceptance regardless of cache_hits

verify() lives in /work/avner/git/ssd-phnx/ssd/utils/verify.py and is pure
(tensors in, tensors out), so no GPU / no model weights are needed.
"""
from __future__ import annotations

import pytest
import torch

from ssd.utils.verify import verify

pytestmark = pytest.mark.tier0


# ---------------------------------------------------------------------------
# Oracle: pure-python re-implementation of the greedy-only branch.
# ---------------------------------------------------------------------------
def _greedy_oracle(
    logits_p: torch.Tensor,
    speculations: torch.Tensor,
) -> tuple[list[list[int]], list[int]]:
    """Pure-python greedy verify, ignoring logits_q.

    accepted_suffix[b] = [starts[b]] + draft_tokens[b, :accept_count[b]]
    accept_count is the number of leading draft tokens equal to the target's argmax.
    recovery token is target argmax at position accept_count.
    """
    B, Kp1, _V = logits_p.shape
    K = Kp1 - 1
    starts = speculations[:, 0].tolist()
    draft = speculations[:, 1:]
    preds_p = logits_p.argmax(dim=-1)  # [B, K+1]

    accepted_suffixes: list[list[int]] = []
    recovery: list[int] = []
    for b in range(B):
        n = 0
        for j in range(K):
            if int(draft[b, j].item()) == int(preds_p[b, j].item()):
                n += 1
            else:
                break
        suffix = [starts[b]] + draft[b, :n].tolist()
        accepted_suffixes.append(suffix)
        recovery.append(int(preds_p[b, n].item()))
    return accepted_suffixes, recovery


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _peaked_logits(B: int, Kp1: int, V: int, token_ids: torch.Tensor, peak: float = 50.0) -> torch.Tensor:
    """Build logits where token_ids[b, i] is the clear argmax on row (b, i)."""
    assert token_ids.shape == (B, Kp1)
    logits = torch.randn(B, Kp1, V) * 0.01
    logits.scatter_(2, token_ids.unsqueeze(-1), peak)
    return logits


# ---------------------------------------------------------------------------
# Greedy tests
# ---------------------------------------------------------------------------
class TestGreedy:
    """temp_t == 0, temp_q == 0: pure argmax compare."""

    @pytest.mark.parametrize("K", [1, 3, 6])
    def test_all_accept(self, K):
        """Draft matches target's argmax at every position → accept all K."""
        torch.manual_seed(0)
        B, V = 4, 64
        # Target's argmax on each (b, i) — pick any legal vocab ids
        target_argmax = torch.randint(0, V, (B, K + 1))
        logits_p = _peaked_logits(B, K + 1, V, target_argmax)
        # Draft proposes exactly the same tokens as target argmax (offset by 1 — starts token takes index 0)
        starts = torch.randint(0, V, (B,))
        speculations = torch.empty(B, K + 1, dtype=torch.int64)
        speculations[:, 0] = starts
        speculations[:, 1:] = target_argmax[:, :K]

        logits_q = torch.randn(B, K, V)  # unused in greedy
        temps_t = torch.zeros(B)
        temps_q = torch.zeros(B)

        got = verify(logits_p, logits_q, speculations, temps_t, temps_q)
        expect = _greedy_oracle(logits_p, speculations)
        assert got == expect
        # Each suffix is len K+1 (starts + K accepted)
        for s in got[0]:
            assert len(s) == K + 1

    def test_first_mismatch_rejects_rest(self):
        """If the draft mismatches at position j, we accept j and recovery = target argmax at j."""
        B, K, V = 2, 4, 32
        torch.manual_seed(1)
        target_argmax = torch.tensor([
            [10, 11, 12, 13, 14],
            [20, 21, 22, 23, 24],
        ], dtype=torch.int64)
        logits_p = _peaked_logits(B, K + 1, V, target_argmax)

        # Draft matches at j=0 and j=1 for seq 0 (so accept 2, recovery = 12),
        # and matches at j=0 only for seq 1 (accept 1, recovery = 21).
        speculations = torch.tensor([
            [99, 10, 11, 0, 0],    # mismatch at j=2 (draft=0, target=12)
            [88, 20, 999, 0, 0],   # mismatch at j=1 (draft=999, target=21)
        ], dtype=torch.int64)

        logits_q = torch.randn(B, K, V)
        suffixes, recovery = verify(logits_p, logits_q, speculations, torch.zeros(B), torch.zeros(B))

        assert suffixes[0] == [99, 10, 11]
        assert suffixes[1] == [88, 20]
        assert recovery[0] == 12
        assert recovery[1] == 21

    def test_no_accepts(self):
        """First draft token mismatches — accept 0, recovery = target argmax at 0."""
        B, K, V = 2, 3, 32
        target_argmax = torch.tensor([
            [5, 6, 7, 8],
            [15, 16, 17, 18],
        ], dtype=torch.int64)
        logits_p = _peaked_logits(B, K + 1, V, target_argmax)
        speculations = torch.tensor([
            [100, 999, 999, 999],
            [200, 999, 999, 999],
        ], dtype=torch.int64)
        logits_q = torch.randn(B, K, V)
        suffixes, recovery = verify(logits_p, logits_q, speculations, torch.zeros(B), torch.zeros(B))
        assert suffixes[0] == [100]  # just the starts token
        assert suffixes[1] == [200]
        assert recovery == [5, 15]


# ---------------------------------------------------------------------------
# Sampled tests — target-sampled, draft-greedy (no ratio branch)
# ---------------------------------------------------------------------------
class TestTargetSampled:
    """temp_t > 0, temp_q == 0, cache_hits=0, jit_speculate=False.

    Acceptance stays greedy (no ratio branch) because cache_hits are all 0
    and jit_speculate=False. But recovery is sampled from p.
    """

    def test_accept_decision_is_greedy_on_miss(self):
        B, K, V = 3, 2, 16
        torch.manual_seed(42)
        target_argmax = torch.tensor([
            [0, 1, 2],
            [5, 6, 7],
            [10, 11, 12],
        ], dtype=torch.int64)
        logits_p = _peaked_logits(B, K + 1, V, target_argmax)
        # All matches → full accept regardless of sampling
        speculations = torch.stack([
            torch.tensor([99, 0, 1]),
            torch.tensor([99, 5, 6]),
            torch.tensor([99, 10, 11]),
        ]).to(torch.int64)

        logits_q = torch.randn(B, K, V)
        temps_t = torch.tensor([1.0, 1.0, 0.0])
        temps_q = torch.zeros(B)
        cache_hits = torch.zeros(B, dtype=torch.int64)  # all misses

        # Run verify three times with different seeds; accept counts must be deterministic.
        for seed in [0, 1, 2]:
            torch.manual_seed(seed)
            suffixes, _recovery = verify(
                logits_p, logits_q, speculations, temps_t, temps_q,
                cache_hits=cache_hits, jit_speculate=False,
            )
            assert [len(s) for s in suffixes] == [K + 1, K + 1, K + 1]


# ---------------------------------------------------------------------------
# jit_speculate=True: ratio acceptance even when cache_hits are zero
# ---------------------------------------------------------------------------
class TestJitSpeculate:
    """jit_speculate=True ignores cache_hits and takes the ratio path when any temp > 0."""

    def test_ratio_branch_is_taken(self):
        """With jit_speculate=True and temps>0 we exercise ratio acceptance code (probabilistic)."""
        B, K, V = 2, 2, 8
        torch.manual_seed(7)
        target_argmax = torch.tensor([
            [0, 1, 2],
            [3, 4, 5],
        ], dtype=torch.int64)
        logits_p = _peaked_logits(B, K + 1, V, target_argmax, peak=5.0)  # less peaked: some prob mass elsewhere
        logits_q = _peaked_logits(B, K, V, target_argmax[:, :K], peak=5.0)

        speculations = torch.stack([
            torch.tensor([99, 0, 1]),
            torch.tensor([99, 3, 4]),
        ]).to(torch.int64)

        temps_t = torch.tensor([1.0, 1.0])
        temps_q = torch.tensor([1.0, 1.0])
        # Key: cache_hits=None + jit_speculate=True → ratio path is active.
        torch.manual_seed(0)
        suffixes, recovery = verify(
            logits_p, logits_q, speculations, temps_t, temps_q,
            cache_hits=None, jit_speculate=True,
        )
        # Sanity: outputs have the right shapes and types (we don't assert exact equality
        # since ratio acceptance samples).
        assert len(suffixes) == B
        assert len(recovery) == B
        for s in suffixes:
            assert 1 <= len(s) <= K + 1


# ---------------------------------------------------------------------------
# Cache-hit gating: jit_speculate=False, some rows hit, some miss
# ---------------------------------------------------------------------------
class TestCacheHitGating:
    """Mixed cache_hits with temps>0 and jit_speculate=False.

    Rows with hit=1 may go through ratio acceptance; rows with hit=0 stay greedy.
    We test this by setting logits such that the greedy decision is a full accept
    for miss rows, and verifying that miss rows always accept fully (irrespective
    of RNG state), while hit rows' accept counts are equal to greedy in the
    specific case where p and q agree (accept prob = 1).
    """

    def test_miss_rows_are_greedy_always(self):
        B, K, V = 4, 3, 16
        torch.manual_seed(11)
        # Target argmax per row
        target_argmax = torch.tensor([
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15],
        ], dtype=torch.int64)
        logits_p = _peaked_logits(B, K + 1, V, target_argmax, peak=50.0)
        # q distribution identical to p for the first K positions → ratio=1 on hit rows
        logits_q = _peaked_logits(B, K, V, target_argmax[:, :K], peak=50.0)

        speculations = torch.empty(B, K + 1, dtype=torch.int64)
        speculations[:, 0] = torch.tensor([100, 200, 300, 400])
        speculations[:, 1:] = target_argmax[:, :K]  # all proposals match argmax

        temps_t = torch.ones(B)
        temps_q = torch.ones(B)
        cache_hits = torch.tensor([1, 0, 1, 0], dtype=torch.int64)

        # With extremely peaked p and q matching p, ratio≈1 always and greedy-on-miss
        # also accepts fully. So all four rows accept K.
        for seed in [0, 1, 2, 3, 4]:
            torch.manual_seed(seed)
            suffixes, _rec = verify(
                logits_p, logits_q, speculations, temps_t, temps_q,
                cache_hits=cache_hits, jit_speculate=False,
            )
            accept_counts = [len(s) - 1 for s in suffixes]
            assert accept_counts == [K, K, K, K]


# ---------------------------------------------------------------------------
# Structural sanity: output shapes/types
# ---------------------------------------------------------------------------
def test_output_shapes_and_types():
    B, K, V = 2, 4, 32
    torch.manual_seed(0)
    logits_p = torch.randn(B, K + 1, V)
    logits_q = torch.randn(B, K, V)
    speculations = torch.randint(0, V, (B, K + 1), dtype=torch.int64)
    suffixes, recovery = verify(logits_p, logits_q, speculations, torch.zeros(B), torch.zeros(B))
    assert isinstance(suffixes, list) and len(suffixes) == B
    assert all(isinstance(s, list) and len(s) >= 1 for s in suffixes)
    assert isinstance(recovery, list) and len(recovery) == B
    assert all(isinstance(r, int) for r in recovery)
