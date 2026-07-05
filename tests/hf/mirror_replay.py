"""Trace-driven HF replica of the eagle/phoenix draft engine (B=1).

Replays a dump trace round by round, maintaining the draft-side conditioning
stream exactly as the engine does:

  - extend positions (accepted tokens)      <- dumped target acts (fc'd for eagle)
  - recovery position                       <- dumped recovery act
  - current-round spec positions            <- previous round's tree prenorms
    (the recurrence chain carried across rounds, NOT a fresh teacher-forced
    recomputation — this is where the engine and the test's one-round-delay
    reconstruction legitimately differ)

Per round it recomputes the glue decode, fork sets (via the production
get_forked_recovery_tokens_from_logits), and the served branch's tree
recurrence, then compares its served tokens/logits with the engine's dumped
response for the NEXT round. Divergence localizes engine bugs; agreement means
the engine faithfully runs its own algorithm (and any test failure is a
reconstruction artifact).

Inputs come solely from the dumps (no 8B target model needed): the dumped
extend activations ARE the engine's ground-truth conditioning inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from tests.hf.trace_reader import SeqTrace, chain_prefixes
from tests.hf.test_ssd_vs_hf_reference import convert_to_full_vocab_logits
from ssd.utils.async_helpers.async_spec_helpers import get_forked_recovery_tokens_from_logits


@dataclass
class RoundReport:
    round_index: int
    engine_hit: int | None
    mirror_hit: bool | None  # None on round 0 (no cache yet)
    served_k: int | None
    token_match: list[int] | None  # engine spec tokens == mirror branch argmax
    engine_token_ranks: list[int] | None  # rank of engine tokens in mirror logits
    max_l1_prob_gap: float | None


class EagleMirror:
    """B=1 replica. draft: eagle3_hf.Eagle3Model or phoenix_hf.PhoenixModel."""

    def __init__(self, draft, prompt_tokens: list[int], prompt_eagle_acts: torch.Tensor,
                 K: int, fan_out_list: list[int], fan_out_list_miss: list[int],
                 phoenix: bool = False):
        self.draft = draft
        self.device = draft.device
        self.dtype = draft.lm_head.weight.dtype
        self.K = K
        self.phoenix = phoenix
        self.cfg = SimpleNamespace(
            speculate_k=K, fan_out_list=fan_out_list, fan_out_list_miss=fan_out_list_miss,
        )
        # prompt acts arrive already dup-shifted from the engine's prefill
        # request (index i conditions token i)
        self.tokens = list(prompt_tokens)
        acts = torch.as_tensor(prompt_eagle_acts, device=self.device, dtype=self.dtype)
        self.cond = self._proj(acts)  # [len(prompt), hidden]
        # per-(k, rec) cache: spec tokens, spec full-vocab logits, spec prenorms
        self.cache: dict[tuple[int, int], tuple[list[int], torch.Tensor, torch.Tensor]] = {}
        self.fork_sets: dict[int, list[int]] | None = None
        self.pending_spec: list[int] = []
        self._row0_key: tuple[int, int] | None = None

    def _proj(self, acts: torch.Tensor) -> torch.Tensor:
        if self.phoenix:
            return acts.to(self.dtype)
        return self.draft.fc(acts.to(self.dtype))

    def _prenorm(self, tokens: list[int], cond: torch.Tensor) -> torch.Tensor:
        ids = torch.tensor(tokens, device=self.device, dtype=torch.long)
        with torch.no_grad():
            return self.draft.forward_with_cond(ids, torch.arange(len(tokens), device=self.device), cond)

    def _logits(self, prenorm_rows: torch.Tensor) -> torch.Tensor:
        logits = self.draft.lm_head(self.draft.norm(prenorm_rows))
        if not self.phoenix:
            logits = convert_to_full_vocab_logits(self.draft, logits)
        return logits

    def serve(self, k: int | None, rec_token: int) -> tuple[bool, tuple | None]:
        """Round-r serve: is (k, rec) in the mirror cache, and its entry."""
        if k is None or not self.cache:
            return False, None
        entry = self.cache.get((k, rec_token))
        return entry is not None, entry

    def advance(self, row, spec_tokens_engine: list[int], mirror_entry) -> None:
        """Consume round r: update conditioning with the dumped extend/recovery
        acts, run the glue decode over [prefix..., rec, spec...], fork, and
        build the next cache. spec_tokens_engine = the tokens the ENGINE
        actually returned this round (the trunk both sides condition on)."""
        K = self.K
        n = row.extend_count or 0
        ext_acts = torch.as_tensor(row.extend_activations, device=self.device, dtype=self.dtype)
        if n > 0:
            ids = row.extend_token_ids[:n].tolist()
            assert ids == self.pending_spec[:n], (
                f"extend tokens {ids} != accepted prefix of previous spec {self.pending_spec[:n]}"
            )
            self.tokens.extend(ids)
            self.cond = torch.cat([self.cond, self._proj(ext_acts[:n])])
        # recovery token, conditioned on the dumped recovery activation
        self.tokens.append(row.rec_token)
        self.cond = torch.cat([self.cond, self._proj(ext_acts[n : n + 1])])
        assert len(self.tokens) == row.num_tokens, (
            f"mirror prefix {len(self.tokens)} != dumped num_tokens {row.num_tokens}"
        )

        # spec positions: conditioned on the serving branch's prenorms (the
        # recurrence chain) — or, if this round was a miss with no mirror
        # entry, on zeros for phoenix-style... engine uses stale cache row 0;
        # for miss rounds we take the mirror's branch (0, first fork) when
        # available, else zeros. (fast mode: engine serves stale row 0 = its
        # (0, first-fork) branch of the previous tree.)
        if mirror_entry is not None:
            prev_prenorms = mirror_entry[2]  # [K, hidden]
        elif self.phoenix:
            prev_prenorms = self.cond[-1:].expand(K, -1)
        else:
            first = self.cache.get(self._row0_key) if self.cache else None
            prev_prenorms = first[2] if first is not None else torch.zeros(
                K, self.cond.shape[1], device=self.device, dtype=self.dtype
            )
        if self.phoenix:
            # phoenix conditions every spec position on the recovery activation
            spec_cond = self.cond[-1:].expand(K, -1).clone()
        else:
            spec_cond = prev_prenorms

        glue_tokens = self.tokens + list(spec_tokens_engine)
        glue_cond = torch.cat([self.cond, spec_cond])
        prenorm = self._prenorm(glue_tokens, glue_cond)
        anchor_rows = prenorm[-(K + 1):]  # rec + K spec positions
        glue_logits = self._logits(anchor_rows)  # [K+1, V]

        # fan layout follows the round's ACTUAL hit/miss status, like the engine
        # (fast-mode miss => all fanout re-allocated to k=0 for the next tree)
        was_hit = bool(row.cache_hit) if row.cache_hit is not None else False
        forks = get_forked_recovery_tokens_from_logits(
            self.cfg,
            glue_logits.unsqueeze(0).float(),
            torch.tensor([1 if was_hit else 0], dtype=torch.int64, device=self.device),
            torch.tensor(
                [[row.rec_token] + list(spec_tokens_engine)], dtype=torch.int64, device=self.device,
            ),
            tokenizer=None,
        )[0]
        fan = self.cfg.fan_out_list if was_hit else self.cfg.fan_out_list_miss
        k_layout = []
        for kk, f in enumerate(fan):
            k_layout += [kk] * f
        self.fork_sets = {}
        for kk, tok in zip(k_layout, forks.tolist()):
            self.fork_sets.setdefault(kk, []).append(tok)

        # tree decode: one branch per fork candidate
        new_cache = {}
        row0_key = None
        for i, (kk, fork_tok) in enumerate(zip(k_layout, forks.tolist())):
            branch_tokens = self.tokens + list(spec_tokens_engine[:kk]) + [fork_tok]
            branch_cond = torch.cat([self.cond, spec_cond[:kk], anchor_rows[kk : kk + 1]])
            if self.phoenix:
                branch_cond[-1] = self.cond[-1]
            spec, logit_rows, prenorm_rows = [], [], []
            for _d in range(self.K):
                pn = self._prenorm(branch_tokens, branch_cond)
                logits = self._logits(pn[-1:])[0]
                nxt = int(logits.argmax())
                spec.append(nxt)
                logit_rows.append(logits)
                prenorm_rows.append(pn[-1])
                branch_tokens = branch_tokens + [nxt]
                nxt_cond = self.cond[-1:] if self.phoenix else pn[-1:]
                branch_cond = torch.cat([branch_cond, nxt_cond])
            new_cache[(kk, fork_tok)] = (
                spec, torch.stack(logit_rows), torch.stack(prenorm_rows),
            )
            if i == 0:
                row0_key = (kk, fork_tok)
        self.cache = new_cache
        self._row0_key = row0_key
        self.pending_spec = list(spec_tokens_engine)


def replay(trace: SeqTrace, prompt: list[int], draft, K: int,
           fan_out_list: list[int], fan_out_list_miss: list[int],
           phoenix: bool = False, verbose: bool = True) -> list[RoundReport]:
    mirror = EagleMirror(
        draft, prompt, trace.prompt_eagle_acts, K, fan_out_list, fan_out_list_miss, phoenix=phoenix,
    )
    chain_prefixes(trace, prompt)  # validates the dump chain
    reports = []
    for r, row in enumerate(trace.rows):
        mirror_hit, entry = mirror.serve(row.k_prev, row.rec_token)
        rep = RoundReport(r, row.cache_hit, None if r == 0 else mirror_hit,
                          row.k_prev, None, None, None)
        if r > 0 and entry is not None:
            spec_m, logits_m, _ = entry
            rep.token_match = [int(a == b) for a, b in zip(row.spec_tokens, spec_m)]
            rep.engine_token_ranks = [
                int((logits_m[j] > logits_m[j][t]).sum()) for j, t in enumerate(row.spec_tokens)
            ]
            if row.logits is not None:
                pe = torch.softmax(row.logits.to(torch.float32), dim=-1)
                pm = torch.softmax(logits_m.cpu().to(torch.float32), dim=-1)
                rep.max_l1_prob_gap = float((pe - pm).abs().sum(-1).max())
        reports.append(rep)
        if verbose:
            print(f"[{r:3d}] eng_hit={row.cache_hit} mir_hit={rep.mirror_hit} k={row.k_prev} "
                  f"match={rep.token_match} ranks={rep.engine_token_ranks} "
                  f"l1={None if rep.max_l1_prob_gap is None else round(rep.max_l1_prob_gap, 3)}",
                  flush=True)
        mirror.advance(row, row.spec_tokens, entry)
    return reports
