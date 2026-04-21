"""Ad-hoc script: quantify how far sync-spec and async+force-jit traces diverge."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.e2e._helpers import (  # noqa: E402
    CANONICAL_PROMPTS, base_config, require_1b_draft, require_8b_target, run_llm_subprocess,
)


def _per_seq(trace):
    id_map: dict[int, int] = {}
    out: dict[int, list] = {}
    for step in trace:
        for sid, suf, rec in step:
            if sid not in id_map:
                id_map[sid] = len(id_map)
                out[id_map[sid]] = []
            out[id_map[sid]].append((list(suf), int(rec)))
    return out


def main():
    target, draft = require_8b_target(), require_1b_draft()
    prompts = CANONICAL_PROMPTS
    common = dict(speculate=True, speculate_k=2, enforce_eager=True, max_new_tokens=16)

    sync_cfg = {**base_config(prompts), "model": target, "draft": draft,
                "draft_async": False, "num_gpus": 1, **common}
    async_cfg = {**base_config(prompts), "model": target, "draft": draft,
                 "draft_async": True, "force_jit_speculate": True, "jit_speculate": True,
                 "async_fan_out": 2, "num_gpus": 2, **common}

    sync = run_llm_subprocess(sync_cfg, trace_accepts=True)
    asn = run_llm_subprocess(async_cfg, trace_accepts=True)

    a = _per_seq(sync["per_step_accepts"])
    b = _per_seq(asn["per_step_accepts"])

    print(f"final token streams equal: {sync['token_ids'] == asn['token_ids']}")
    print()

    for seq_idx in sorted(a.keys()):
        ta, tb = a[seq_idx], b[seq_idx]
        print(f"=== seq #{seq_idx} ===")
        print(f"  sync  steps: {len(ta)}, async steps: {len(tb)}")

        def stats(trace):
            drafts_per_step = [len(suf) - 1 for suf, _ in trace]
            total_drafts = sum(drafts_per_step)
            completions = total_drafts + len(trace)  # each step adds drafts + 1 recovery
            proposals = len(trace) * 2  # speculate_k=2 draft proposals per step
            return drafts_per_step, total_drafts, completions, proposals

        sda, tda, coma, pra = stats(ta)
        sdb, tdb, comb, prb = stats(tb)

        print(f"  sync  drafts accepted per step: {sda}  (total {tda}/{pra} = {tda/pra:.1%})")
        print(f"  async drafts accepted per step: {sdb}  (total {tdb}/{prb} = {tdb/prb:.1%})")
        print(f"  sync  completion tokens (drafts+recoveries): {coma}")
        print(f"  async completion tokens: {comb}")

        # How many of the sync-trace (suffix, recovery) pairs also appear in async trace?
        common = set(map(lambda x: (tuple(x[0]), x[1]), ta)) & set(map(lambda x: (tuple(x[0]), x[1]), tb))
        print(f"  shared (suffix, recovery) pairs: {len(common)} "
              f"(sync unique={len(ta) - len(common)}, async unique={len(tb) - len(common)})")

        # Recovery tokens alone — match the actual per-recovery token trace.
        sync_recs = [r for _, r in ta]
        asn_recs = [r for _, r in tb]
        print(f"  recovery tokens equal (as sequence)? {sync_recs == asn_recs}")

        # If recovery sequences are subsequences of each other (async = sync with extras)
        if len(sync_recs) <= len(asn_recs):
            shorter, longer = sync_recs, asn_recs
            label = "sync subseq of async"
        else:
            shorter, longer = asn_recs, sync_recs
            label = "async subseq of sync"
        def is_subseq(s, l):
            it = iter(l)
            return all(any(x == y for y in it) for x in s)
        print(f"  {label}: {is_subseq(shorter, longer)}")
        print()


if __name__ == "__main__":
    main()
