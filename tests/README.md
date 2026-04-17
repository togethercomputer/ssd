# SSD testbed

See `ssd_test_plan_cc.md` for the full plan, invariant list, and tier definitions.
This README is the how-to-run quick reference.

## Running

```bash
# Activate the SSD env.
source /work/avner/git/ssd-phnx/.venv/bin/activate

# Fast subset (tier 0 + smoke): ~1-2 min on H100. Intended for per-commit CI.
./tests/run_fast.sh

# Full tier 0+1: ~8-10 min on H100.
./tests/run_tier1.sh

# Ad-hoc:
pytest tests/unit -m tier0              # CPU unit tests only
pytest tests/e2e -m tier1               # all tier 1
pytest tests -m "tier0 or smoke"        # fast subset
pytest tests/unit/test_verify.py -v     # one file
```

## Current coverage (Tiers 0–1)

| Tier | Invariant | Test file |
|------|-----------|-----------|
| 0 / I8  | `verify()` correctness across branches          | `tests/unit/test_verify.py`             |
| 0 / I9  | mask helpers: cached ≡ vectorized + structure   | `tests/unit/test_mask_helpers.py`       |
| 0 / I10 | BlockManager allocate / deallocate / refcount   | `tests/unit/test_block_manager.py`      |
| 0 / I7  | tree-cache lookup semantics                     | `tests/unit/test_tree_cache_semantics.py` |
| 0 / I11 | handshake pack/unpack round-trip                | `tests/unit/test_handshake_roundtrip.py` |
| 1 / I1  | async+force-jit ≡ no-spec (greedy, 8B)          | `tests/e2e/test_sync_vs_force_jit.py`   |
| 1 / I2  | force-jit ≡ jit ≡ fast (greedy, 8B)             | `tests/e2e/test_greedy_strategy_equivalence.py` |
| 1 / I3  | cudagraph ≡ eager (greedy, 8B)                  | `tests/e2e/test_cudagraph_vs_eager.py`  |
| 1 / I4  | batch position independence                     | `tests/e2e/test_batch_independence.py`  |
| 1 / I5  | duplicate-prompt prefix-cache correctness       | `tests/e2e/test_prefix_cache.py`        |
| 1 / I6  | preemption round-trip                           | `tests/e2e/test_preemption.py`          |

Tiers 2–5 (HF reference, SSD↔TGL fixtures, 70B TP=4, perf regression) are
scoped out of this pass; see plan for details.

## Environment

- SSD uses `/work/avner/git/ssd-phnx/.venv` (managed by uv).
- Tier 1 tests assume model snapshots under `/scratch/avner/huggingface/hub/`
  — specifically:
  - target: `models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249...`
  - draft:  `models--meta-llama--Llama-3.2-1B-Instruct/snapshots/921317...`
  - (Tests auto-skip if a required snapshot is missing.)

## Implementation notes

- Tier 1 tests run each LLM config in a fresh subprocess via `tests/e2e/_runner.py`.
  This is necessary because `LLMEngine.exit` calls `os._exit(0)` during teardown;
  running two LLM instances inside one pytest process would kill the test runner.
- Tier 0 tests run in-process and do not allocate any CUDA memory.

## Known issue / next steps

- **Sync-spec (`draft_async=False`) crashes at draft-model load** on the
  `cc/sglang-fa4` branch: `AttributeError: ModuleList has no attribute '20'`
  — the draft model loader appears to use target-layer indices to traverse the
  draft model. I1 was therefore pivoted to compare `async+force-jit` against
  `no-spec` (greedy output must match), which is an equally strong correctness
  property. When sync-spec is fixed, a direct sync-vs-async test can be added
  to `test_sync_vs_force_jit.py`.
