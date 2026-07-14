# SSD test plan (refined)

This is a refinement of `ssd_test_plan.md`. The original plan correctly identifies the properties the SSD async-speculation system should have. This refinement makes those properties **operational** (i.e., testable with precise pass/fail criteria), organizes the tests into **tiers** with clear scope and runtime expectations, and identifies the fixture capture points needed for cross-repo (SSD ↔ TGL) equivalence testing.

## Primary targets under test

- SSD repo: `/work/avner/git/ssd`, branch `avner/sglang-fa4`.
- TGL repo: `/work/avner/git/tgl`, branch `avner/ssd-port`.

All work on these targets is done via the sibling worktree `/work/avner/git/ssd-phnx` (branch `cc/sglang-fa4`) so that in-flight experiments on `avner/sglang-fa4` are not disturbed.

## Key refinements over the original plan

1. **"Identical" is split into two regimes.**
   - *Greedy (temperature == 0)*: bitwise-identical token streams. This is the strict oracle.
   - *Sampled (temperature > 0)*: distributional match — acceptance rate and cache-hit rate within a tolerance over N prompts, RNG-seed controlled.
   Every equivalence claim below specifies which regime applies.

2. **SSD-vs-TGL equivalence is framed at the component level, not end-to-end.** The two systems have different schedulers, different prefill ordering, and different tokenization edges; an end-to-end equivalence requirement would force scheduler changes that are out of scope. Instead, we capture fixtures from one repo and replay them in the other, checking that the algorithmic components (draft-tree contents given fixed inputs, accept-longest-prefix logic given fixed logits) agree exactly.

3. **The HF "naive reference" is scoped narrowly.** HF does not natively do async-speculation, so we do **not** re-implement the async algorithm in HF. Instead:
   - HF is used only as a **ground-truth greedy token oracle** for the target model. Target-greedy output of SSD/TGL must equal HF greedy output token-for-token on short prompts.
   - The spec-algorithm invariants (accept-longest-prefix, ratio-accept with cache-hit gating, tree-mask shapes, etc.) are tested against a **small pure-python oracle** we write inline in the tests — no HF, no weights.

4. **Tests are organized into tiers** based on hardware cost and runtime:

   | Tier | Hardware           | Model   | Typical runtime | What it covers                                                                  |
   |------|--------------------|---------|-----------------|---------------------------------------------------------------------------------|
   | 0    | CPU-only           | none    | seconds         | Pure logic: verify(), block manager, mask helpers, oracles                      |
   | 1    | 1× H100 (or A100)  | 8B      | 1–5 min         | E2E correctness w/ real weights, greedy equivalence between modes               |
   | 2    | 1× H100            | 8B      | 5–15 min        | HF greedy reference match on short prompts                                      |
   | 3    | 1× H100            | 8B      | 1–5 min         | Fixture-based SSD ↔ TGL component equivalence                                   |
   | 4    | 4× H100            | 70B     | 15–60 min       | Same invariants as tiers 1–3 at TP=4                                            |
   | 5    | 1× or 4× H100      | 8B/70B  | 10–30 min       | Performance regression — JSON metrics, baseline comparison, plot generation     |

   **Fast subset** (for per-commit CI) = Tier 0 + one smoke test from Tier 1.
   **This PR implements Tiers 0 and 1.** Tiers 2–5 are tracked but not in scope.

5. **"Identical across draft strategies" (force-jit / jit / fast)** is greedy-only.
   In greedy mode the final token stream is independent of which tokens the draft proposed — the target's argmax always decides. So in greedy mode all three backup strategies must produce the same token stream; only *speed* and *acceptance rate* differ. In sampled mode they will not match token-for-token, and we do not require it.

## Invariants (operationalized)

Each invariant below specifies the precise equality used and the oracle it is checked against.

### I1. `force-jit` ≡ synchronous speculative decoding (greedy)
- **What**: For temperature=0 and fixed prompt, running SSD with `--async --backup force-jit` produces the same token stream as running SSD with `--async=False` (sync spec) using the same speculator.
- **Why it should hold**: `force-jit` always runs the draft synchronously, so the only difference between it and sync spec is the process topology (separate process vs colocated), which must not affect outputs.
- **Tolerance**: Bitwise token match, over a set of canonical prompts.
- **Tier**: 1 (SSD side). TGL side is Tier 4 eventually.

### I2. Greedy token stream independent of backup strategy
- **What**: For temperature=0, `force-jit`, `jit`, and `fast` produce the same output token stream for the same prompts.
- **Tolerance**: Bitwise token match.
- **Tier**: 1.

### I3. CUDA-graph ≡ eager
- **What**: Greedy output with `enforce_eager=True` equals output with CUDA graphs enabled.
- **Tolerance**: Bitwise token match.
- **Tier**: 1.

### I4. Batch independence
- **What**: Greedy output for a prompt is the same whether the prompt is run alone (batch=1) or in a batch at arbitrary position alongside other prompts.
- **Tolerance**: Bitwise token match for the prompt of interest.
- **Tier**: 1.

### I5. Prefix-caching correctness
- **What**: Running a prompt with a shared prefix twice consecutively produces the same output, and the second run reports `num_cached_tokens > 0` for the shared prefix.
- **Tier**: 1.

### I6. Preemption round-trip
- **What**: A sequence that gets preempted (blocks freed, moved back to waiting, re-prefilled) produces the same final output as a sequence that was never preempted. Forced by setting `max_num_seqs` and `num_kvcache_blocks` to a value that guarantees preemption.
- **Tier**: 1.

### I7. Tree-cache invalidation
- **What** (unit): After a sequence's state rolls back (accepted a short suffix, recovery token set), the draft-side tree cache for that `(seq_id, keep_idx, recovery_token)` key must be reused if the same key appears; a different key must miss.
- **Tier**: 0 (tested with a pure-Python model of the tree cache).

### I8. `verify()` correctness against a pure-Python oracle
- **What**: `ssd.utils.verify.verify` produces the expected `(accepted_suffixes, recovery_tokens)` on synthetic logits_p, logits_q, and speculations, for all branches:
  - all-greedy (temp_p=0, temp_q=0)
  - target-sampled, draft-greedy (temp_p>0, temp_q=0)
  - both-sampled, cache hit (ratio acceptance)
  - both-sampled, cache miss (fall back to greedy when `jit_speculate=False`)
  - `jit_speculate=True` uses ratio acceptance regardless of cache hit
- **Tolerance**: Exact for greedy branches; probabilistic match on seed-controlled distribution for ratio branches.
- **Tier**: 0.

### I9. Mask-helper equivalence and structure
- **What**: `get_custom_mask_cached` (B≤8 path) and `get_custom_mask_vectorized` (B>8 path) produce the **same flattened mask** for any given (context_lens, step, K, F, B, fan_out_list, fan_out_list_miss, cache_hits). Separately, the mask shape/semantics match a small reference implementation (`get_mask_iter_i`-style).
- **Tier**: 0.

### I10. Block-manager semantics
- **What**: `BlockManager` allocate/deallocate/may_append correctly:
  - refcount goes to zero → block returns to free pool.
  - shared prefix → `hash_to_block_id` reuse; `num_cached_tokens` reflects reuse.
  - incomplete last block has `hash == -1` and is never put into `hash_to_block_id`.
  - `can_allocate` / `can_append` return false when the pool is empty.
  - draft and target managers are independent.
- **Tier**: 0.

### I11. Handshake pack/unpack round-trip
- **What**: `TargetDraftHandshake.send_request` / `receive_response` tensor shapes and semantics are invertible. We pack a known set of inputs, simulate "wire transfer" by copying to CPU and back, and check that the receiver observes the same values.
- **Tier**: 0 (simulated; no NCCL).

### I12. SSD ↔ TGL fixture-based equivalence
- **What**: Captured inputs `(cache keys, seqs metadata, block tables, target hidden states)` fed into the SSD draft-tree builder produce the same tree as when fed into TGL's draft-tree builder. Captured `(logits_p, logits_q, speculations)` fed into SSD's `verify()` produce the same accept-count and recovery-token decision as TGL's equivalent.
- **Tier**: 3 (out of scope for this PR; we add the fixture-capture hook so the fixture set can be collected when we get to it).

### I13. HF target greedy match
- **What**: `LLM(target_only=True).generate(prompt, temperature=0)` output tokens equal `AutoModelForCausalLM.generate(..., do_sample=False)` output tokens on a small set of short prompts.
- **Tier**: 2 (out of scope for this PR).

### I14. Performance regression
- **What**: For a canonical benchmark config (dataset, batch size, input/output lengths), measured `tokens_per_sec` and per-component `ms` metrics do not regress by more than a threshold (default 5%) vs. a checked-in baseline JSON.
- **Tier**: 5 (out of scope for this PR).

## This PR (Tiers 0 + 1) — concrete test list

### Tier 0 (CPU-only, no model weights)

Files under `tests/unit/`:

- `test_verify.py` — invariant I8. Constructs synthetic logits and speculations, exercises each branch of `ssd.utils.verify.verify`, asserts accepted suffixes and recovery tokens against a pure-Python oracle. Uses fixed `torch.manual_seed` where sampling is involved.
- `test_block_manager.py` — invariant I10. Exercises allocate/deallocate/shared-prefix/may_append/refcount. Tests both `is_draft=False` and `is_draft=True`.
- `test_mask_helpers.py` — invariant I9. For a matrix of `(K, F, B, context_lens, step, fan_out_list, fan_out_list_miss, cache_hits)`, builds the mask via the cached path and the vectorized path and asserts they agree; also checks shape and causal structure against a reference built from `get_mask_iter_i` primitives. Uses CUDA if available; otherwise CPU. Tier 0 runs with CPU.
- `test_tree_cache_semantics.py` — invariant I7. Pure-Python model of the draft's `prev_fork_keys` / cache hit logic. Verifies key matching, rollback invalidation, collision behavior (same seq_id, different recovery_token → miss).
- `test_handshake_roundtrip.py` — invariant I11. Uses `TargetDraftHandshake`-shaped tensor buffers but substitutes NCCL send/recv with in-memory copies to exercise pack/unpack logic and shape contracts.

### Tier 1 (1× H100, 8B, real weights, greedy)

Files under `tests/e2e/`:

- `test_sync_vs_force_jit.py` — I1. Two LLMs with same config, one sync-spec, one async+force-jit; same prompts, temp=0, assert equal token streams.
- `test_greedy_strategy_equivalence.py` — I2. `force-jit`, `jit`, `fast` all produce the same greedy output. Runs three configs in sequence (one LLM at a time to avoid OOM).
- `test_cudagraph_vs_eager.py` — I3. Same config with `enforce_eager=True` vs `False`, assert equal greedy output.
- `test_batch_independence.py` — I4. Prompt P run solo vs run at each position in a batch of N prompts; greedy output of P must match.
- `test_prefix_cache.py` — I5. Run a prompt with a long shared prefix twice; verify second run hits cache (`num_cached_tokens > 0` reported via METRICS) and produces identical output.
- `test_preemption.py` — I6. Configure KV pool such that preemption is guaranteed; verify final outputs equal those of an unpreempted run.

All Tier 1 tests default to a short prompt set (≤5 prompts, ≤128 output tokens) so the whole tier finishes in a few minutes on a single H100.

### Fast subset (per-commit)

All of Tier 0 plus a single smoke test from Tier 1 (`test_sync_vs_force_jit.py::test_two_prompts_greedy`).

Invocation (documented in `tests/README.md`):
```
# fast
pytest tests/unit tests/e2e/test_sync_vs_force_jit.py::test_two_prompts_greedy -m "tier0 or smoke"
# full tier 0+1
pytest tests/unit tests/e2e -m "tier0 or tier1"
```

## Out of scope for this PR (tracked)

- Tier 2 (HF greedy reference).
- Tier 3 (SSD ↔ TGL fixture equivalence). The fixture format and capture hooks will be designed when we get here; they will live in `tests/fixtures/` and be produced by an opt-in flag in each repo's engine.
- Tier 4 (70B TP=4). Requires a 4-GPU host; same invariants as Tiers 1–3.
- Tier 5 (perf regression). Will reuse the output of `bench/extract_metrics.py` and add baseline JSON checked into `tests/perf_baselines/`.
- EAGLE-3 hidden-state specific tests (captured as a Tier 1 follow-up).
- VLM / non-Llama models / TP mismatch between draft and target — explicit non-goals.

## Infrastructure

- **Environments**: SSD tests use `/work/avner/git/ssd-phnx/.venv` (uv-managed). TGL tests (Tier 3+) will use `/work/avner/git/tgl/.venv`.
- **GPU selection**: pytest marker `tier1`/`tier4` auto-skips when `torch.cuda.device_count()` is insufficient. Tier 0 never uses CUDA.
- **Results storage**: Tier 5 metrics JSON lands under `tests/perf_results/<commit-sha>.json` and plots under `tests/perf_results/plots/` (gitignored except for baselines).
- **CI** (proposal for future): GitHub Actions self-hosted runner with 1 H100 runs fast subset + Tier 1 per commit; nightly workflow runs Tiers 2, 3, 5; manual dispatch for Tier 4.

## Open questions for the user

- (none; aligned on scope: Tier 0 + Tier 1 this pass, fixture-based for SSD↔TGL later.)
