# SSD Correctness Campaign — July 2026

Bug hunt + test build-out across the SSD engine (`avner/ssd-correctness`, off
`avner/sglang-fa4-phnx`) and TGL (`avner/ssd-port-correctness`, off
`avner/ssd-port`), focused on Eagle speculators and batch sizes > 1.
All GPU verification ran on research-secure-27/-05 (8×H100 each).

## Bugs found and fixed

### TGL — batch>1 request handling (the big ones)

**T1. `bool(tensor)` crash made async eagle/phoenix batch-1-only.**
`async_spec_worker._draft_forward` computed `is_first_decode = accept_length == -2`;
after the first decode round `accept_length` is a `[B]` tensor, so branching on it
raises *"Boolean value of Tensor with more than one element is ambiguous"* for
B > 1 at the second decode round. B=1 survived via single-element coercion.
Fixed by branching on the sentinel's type; commit `a75b5238e`.

**T2. Eagle activation slice misalignment corrupted every batch row after the first.**
The flat `eagle_acts` / `input_ids` buffers hold `accept_length[i]+1` rows per
request (accepted tokens + the verified-position row — see the `extend_lens`
accounting in `forward_draft_extend_after_decode`), but the read pointer advanced
by only `accept_length[i]`, and not at all for zero-accept requests. Every request
after the first read activations shifted into its predecessor's block — silent
Eagle-conditioning corruption, hidden by a `min()` clamp. Fixed (+ an alignment
assert that makes any future drift loud); commit `a75b5238e`.

**T3. Request join/leave corrupted or crashed async spec state.**
Three related defects around continuous batching (commits `8f196015b`, `9600ceb54`):
- `merge_batch` didn't carry `accept_length`/`accept_length_cpu`, so a freshly
  prefilled request joining a running batch tripped the batch-size assert (or
  misindexed). Now padded with the -1 sentinel (→ forced cache miss on the wire).
- `filter_batch` treated `hidden_states` as `[B, dim]` when between rounds it is
  a RAGGED flat buffer (request i owns `accept_length[i]` rows): whenever a
  request left the batch, request j+1 silently inherited request j's activations.
  Now block-filtered; `accept_length` filtered alongside. When
  `has_been_filtered=True` (the v1 verify path), verify() has ALREADY rebuilt
  spec_info over unfinished requests — re-gathering corrupted it; now skipped
  with size asserts.
- `forward_draft_extend_after_decode` (async override) didn't restore
  `batch.seq_lens`/`seq_lens_cpu`/`req_pool_indices`/`return_logprob` after
  `prepare_extend_after_decode` rebound them to the draft-extend view (which
  excludes finished requests). First finishing request in a batch → scheduler
  `filter_batch` IndexError. The sync worker's backup/restore is now mirrored
  (keeping the +1'd `accept_length` and accepted-chain `input_ids` the async
  wire packing depends on). Also: a joiner's `spec_info.hidden_states` is now
  kept empty after prefill (its prompt acts were already shipped in the
  PrefillRequest), keeping per-request row accounting exact across merges.

**T4. Draft temperatures were never sent** (`# TODO: Set temperatures`): the
draft always speculated greedily regardless of request temperature. Now plumbed
from `sampling_info` — with one crucial subtlety found the hard way: sglang
normalizes greedy requests to `temperature=1.0` + `top_k=1`
(`sampling_params.py`), so the verbatim value made the draft SAMPLE its
speculations for greedy requests (acceptance tanked, every
reconstruction-based check failed with ranks in the hundreds, and for a while
stale `.pyc` bytecode masked the change entirely). `top_k<=1` rows are now
mapped to temperature 0. Commits `8f196015b`, `6f81b7e78`.

**T5. Finished requests desynced batch state (crash on first early finisher).**
`prepare_extend_after_decode` rebinds `batch.seq_lens/seq_lens_cpu/
req_pool_indices` to the draft-extend view, which excludes finished requests;
the sync worker restores them afterwards, the async override didn't. The first
request to finish ahead of its batch left these fields sized to the unfinished
subset while `batch.reqs` stayed full-size → scheduler `filter_batch`
IndexError (found live by the batched server test). Restored like the sync
path — deliberately keeping the +1'd `accept_length` and accepted-chain
`input_ids` the async wire packing depends on. Relatedly,
`filter_batch(has_been_filtered=True)` (the v1 verify path) arrives with
spec_info ALREADY rebuilt over unfinished requests; re-gathering with
old-batch indices corrupted it — now skipped with size asserts. Commit
`9600ceb54`.

### SSD repo — shared engine code

**S1. `verify()` temperature>0 landmines** (`ssd/utils/verify.py`): with
`communicate_cache_hits=False` it silently fell back to greedy acceptance
(losslessness silently broken); with `communicate_logits=False` on ratio rows it
crashed with an opaque TypeError. Both now raise explicit `ValueError`s naming
the config fix. Commit `d7325aa`.

**S2. Uninitialized activations on the empty-cache fast path**
(`draft_runner.hit_cache`): `out_activations = torch.empty(...)` was returned
unfilled on round 1 in fast mode and flowed into the next glue decode as
conditioning — nondeterministic, NaN-able. Now zero-filled. Commit `d7325aa`.

**S3. Latent API bug**: `SpeculationResponse` defines a classmethod `receive`
that is shadowed by the later instance method of the same name (the classmethod
is dead code); callers must use `prepare()` + instance `.receive()`. Documented
in the harness; cleanup optional.

**S4. CUDA-graph ghost-row padding crashed on narrow wire block tables**
(`cudagraph_helpers.run_glue_decode_cudagraph`): padding batch size up to a
graph bucket assigned a wire-width block-table row (TGL sizes tables
dynamically, ~3 wide) into the 32-wide static buffer row without slicing —
broadcast error, draft process death, the first time B needed bucket padding
(e.g. B=3 → bucket 4). Structurally impossible at `max_num_seqs=1`, which is
why every single-request run passed. Commit `651969f`.

**S5. Cache-miss responses leaked state across sequences and sessions.**
Miss rows were filled from tree-cache row 0 ("any consistent tokens are ok").
Two real problems: at B>1 row 0 belongs to a *different sequence* (cross-request
proposal leak), and responses depended on session history — including through
the next round's fork sets, since fork selection excludes the trunk token at
each depth (caught by the scripted suite's rerun-determinism and
row-permutation-equivariance tests). But plain zeros cost ~0.3–0.4 accept
length: row-0 reuse was a *smart fallback* (the sequence's previous k=0
top-fork branch often still fits after a near-miss). Final fix: fall back to
the SAME sequence's first cache row (bit-identical to the old behavior at B=1,
no cross-sequence leak at B>1), deterministic zeros only for sequences with no
cache entries. Commits `4834672`, `0b4cb6f`.

### Test bugs (the reference test was wrong, not the engine)

**X1. Accept-length estimator mismatch** — the test averaged between-round
prefix diffs, which drops the final round; the engine reports
`completion_tokens / num_verify_rounds`. The ~0.04–0.05 systematic gap failed 3
standalone combos at tolerance 0.03 (and motivated a tolerance loosening in a
prior working-tree change). The reconstruction now computes the engine's exact
formula and cross-checks the dump chain against the emitted completion.

**X2. Eagle cache-hit rounds were held to an impossible standard.** The engine
conditions served branches on a private multi-round recurrence chain (tree
prenorms → next glue's spec slots → next tree). The chain is mildly chaotic:
at occasional near-tie rounds ANY external reconstruction ranks the engine's
(correct) tokens badly. We proved the engine right by (a) replaying dumped
request streams through a fresh DraftRunner — bit-exact reproduction of all 63
rounds, in BOTH cudagraph and eager modes; (b) diffing dumped extend/prefill
activations vs HF-computed target activations (~1–3% rel-norm everywhere); and
(c) a faithful HF chain-mirror that explains the "bad" rounds (its worst round
ranks [1,1,4,8] under true chain conditioning). The test now applies the strict
per-token rank bound only to chain-free rounds (force-jit, jit-reconstructed
misses, all phoenix rounds — phoenix has no recurrence) and a statistical bound
(≥90% tokens, ≥80% rounds within rank 4) to eagle hit rounds.

**X3. Simulation fanout mismatch** — `full_ssd_simulation` defaulted to
`fan_out=5` against a fanout-3 engine; now plumbed from the test parameter.

**X4. Teardown masked launch failures** — if `launch_tgl_server` raised, the
`finally` block died on unbound `tgl_server`, hiding the real error.

**X5. Suspected phoenix off-by-one in the sim (B8) — investigated, NOT a bug.**
The sim's `full_target_activations[base_len-1]` is the round-(r-1) recovery
activation, which is exactly what the engine reuses; the suspicion came from a
rounds-offset confusion. Verified against the reconstruction convention that
passes strict thresholds.

**X6. Stale e2e model paths** (`/scratch/avner/...`) made the entire SSD-engine
e2e suite silently skip; now shared with `tests/hf/helpers.py` constants.

### Not-a-bug findings worth knowing

- **Fast-mode misses return stale cache row 0** (not zeros, unless the cache is
  empty). Internally consistent by construction (tokens/logits/activations from
  one row; the glue trunk conditions on those same tokens) and verified by unit
  test. `claude.md`'s "returns zeros" description matches only the empty-cache
  round. Verification handles miss rows with greedy acceptance + recovery from
  p, so losslessness is unaffected.
- **The recurrence chain costs ~0.2 accept length vs force-jit at K=4**
  (2.47 vs 2.68 on the jit run) — algorithm-inherent, not an implementation bug.
- **Engine determinism**: dumps replayed through a fresh DraftRunner reproduce
  responses bit-exactly; cudagraph ≡ eager.

## New test infrastructure

- `tests/unit/` additions (CPU, seconds): fork-token selection semantics,
  `hit_cache`/`_populate_tree_cache` on the production code (mock self),
  glue-layout equivalence (uniform fast path ≡ varlen fallback per sequence,
  mixed hit/miss, eagle+phoenix), `verify()` flag/temperature matrix, and a
  dump round-trip validating the new batch-aware reader.
- `tests/hf/trace_reader.py`: batch-aware dump reader (per-seq splitting keyed
  by `hash_rid`, prefix chaining cross-checked against `num_tokens`,
  engine-comparable accept lengths).
- `tests/hf/mirror_replay.py`: faithful HF replica of the eagle/phoenix draft
  (glue + fork + tree recurrence chain), trace-driven; doubles as the scripted
  suite's oracle.
- `tests/draft_runner/`: scripted-target harness driving a REAL DraftRunner
  over NCCL (2 GPUs, no server): trace replay (bit-exact vs dumps),
  hit/miss-exact scripted schedules (hits at the chain-free k=0 position),
  row-permutation equivariance (bit-exact batch-independence check),
  rerun determinism, batch shrink/regrow with stale-cache probes.
- `tests/hf/test_ssd_vs_hf_reference_batch.py`: N=4 concurrent requests against
  one TGL server; per-sequence completion/reconstruction/accept-length checks +
  batching-honesty gate. `SSD_TRACE_REUSE` skips the server phase when
  iterating on reconstruction.

## Verification status (final code state, fresh bytecode)

- `tests/unit`: **116 passed** (CPU, seconds).
- Full reference matrix (B=1, lookahead 4): **9/9 passed** (standalone/eagle/
  phoenix × fast/jit/force-jit). Acceptance metrics bit-identical to the
  original healthy baseline (e.g. eagle-fast 2.0317 over 63 rounds,
  standalone-force-jit 3.6571 over 35 rounds) — confirming the own-row miss
  fallback reproduces the original engine exactly at B=1.
- Batched server test (TGL, N=4 concurrent): **4/4 combos passed**
  (eagle-fast, eagle-jit, phoenix-fast, standalone-fast) — completions exactly
  HF per request, accept lengths exact, shipped activations clean per slot,
  joins/leaves exercised (B ramps 1→4, early finishers leave). Chain-round
  fractions at B=4 sit at 0.94–1.0 token / 0.77–0.95 round — essentially B=1
  levels once the greedy-temperature bug was fixed (the earlier 0.43–0.71
  readings were sampled-draft artifacts, not batch effects).
- Scripted DraftRunner suite: **8/8 passed** (hit/miss exactness at B=1/2/4,
  mirror content, row-permutation equivariance bit-exact, rerun determinism,
  batch shrink/regrow with stale-cache probe).
- Operational gotchas recorded for future runs: purge `__pycache__` after
  over-NFS edits before remote runs (stale bytecode silently masked an engine
  change); pin every run's GPUs; derive the async store port from the HTTP
  port (fixed 29600 collides with other users' servers on shared nodes).

## The "standalone engine async hang" — resolved (three stacked causes)

The `ssd.LLM` async "hang" decomposed into three independent defects, none of
them in the speculation algorithm (a single-prompt async run completes fine
standalone on a quiet node):

1. **Fixed rendezvous port** (`model_runner.py` hardcoded `tcp://localhost:1223`
   for the default process group): any foreign process holding the port breaks
   startup — a torch-store-speaking squatter absorbs the rendezvous and the
   engine waits forever. LLMEngine now free-port-scans before spawning
   (children inherit via the pickled config); verified to pass with 1223
   deliberately squatted. Commit `a841317`.
2. **The e2e runner never tore the engine down** — it relied on interpreter
   shutdown, where multiprocessing's atexit joins the non-daemon draft child
   whose exit depends on NCCL teardown ordering. Now: explicit
   `llm.exit(hard=False)` + `os._exit(0)` after emitting the result.
3. **`run_llm_subprocess` captured output via pipes**, so `subprocess.run`
   returned only on pipe EOF — held open by any lingering engine grandchild.
   Successful runs surfaced as 600s timeouts. Now: temp-file redirection
   (returns when the runner exits), `start_new_session` + `killpg` on timeout
   and completion (no more orphaned engine families squatting on GPUs).
   Commits `4eef975`.

`claude.md`'s fast-backup description updated to the own-branch-fallback
semantics (commit `25ece79`).

## e2e suite status (post-fix rerun)

`tests/e2e/test_sync_vs_force_jit.py`: **3 passed, 1 xfailed in 5:56** (was 3
timeout-failures in 43:24). Single-prompt sync ≡ async on tokens AND per-step
trace; multi-prompt final tokens identical; seq #0 trace exact at 64 tokens.
The historical seq #1 trace divergence (B10) reproduces exactly as recorded
and is now *characterized* rather than mysterious: benign per-row bf16
numerics between the two process topologies flipping near-tie draft proposals
at batch rows > 0 — final tokens asserted identical, the draft process proven
bit-deterministic and row-permutation-equivariant at fixed shapes, and its
conditioning inputs proven HF-correct. Kept as a strict xfail with the
evidence written into its reason.

## Known remaining issues

- verify()'s greedy fallback on miss rows at temperature>0 accepts a proposed
  token iff it equals argmax(p) — a slight distributional bias inherent to the
  fast backup (documented; ratio acceptance can't apply without q).
