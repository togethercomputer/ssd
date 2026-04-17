# Fused CUDA graph for the async decode_tree loop

**Status:** design sketch. Implementation gated behind a config flag once reviewed.
**Scope:** argmax (greedy) only. Multinomial sampling out of scope — error out if any
draft temperature is non-zero.

## Goal

The async draft path runs K model forwards per `_decode_tree` call, each captured
in its own CUDA graph and driven by a Python `for` loop
(`ssd/engine/draft_runner.py::_decode_tree`). The per-step Python interleaving +
the per-step `set_context` / buffer-copy / argmax / hidden-states chaining
observably costs ~0.1–0.4 ms per step. At K=15 that's several ms.

This design replaces the K per-step graphs with **one fused graph per batch-size
bucket** that captures K back-to-back model forwards plus argmax plus the
inter-step chaining (input_ids, EAGLE hidden_states). Replay is one
`graph.replay()` call.

TGL's sync eagle path does exactly this — see
`/work/avner/git/tgl/python/sglang/srt/speculative/eagle_worker.py:654`
(`draft_forward`) and
`/work/avner/git/tgl/python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py:215`
(`capture_one_batch_size` calls `run_once()` which runs the full K-step loop
inside `torch.cuda.graph(...)`). That's the reference implementation we're
porting the *shape* of.

## Expected speedup

Measured on 8B eagle async-fast K=15 (post-Opt1) with `SSD_PROFILE_DRAFT=1`:

- Pure `graph.replay()` per step: **0.37–0.39 ms**
- Sum of K=15 replays: **5.68 ms**
- Current per-step total (incl. Python): 0.56–0.97 ms, Σ = 8.93 ms
- `decode_tree_ms` (outer, incl. other pre/post work): **9.90 ms**

Fusing K steps into one graph replay eliminates essentially all non-forward
overhead (~0.18 ms/step for steps 1+, ~0.6 ms for step 0 = ~3 ms/call). The
forward cost itself is unchanged.

Lower bound for fused decode_tree at K=15: **~5.7 ms**, essentially matching
TGL sync's `sync_draft_ms = 5.85 ms` reference. That's a **~42 % reduction
from 9.90 ms**, on top of Opt1.

Why we match TGL rather than falling short: at B=1 the 1B draft's per-step
forward is weight-bandwidth-bound, and the 64× extra tokens/step in async
tree decode (vs TGL's topk=1 single token/step) don't cost proportional time.
Earlier design draft overestimated the gap.

## Background: what the per-step graph captures today

Captured inside `torch.cuda.graph(graph, graph_pool)` at
`ssd/engine/helpers/cudagraph_helpers.py:706`:

```
outputs[:bs*MQ_LEN] = model(input_ids[:bs*MQ_LEN],
                             positions[:bs*MQ_LEN],
                             fi_hidden_states[:bs*MQ_LEN])   # EAGLE: + hidden_states
logits[:bs*MQ_LEN]  = model.compute_logits(outputs[:bs*MQ_LEN], last_only=False)
```

Per step, driven by `_decode_tree_step` (outside the graph):
1. `set_context(slot_mapping=step_slot_maps[s], context_lens=step_context_lens[s],
   block_tables=dbt, tree_cu_seqlens_q=..., tree_mask_bias=...)` — writes to a
   thread-local dict the FA4 backend reads inside the model.
2. Copy `input_ids`, `positions`, `slot_mapping`, `context_lens`, `hidden_states`
   (EAGLE) into graph-bound buffers.
3. `graph.replay()`.
4. `reset_context()`.
5. `next_tokens = logits.argmax(-1)` (greedy) — kernel launched outside the graph.
6. EAGLE: `payload["hidden_states"] = prenorm` — next step reads this.

Outside `_decode_tree_step`, pre-loop in `_decode_tree`:
- `step_positions`, `step_rope_positions`, `step_context_lens`, `step_slot_maps`
  are precomputed for all K steps (see `_compute_step_positions_and_slot_maps`).
- `tree_mask_bias` is built once at the widest step=K-1 shape and reused for
  all K steps (the Opt1 change in `cudagraph_helpers.py`).

So today, everything that varies per step *structurally* is already materialized
in `[K, N]`-shaped GPU buffers before the loop. The per-step Python work is just
writing slices of those buffers into `graph_vars[...]` and triggering replay.

## Proposed design

One CUDA graph per batch-size bucket captures **K back-to-back
attention-bearing forwards** plus K argmax-and-write ops. All per-step
variation is served by fixed-offset reads into pre-filled `[K, ...]` buffers.

### What goes inside the graph

For `depth = 0, 1, …, K-1` (K iterations, all inside one
`torch.cuda.graph(...)` context):

1. Slice-based context setup:
   - `set_context(slot_mapping=step_slot_maps[depth],
     context_lens=step_context_lens[depth], block_tables=dbt,
     tree_cu_seqlens_q=tree_cu_seqlens_q, tree_mask_bias=tree_mask_bias)`.
   - This is Python, but it only stores references to EXISTING tensor views
     inside a thread-local. At capture time, the FA4 backend compiles against
     those references. Since each depth binds a different *view* of the same
     `step_slot_maps[K, flat]` tensor, each forward reads from its own offset.
   - We do NOT call `reset_context()` between steps — the context is overwritten
     by the next `set_context`. Final `reset_context()` happens once after the graph.

2. Forward:
   ```
   outputs_buf[:flat] = model(current_input_ids[:flat],
                              step_rope_positions[depth, :flat],
                              current_hidden_states[:flat])   # EAGLE
   logits_buf[depth, :flat] = model.compute_logits(outputs_buf[:flat],
                                                    last_only=False)
   ```
   Notes:
   - `current_input_ids[flat]` is a *single* buffer (not `[K, flat]`). Step 0
     reads the externally-prefilled value; step s writes its argmax into it
     before step s+1 reads it. Buffer aliasing makes the dataflow implicit in
     the captured graph.
   - For EAGLE, `current_hidden_states[flat, H]` is also a single buffer. Step s
     writes prenorm (= outputs_buf post-layernorm) into it for step s+1 to read.
   - `logits_buf[K, flat, V]` stores per-step logits because the caller needs
     all K slices (they get written into the tree cache). This is the one
     [K, …] *output* buffer we need.
   - `spec_activations` (EAGLE only) is similarly a `[K, flat, H]` output
     buffer fed from each step's prenorm. Can share storage with
     `step_prenorms` if we pre-size that way.

3. In-graph argmax + write:
   ```
   next_tokens = logits_buf[depth, :flat].argmax(dim=-1)   # shape [flat]
   spec_tokens_buf[depth, :flat] = next_tokens
   if depth < K-1:
       current_input_ids[:flat] = next_tokens
   ```
   `argmax` is a standard reduction kernel, fully capturable.

### What stays outside the graph

Pre-replay, every call:
- Compute `step_slot_maps`, `step_positions`, `step_rope_positions`,
  `step_context_lens` — already done in `_compute_step_positions_and_slot_maps`.
- Build `tree_mask_bias` once — already done by Opt1.
- Write step-0's `current_input_ids` (the forked recovery tokens) and EAGLE
  `current_hidden_states` into their bound buffers.
- Copy `step_slot_maps`, `step_rope_positions`, `step_context_lens` into their
  bound `[K, ...]` buffers (one copy each, not K copies).
- Write `block_tables` into its bound buffer.
- Temperature-zero check (see below).
- `graph.replay()`.
- Read `spec_tokens_buf[:K, :N]`, `logits_buf[:K, :N, :V]`, `spec_activations_buf[:K, :N, :H]`.

## Buffer layout

New graph-vars (on top of what exists today in
`capture_fi_tree_decode_cudagraph`):

| Name | Shape | Notes |
|---|---|---|
| `step_slot_maps_buf` | `[K, max_flat]` int32 | pre-filled per call |
| `step_rope_positions_buf` | `[K, max_flat]` int64 | pre-filled per call |
| `step_context_lens_buf` | `[K, max_bs]` int32 | pre-filled per call |
| `tree_mask_bias` (existing) | `[max_flat * max_kv_stride]` fp32 | pre-filled once |
| `current_input_ids` (existing `input_ids`) | `[max_flat]` int64 | step 0 writes externally; steps 1..K-1 via argmax |
| `current_hidden_states` (existing `hidden_states`) | `[max_flat, H]` bf16 | EAGLE only |
| `logits_buf` | `[K, max_flat, V]` bf16 | per-step logits output |
| `spec_tokens_buf` | `[K, max_flat]` int64 | per-step argmax output |
| `spec_activations_buf` (EAGLE) | `[K, max_flat, H]` bf16 | per-step prenorm |

Memory: for sglang 8B eagle with K=15, flat=64, V=128k, bf16:
`logits_buf` = 15 × 64 × 128000 × 2 = 240 MB.
That's large. Options:
- Share storage with the `logits` buffer already captured (reuse the same tensor
  across steps if the caller only needs the *last* step's logits — but the
  current `_decode_tree` writes every step's logits into `spec_logits[:, depth, :]`,
  so we do need them all).
- Build `logits_buf` at `K × max_flat × V` once at capture time (one 240 MB
  allocation, not per-call).

For K=5 the cost is 80 MB — trivially fine. For K=15 we need the caller (and
downstream `_populate_tree_cache`) to be OK with the memory pressure. The 70B
deployment has 80 GB per GPU; 240 MB is < 0.3 %.

## Capture-time changes

`capture_fi_tree_decode_cudagraph` becomes (per batch-size bucket `bs`):

```python
with torch.cuda.graph(graph, graph_pool):
    current_input_ids_view = input_ids[:bs * MQ_LEN]
    current_hidden_view = fi_hidden_states[:bs * MQ_LEN] if EAGLE else None
    for depth in range(K):
        set_context(
            is_prefill=False,
            slot_mapping=step_slot_maps_buf[depth, :bs * MQ_LEN],
            context_lens=step_context_lens_buf[depth, :bs],
            block_tables=block_tables[:bs],
            tree_cu_seqlens_q=tree_cu_seqlens_q_dict[bs],
            tree_mask_bias=tree_mask_bias,
        )
        outputs_view = outputs[:bs * MQ_LEN]
        if EAGLE:
            outputs_view[:] = model(current_input_ids_view,
                                    step_rope_positions_buf[depth, :bs * MQ_LEN],
                                    current_hidden_view)
        else:
            outputs_view[:] = model(current_input_ids_view,
                                    step_rope_positions_buf[depth, :bs * MQ_LEN])
        logits_buf[depth, :bs * MQ_LEN] = model.compute_logits(outputs_view, last_only=False)
        next_tokens = logits_buf[depth, :bs * MQ_LEN].argmax(dim=-1)
        spec_tokens_buf[depth, :bs * MQ_LEN] = next_tokens
        if EAGLE:
            # prenorm is read from `outputs_view` (which is the model's prenorm
            # output); copy into current_hidden_view + spec_activations
            spec_activations_buf[depth, :bs * MQ_LEN] = outputs_view
            if depth < K - 1:
                current_hidden_view[:] = outputs_view
        if depth < K - 1:
            current_input_ids_view[:] = next_tokens
# one graph captures all K iterations
```

`reset_context()` is called once after capture, same as today.

## Runtime (`_decode_tree`) changes

Replaces the `for depth in range(K): _decode_tree_step(...)` loop:

```python
assert (payload["temps"] == 0).all(), (
    "decode_tree fused graph currently only supports greedy (temp=0). "
    "Non-zero temperatures would require in-graph multinomial sampling, "
    "which is not yet implemented."
)

# Copy pre-computed per-step metadata into the bound [K, ...] buffers.
gv = model_runner.graph_vars["fi_tree_decode"]
gv["step_slot_maps_buf"][:K, :N] = step_slot_maps       # [K, N] already on device
gv["step_rope_positions_buf"][:K, :N] = step_rope_positions
gv["step_context_lens_buf"][:K, :B] = step_context_lens.to(torch.int32)
gv["block_tables"][:B, :dbt.shape[1]] = dbt
gv["tree_mask_bias"][:mask_bias.shape[0]] = mask_bias   # Opt1 already builds this
gv["input_ids"][:N] = payload["input_ids"]              # initial (forked rec)
if EAGLE:
    gv["hidden_states"][:N] = payload["hidden_states"]

model_runner.graphs["fi_tree_decode_fused"][bucket_bs].replay()

spec_tokens = gv["spec_tokens_buf"][:K, :N].transpose(0, 1).contiguous()   # [N, K]
spec_logits = gv["logits_buf"][:K, :N].transpose(0, 1).contiguous()       # [N, K, V]
if EAGLE:
    spec_activations = gv["spec_activations_buf"][:K, :N].transpose(0, 1).contiguous()
```

## Correctness plan

Four levels of verification, cheapest to most expensive:

1. **Unit: equivalence of fused graph output vs per-step graph output** on
   matched inputs (same `step_slot_maps`, `step_rope_positions`, etc., same
   `tree_mask_bias`). Build a synthetic payload, run both paths, assert that
   `spec_tokens`, `spec_logits`, `spec_activations` are bitwise equal. This is
   a tier1 test (needs real model weights or we'd be validating nothing), in
   the shape of `tests/e2e/test_cudagraph_vs_eager.py`.

2. **Tier1 E2E: sync_vs_force_jit with fused-graph flag enabled.** The existing
   `tests/e2e/test_sync_vs_force_jit.py::test_single_prompt_greedy_matches_tokens_and_trace`
   already compares token stream + per-step accept trace between sync spec and
   async+force-jit. Flag the async config to use the fused graph and re-run —
   the test must still pass with identical traces.

3. **Tier1 E2E: cudagraph_vs_eager with fused-graph flag enabled.** Same test,
   with the fused path.

4. **Bench A/B on 8B eagle k=5 and k=15.** Compare decode_tree_ms, avg
   acceptance length (must be unchanged), total throughput. Acceptance length
   is the sensitive metric: a wrong buffer aliasing or order-of-writes bug
   would show up as drift in accepted-tokens-per-step even if the run completes
   without error.

## Risks and open questions

**R1. FA4 metadata wiring at capture.** The FA4 backend reads metadata from the
thread-local context at forward time. When we call `set_context(...)` K times
within one `torch.cuda.graph(...)` block, each call overwrites the thread-local
dict. The compiled model code reads whatever is in the dict *at capture time of
each forward*. As long as each forward call sees its own `set_context` values
(because Python runs first, graph captures the *result*), each per-step
forward's captured ops will reference the correct buffer slice.

The mitigation is: the `set_context` call must happen *immediately before* the
corresponding `model(...)` call inside the graph scope, and must not be
reordered by Python. That's the natural order in the sketch above.

**R2. `cache_hits` at capture time.** Unlike today's per-step graph (which
captures one mask-building position + one forward), the fused graph captures
K forwards but does NOT capture the mask build. The mask is written into the
bound `tree_mask_bias` buffer by the runtime before replay. Every captured
forward reads from the same `tree_mask_bias` tensor — so whichever mask is
currently in the buffer is what all K forwards see. This matches Opt1's
observed correctness (the step=K-1 mask is valid for all steps).

**R3. `argmax` inside the graph.** `torch.argmax` is an elementwise+reduction
kernel and captures cleanly. No known issues.

**R4. Non-greedy path.** `sampler(is_tree=True)` uses `torch.multinomial` which
has CPU/GPU sync issues inside graphs. Out of scope for this change — we
assert temp==0 at entry.

**R5. Mixed hit/miss `cache_hits` across a batch.** Already handled by Opt1:
the tree mask differs per-sequence inside `tree_mask_bias`. The fan_idx /
rope_positions differ per-sequence via `step_rope_positions` (computed from
`_fan_idx_flat(cache_hits)`). Both are pre-filled into buffers before replay,
so the captured graph handles mixed hit/miss with no change.

**R6. Padding (`wrapper_bs > orig_B`).** Same as today: pick a captured
`bucket_bs`, pad inputs up to `bucket_bs * MQ_LEN`, pad `cache_hits` with
zeros. The padding lives in the caller, not the graph.

**R7. `_build_tree_batch`'s pre-compute of `_pre_positions` /
`_pre_rope_positions`.** These are computed for the caller's downstream use and
feed into the *outside-graph* write of `step_rope_positions`. That chain stays.

**R8. Logits memory at large K.** 240 MB for K=15, V=128k, max_flat=64, bf16.
Solvable by (a) making the buffer per-bucket rather than once at max_bs — a
B=1 bucket only needs `K × 1 × MQ_LEN × V × 2` = ~4 MB, or (b) accepting the
cost (0.3 % of an 80 GB GPU). Recommend (a) during implementation.

## Rollout

- **Flag:** add `config.fused_tree_decode_graph: bool = False`. Capture both the
  old per-step graph and the new fused graph when the flag is True; gate the
  replay path in `_decode_tree` on the flag. This lets us A/B in the same
  binary and roll back without reverting a commit.
- **Landing order:**
  1. Add the flag, capture both graph kinds (no-op default).
  2. Implement the fused-graph `_decode_tree` replay under the flag.
  3. Add the tier1 unit-equivalence test (item 1 of Correctness plan).
  4. Verify tier1 E2E tests pass with the flag on.
  5. Bench A/B at k=5 and k=15 with acceptance-length sanity check.
  6. Default flag to True once (1–5) are green on 8B eagle3.
  7. Remove the legacy per-step graph capture + the flag in a follow-up.
- **Dependency on Opt1:** this design requires Opt1 (tree_mask_bias built once)
  to be in. It is (commit `9842a90`).

## Non-goals (explicit)

- Multinomial sampling — deferred.
- Changing the tree structure (fan_out_list) — unchanged.
- Merging the first glue_decode forward into the fused graph — that's a
  separate, larger change; glue_decode's shape differs from tree_decode's.
- Removing the `cache_hits_list` / `cache_hits.tolist()` sync points in
  `_build_tree_batch` — tried in a follow-up to Opt1 and produced a runtime
  behavior change (avg acceptance 1.35 → 1.16) I haven't root-caused. The
  equivalent test `tests/unit/test_fan_idx_flatten.py` passes on 215 configs,
  so the sync itself was load-bearing somewhere. Park this until the fused
  graph is in — the fused graph eliminates most of the per-step Python cost,
  so vectorizing the fan_idx build stops mattering anyway.
