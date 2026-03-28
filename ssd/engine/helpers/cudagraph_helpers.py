import os
import math
import torch

from ssd.utils.context import set_context, get_context, reset_context
from time import perf_counter


## RUN CUDAGRAPHS
@torch.inference_mode()
def run_verify_cudagraph(model_runner, input_ids, positions, last_only, graph_vars):
    context = get_context()
    k_plus_1 = model_runner.config.speculate_k + 1
    orig_bs = input_ids.size(0) // k_plus_1  # orig_bs = N here

    wrapper_bs = next(
        x for x in model_runner.graph_bs_list["verify"] if x >= orig_bs)
    graph = model_runner.graphs["verify"][wrapper_bs]

    for k, v in graph_vars.items():
        if k != "outputs":
            v.zero_()

    # Pad to graph bucket size if needed (fixes B>=6 crash from non-monotonic cu_seqlens_q)
    if wrapper_bs > orig_bs:
        pad_bs = wrapper_bs - orig_bs
        pad_flat = pad_bs * k_plus_1
        dev = input_ids.device

        input_ids = torch.cat([input_ids, torch.zeros(pad_flat, dtype=input_ids.dtype, device=dev)])
        positions = torch.cat([positions, torch.zeros(pad_flat, dtype=positions.dtype, device=dev)])
        slot_mapping = torch.cat([
            context.slot_mapping,
            torch.full((pad_flat,), -1, dtype=context.slot_mapping.dtype, device=dev)])
        # Repeat last real row for ghost sequences (valid page table / context len)
        bt = context.block_tables
        cl = context.context_lens
        block_tables = torch.cat([bt, bt[orig_bs-1:orig_bs].expand(pad_bs, -1).contiguous()])
        context_lens = torch.cat([cl, cl[orig_bs-1:orig_bs].expand(pad_bs).contiguous()])
        bs = wrapper_bs
    else:
        slot_mapping = context.slot_mapping
        block_tables = context.block_tables
        context_lens = context.context_lens
        bs = orig_bs

    graph_vars["input_ids"][:bs * k_plus_1] = input_ids
    graph_vars["positions"][:bs * k_plus_1] = positions
    graph_vars["slot_mapping"][:bs * k_plus_1] = slot_mapping
    graph_vars["context_lens"][:bs] = context_lens
    # Construct cu_seqlens_q for FULL padded batch (monotonically increasing)
    seqlen_q = torch.full(
        (bs,), k_plus_1, dtype=torch.int32, device=graph_vars["cu_seqlens_q"].device)
    cu = graph_vars["cu_seqlens_q"][:bs + 1]
    cu.zero_()
    cu[1:].copy_(torch.cumsum(seqlen_q, 0))

    if block_tables is not None:
        graph_vars["block_tables"][:bs, :block_tables.size(1)] = block_tables

    _pt = os.environ.get("SSD_PROFILE_TARGET", "0") == "1"
    if _pt:
        torch.cuda.synchronize()
        _t0 = perf_counter()

    graph.replay()

    if _pt:
        torch.cuda.synchronize()
        _t1 = perf_counter()

    # Extract outputs for the ORIGINAL batch size only
    outputs = graph_vars["outputs"][:orig_bs * k_plus_1]
    logits = model_runner.model.compute_logits(outputs, last_only)

    if _pt:
        torch.cuda.synchronize()
        _t2 = perf_counter()
        has_eagle = "eagle_acts" in graph_vars
        print(f"[cuda_graph_helpers.run_verify_cudagraph][PROFILE verify_cg] replay={(_t1-_t0)*1000:.2f}ms logits={(_t2-_t1)*1000:.2f}ms eagle={has_eagle} bs={orig_bs} rank={model_runner.rank}", flush=True)

    # For eagle target, also return eagle_acts
    if "eagle_acts" in graph_vars:
        eagle_acts = graph_vars["eagle_acts"][:orig_bs * k_plus_1]
        return logits, eagle_acts
    return logits


@torch.inference_mode()
def run_decode_cudagraph(model_runner, input_ids, positions, last_only, graph_vars, hidden_states=None):
    context = get_context()

    flat_batch_size = input_ids.size(0)

    graph = model_runner.graphs["decode"][next(
        x for x in model_runner.graph_bs_list["decode"] if x >= flat_batch_size)]

    for k, v in graph_vars.items():
            if k != "outputs":
                v.zero_()

    graph_vars["input_ids"][:flat_batch_size] = input_ids
    graph_vars["positions"][:flat_batch_size] = positions
    graph_vars["slot_mapping"][:flat_batch_size] = context.slot_mapping
    graph_vars["context_lens"][:flat_batch_size] = context.context_lens

    if hidden_states is not None and "hidden_states" in graph_vars:
        graph_vars["hidden_states"][:flat_batch_size] = hidden_states

    if context.block_tables is not None:
        graph_vars["block_tables"][:flat_batch_size,
                                :context.block_tables.size(1)] = context.block_tables

    graph.replay()

    outputs = graph_vars["outputs"][:flat_batch_size]
    logits = model_runner.model.compute_logits(outputs, last_only)
    # EAGLE draft: outputs is prenorm, return both
    if "hidden_states" in graph_vars:
        return logits, outputs
    return logits


PROFILE = os.environ.get("SSD_PROFILE", "0") == "1"
PROFILE_DRAFT = os.environ.get("SSD_PROFILE_DRAFT", "0") == "1"
_draft_events = []  # [(step, label, start_event, end_event), ...]

def flush_draft_profile():
    """Sync once, read all CUDA events, print per-step breakdown, clear list."""
    if not _draft_events:
        return
    torch.cuda.synchronize()
    by_step = {}
    for step, label, ev0, ev1 in _draft_events:
        by_step.setdefault(step, []).append((label, ev0.elapsed_time(ev1)))
    parts = []
    total = 0.0
    for step in sorted(by_step):
        step_total = sum(t for _, t in by_step[step])
        detail = " ".join(f"{l}={t:.2f}" for l, t in by_step[step])
        parts.append(f"s{step}={step_total:.2f}({detail})")
        total += step_total
    print(f"[cuda_graph_helpers.flush_draft_profile][PROFILE draft_detail] K={len(by_step)} total={total:.2f}ms avg_step={total/len(by_step):.2f}ms | {' '.join(parts)}", flush=True)
    _draft_events.clear()

@torch.inference_mode()
def run_fi_tree_decode_cudagraph(model_runner, input_ids, positions, last_only, graph_vars, step, cache_hits, hidden_states=None):
    context = get_context()

    MQ_LEN = sum(model_runner.config.fan_out_list)
    orig_flat = input_ids.size(0)
    assert orig_flat % MQ_LEN == 0, f"ERROR in run_fi_tree_decode_cudagraph: flat_batch_size should be divisible by MQ_LEN, got {orig_flat} and {MQ_LEN}"
    orig_B = orig_flat // MQ_LEN

    # Pick CUDA graph bucket
    wrapper_bs = next(
        x for x in model_runner.graph_bs_list["fi_tree_decode"] if x >= orig_B)
    graph = model_runner.graphs["fi_tree_decode"][wrapper_bs]

    # Prepare padded inputs/context if needed
    if wrapper_bs > orig_B:
        pad_B = wrapper_bs - orig_B
        pad_flat = pad_B * MQ_LEN

        pad_ids = torch.zeros(
            pad_flat, dtype=input_ids.dtype, device=input_ids.device)
        pad_pos = torch.zeros(
            pad_flat, dtype=positions.dtype, device=positions.device)
        input_ids = torch.cat([input_ids, pad_ids], dim=0)
        positions = torch.cat([positions, pad_pos], dim=0)

        slot_map = torch.cat(
            [context.slot_mapping,
             torch.full((pad_flat,), -1, dtype=context.slot_mapping.dtype, device=context.slot_mapping.device)]
        )

        bt = context.block_tables
        cl = context.context_lens
        pad_bt = bt[orig_B - 1:orig_B].expand(pad_B, -1).contiguous()
        pad_cl = cl[orig_B - 1:orig_B].expand(pad_B).contiguous()
        bt = torch.cat([bt, pad_bt], dim=0)
        cl = torch.cat([cl, pad_cl], dim=0)

        set_context(is_prefill=False, slot_mapping=slot_map,
                    context_lens=cl, block_tables=bt,
                    tree_cu_seqlens_q=graph_vars["tree_cu_seqlens_q"][wrapper_bs],
                    tree_mask_bias=graph_vars["tree_mask_bias"])

        block_tables = bt
        context_lens = cl
        flat_batch_size = input_ids.size(0)
        B = wrapper_bs
    else:
        block_tables = context.block_tables
        context_lens = context.context_lens
        flat_batch_size = orig_flat
        B = orig_B
        # Set tree decode metadata on context for FA4
        context.tree_cu_seqlens_q = graph_vars["tree_cu_seqlens_q"][wrapper_bs]
        context.tree_mask_bias = graph_vars["tree_mask_bias"]

    if PROFILE:
        torch.cuda.synchronize()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()

    # Build tree mask bias for this step and copy into pre-allocated buffer
    from ssd.layers.tree_mask import build_tree_mask_bias
    K = model_runner.config.speculate_k
    mask_bias = build_tree_mask_bias(
        context_lens, step=step, K=K, MQ_LEN=MQ_LEN,
        fan_out_list=model_runner.config.fan_out_list,
        fan_out_list_miss=model_runner.config.fan_out_list_miss,
        cache_hits=cache_hits,
        max_kv_stride=model_runner.config.max_model_len,
        device=model_runner.device,
    )
    graph_vars["tree_mask_bias"][:len(mask_bias)] = mask_bias

    # Copy inputs/context into graph buffers
    graph_vars["input_ids"][:flat_batch_size] = input_ids
    graph_vars["positions"][:flat_batch_size] = positions
    graph_vars["slot_mapping"][:flat_batch_size] = get_context().slot_mapping
    graph_vars["context_lens"][:B] = context_lens
    if hidden_states is not None and "hidden_states" in graph_vars:
        if hidden_states.shape[0] < flat_batch_size:
            pad_n = flat_batch_size - hidden_states.shape[0]
            hidden_states = torch.cat([hidden_states, torch.zeros(pad_n, hidden_states.shape[1], dtype=hidden_states.dtype, device=hidden_states.device)])
        graph_vars["hidden_states"][:flat_batch_size] = hidden_states
    if step == 0:
        graph_vars["block_tables"][:B, :block_tables.size(1)] = block_tables

    if PROFILE:
        end_time.record()
        torch.cuda.synchronize()
        buffer_prep_time = start_time.elapsed_time(end_time)
        start_time.record()

    if PROFILE_DRAFT:
        _ev_replay0 = torch.cuda.Event(enable_timing=True); _ev_replay0.record()

    graph.replay()

    if PROFILE_DRAFT:
        _ev_replay1 = torch.cuda.Event(enable_timing=True); _ev_replay1.record()
        _draft_events.append((step, "replay", _ev_replay0, _ev_replay1))

    if PROFILE:
        end_time.record()
        torch.cuda.synchronize()
        replay_time = start_time.elapsed_time(end_time)

    logits_all = graph_vars["logits"][:flat_batch_size]

    if PROFILE:
        print(f"[cuda_graph_helpers.run_fi_tree_decode_cudagraph] step {step}: buffer={buffer_prep_time:.3f}ms, replay={replay_time:.3f}ms", flush=True)

    logits_out = logits_all[:orig_flat]
    if "hidden_states" in graph_vars:
        prenorm = graph_vars["outputs"][:orig_flat]
        return logits_out, prenorm
    return logits_out


## CAPTURE CUDAGRAPHS
@torch.inference_mode()
def capture_cudagraph(model_runner):
    config = model_runner.config
    hf_config = config.hf_config
    max_seqs = min(model_runner.config.max_num_seqs, 512)
    if model_runner.config.speculate and model_runner.config.draft_async and model_runner.is_draft:
        N = max_seqs * (model_runner.config.speculate_k + 1) * \
            model_runner.config.async_fan_out
        max_bs = N * (model_runner.config.speculate_k + 1)
    else:
        max_bs = max_seqs + 1
    max_num_blocks = (config.max_model_len +
                      model_runner.block_size - 1) // model_runner.block_size
    input_ids = torch.zeros(max_bs, dtype=torch.int64)
    positions = torch.zeros(max_bs, dtype=torch.int64)
    slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
    context_lens = torch.zeros(max_bs, dtype=torch.int32)
    block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
    outputs = torch.zeros(max_bs, hf_config.hidden_size)

    if model_runner.config.speculate and model_runner.config.draft_async and model_runner.is_draft:
        # Power-of-two buckets: max_bs = max_seqs*(K+1)^2*F can be huge (e.g. 9600 for k=9,f=3,B=32),
        # linear step-16 would create ~600 graphs. Power-of-two gives ~15.
        N = max_seqs * (model_runner.config.speculate_k + 1) * model_runner.config.async_fan_out
        graph_bs_list = []
        bs = 1
        while bs < max_bs:
            graph_bs_list.append(bs)
            bs *= 2
        if max_bs not in graph_bs_list:
            graph_bs_list.append(max_bs)
        # Ensure N (tree decode batch size) is a bucket for exact-fit replay
        if N not in graph_bs_list:
            graph_bs_list.append(N)
            graph_bs_list.sort()
    else:
        graph_bs_list = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        if max_bs % 16 != 0:
            graph_bs_list.append(max_bs)

    graphs = {}
    graph_pool = None

    is_jit = (model_runner.config.speculate and model_runner.config.draft_async and model_runner.is_draft)

    # Eagle models need special handling during CUDA graph capture
    is_eagle_or_phoenix_draft = config.use_eagle_or_phoenix and model_runner.is_draft
    is_eagle_or_phoenix_target = config.use_eagle_or_phoenix and not model_runner.is_draft
    hidden_states = None
    if is_eagle_or_phoenix_draft:
        # Note: For Eagle3, all callers project target acts via fc() BEFORE passing to CG
        hidden_states = torch.zeros(
            max_bs,
            model_runner.hidden_states_dim,
            dtype=hf_config.torch_dtype,
            device=input_ids.device,
        )

    total_graphs = len(graph_bs_list)
    print(f'[capture_cudagraph] Starting capture of {total_graphs} graphs, bs list: {graph_bs_list[:5]}...{graph_bs_list[-3:]} max_bs={max_bs}', flush=True)
    for idx, bs in enumerate(reversed(graph_bs_list)):
        print(f'[capture_cudagraph] Capturing graph {idx+1}/{total_graphs}, bs={bs}', flush=True)
        graph = torch.cuda.CUDAGraph()
        set_context(
            False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs], is_jit=is_jit)
        if is_eagle_or_phoenix_draft:
            outputs[:bs] = model_runner.model(
                input_ids[:bs], positions[:bs], hidden_states[:bs])    # warmup
        elif is_eagle_or_phoenix_target:
            out, _ = model_runner.model(
                input_ids[:bs], positions[:bs])    # warmup
            outputs[:bs] = out
        else:
            outputs[:bs] = model_runner.model(
                input_ids[:bs], positions[:bs])    # warmup
        with torch.cuda.graph(graph, graph_pool):
            if is_eagle_or_phoenix_draft:
                outputs[:bs] = model_runner.model(
                    input_ids[:bs], positions[:bs], hidden_states[:bs])    # capture
            elif is_eagle_or_phoenix_target:
                out, _ = model_runner.model(
                    input_ids[:bs], positions[:bs])    # capture
                outputs[:bs] = out
            else:
                outputs[:bs] = model_runner.model(
                    input_ids[:bs], positions[:bs])    # capture
        if graph_pool is None:
            graph_pool = graph.pool()
        graphs[bs] = graph
        torch.cuda.synchronize()
        reset_context()

    graph_vars = dict(
        input_ids=input_ids,
        positions=positions,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        outputs=outputs,
    )
    if hidden_states is not None:
        graph_vars["hidden_states"] = hidden_states

    return graph_vars, graph_pool, graphs, graph_bs_list


@torch.inference_mode()
def capture_verify_cudagraph(model_runner):
    config = model_runner.config
    # assert not model_runner.is_draft, "ERROR in capture_verify_cudagraph: verify path only supported for target model"
    hf_config = config.hf_config
    max_bs = min(model_runner.config.max_num_seqs, 512)
    k_plus_1 = model_runner.config.speculate_k + 1

    is_eagle_or_phoenix_target = config.use_eagle_or_phoenix and not model_runner.is_draft

    # For verify, we need to handle k+1 tokens per sequence, and use cu_seqlens_q and max_seqlen_q
    input_ids = torch.zeros(max_bs * k_plus_1, dtype=torch.int64)
    positions = torch.zeros(max_bs * k_plus_1, dtype=torch.int64)
    slot_mapping = torch.zeros(max_bs * k_plus_1, dtype=torch.int32)
    context_lens = torch.zeros(max_bs, dtype=torch.int32)
    block_tables = torch.zeros(
        max_bs, model_runner.max_num_blocks, dtype=torch.int32)
    outputs = torch.zeros(max_bs * k_plus_1, hf_config.hidden_size)
    cu_seqlens_q = torch.zeros(max_bs + 1, dtype=torch.int32)

    # Eagle/Phoenix target: also capture activations from model forward
    eagle_acts = None
    if is_eagle_or_phoenix_target:
        eagle_acts = torch.zeros(
            max_bs * k_plus_1,
            model_runner.eagle_acts_dim,
            dtype=hf_config.torch_dtype,
        )

    base = [1, 2, 4, 8]
    dynamic = list(range(16, max_bs+1, 16))
    all_b = base + dynamic
    if max_bs not in all_b:
        all_b.append(max_bs)
    all_b.sort()
    all_N = [b for b in all_b if b <= max_bs]

    graphs = {}
    graph_pool = None

    for bs in reversed(all_N):
        graph = torch.cuda.CUDAGraph()
        # For verify, each sequence is length K+1, so seqlen_q is [K+1]*bs
        seqlen_q = torch.full((bs,), k_plus_1, dtype=torch.int32)
        cu = cu_seqlens_q[:bs + 1]
        cu.zero_()
        cu[1:].copy_(torch.cumsum(seqlen_q, 0))
        context_lens[:bs] = seqlen_q

        set_context(
            is_prefill=False,
            slot_mapping=slot_mapping[:bs * k_plus_1],
            context_lens=context_lens[:bs],
            block_tables=block_tables[:bs],
            cu_seqlens_q=cu,
            max_seqlen_q=k_plus_1,
        )

        # warmup
        model_out = model_runner.model(
            input_ids[:bs * k_plus_1], positions[:bs * k_plus_1])
        if isinstance(model_out, tuple):
            outputs[:bs * k_plus_1] = model_out[0]
            if eagle_acts is not None:
                eagle_acts[:bs * k_plus_1] = model_out[1]
        else:
            outputs[:bs * k_plus_1] = model_out
        with torch.cuda.graph(graph, graph_pool):
            # capture
            model_out = model_runner.model(
                input_ids[:bs * k_plus_1], positions[:bs * k_plus_1])
            if isinstance(model_out, tuple):
                outputs[:bs * k_plus_1] = model_out[0]
                if eagle_acts is not None:
                    eagle_acts[:bs * k_plus_1] = model_out[1]
            else:
                outputs[:bs * k_plus_1] = model_out

        if graph_pool is None:
            graph_pool = graph.pool()
        graphs[bs] = graph
        torch.cuda.synchronize()
        reset_context()

    graph_vars = dict(
        input_ids=input_ids,
        positions=positions,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        cu_seqlens_q=cu_seqlens_q,
        outputs=outputs,
    )
    if eagle_acts is not None:
        graph_vars["eagle_acts"] = eagle_acts

    return graph_vars, graph_pool, graphs, all_N


@torch.inference_mode()
def run_glue_decode_cudagraph(model_runner, input_ids, positions, last_only, graph_vars, hidden_states=None):
    """Run EAGLE glue decode with FA causal + varlen cu_seqlens_q. No padding within sequences."""
    context = get_context()
    K = model_runner.config.speculate_k
    two_kp1 = 2 * K + 1
    orig_flat = input_ids.size(0)
    orig_B = context.context_lens.size(0)
    dev = input_ids.device

    wrapper_bs = next(
        x for x in model_runner.graph_bs_list["glue_decode"] if x >= orig_B)
    graph = model_runner.graphs["glue_decode"][wrapper_bs]
    max_flat = wrapper_bs * two_kp1

    # Zero all non-output graph vars
    for k, v in graph_vars.items():
        if k != "outputs":
            v.zero_()

    # Copy real data into graph buffers (orig_flat <= max_flat always)
    graph_vars["input_ids"][:orig_flat] = input_ids
    graph_vars["positions"][:orig_flat] = positions
    graph_vars["slot_mapping"][:orig_flat] = context.slot_mapping
    # Pad remaining flat slots with -1 slot_mapping (no KV write)
    if orig_flat < max_flat:
        graph_vars["slot_mapping"][orig_flat:max_flat] = -1

    graph_vars["context_lens"][:orig_B] = context.context_lens
    graph_vars["block_tables"][:orig_B, :context.block_tables.size(1)] = context.block_tables

    # cu_seqlens_q: real seqs, then ghost seqs (repeat last cumsum = 0-length queries)
    cu = context.cu_seqlens_q  # [orig_B + 1]
    graph_vars["cu_seqlens_q"][:orig_B + 1] = cu
    if wrapper_bs > orig_B:
        # Ghost seqs get 0-length queries
        graph_vars["cu_seqlens_q"][orig_B + 1:wrapper_bs + 1] = cu[-1]
        # Ghost seqs need valid block_tables/context_lens (copy last real seq)
        pad_B = wrapper_bs - orig_B
        graph_vars["context_lens"][orig_B:wrapper_bs] = context.context_lens[orig_B - 1]
        graph_vars["block_tables"][orig_B:wrapper_bs] = context.block_tables[orig_B - 1]

    if hidden_states is not None and "eagle_hidden_states" in graph_vars:
        graph_vars["eagle_hidden_states"][:orig_flat] = hidden_states

    graph.replay()

    outputs = graph_vars["outputs"][:orig_flat]
    logits = model_runner.model.compute_logits(outputs, last_only)
    assert logits.dim() == 2, "ERROR in run_glue_decode_cudagraph: logits must be 2D"
    if "eagle_hidden_states" in graph_vars:
        return logits, outputs
    return logits


@torch.inference_mode()
def capture_glue_decode_cudagraph(model_runner):
    """Capture CG for EAGLE glue decode: FA causal + varlen cu_seqlens_q, max flat = B*(2K+1)."""
    config = model_runner.config
    hf_config = config.hf_config
    max_bs = min(config.max_num_seqs, 512)
    K = config.speculate_k
    two_kp1 = 2 * K + 1
    max_flat = max_bs * two_kp1
    max_num_blocks = (config.max_model_len + model_runner.block_size - 1) // model_runner.block_size

    input_ids = torch.zeros(max_flat, dtype=torch.int64, device=model_runner.device)
    positions = torch.zeros(max_flat, dtype=torch.int64, device=model_runner.device)
    slot_mapping = torch.zeros(max_flat, dtype=torch.int32, device=model_runner.device)
    context_lens = torch.full((max_bs,), config.max_model_len, dtype=torch.int32, device=model_runner.device)
    block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32, device=model_runner.device)
    outputs = torch.empty(max_flat, hf_config.hidden_size, device=model_runner.device)
    cu_seqlens_q = torch.zeros(max_bs + 1, dtype=torch.int32, device=model_runner.device)

    eagle_hidden_states = None
    if config.use_eagle_or_phoenix and model_runner.is_draft:
        eagle_hidden_states = torch.zeros(
            max_flat,
            model_runner.hidden_states_dim,
            dtype=hf_config.torch_dtype,
            device=model_runner.device,
        )

    graph_bs_list = [1]
    for bs in [2, 4, 8] + list(range(16, max_bs + 1, 16)):
        if bs <= max_bs:
            graph_bs_list.append(bs)
    if max_bs not in graph_bs_list:
        graph_bs_list.append(max_bs)
    graph_bs_list.sort()

    graphs = {}
    graph_pool = None

    print(f'[cuda_graph_helpers.capture_glue_decode_cudagraph] Capturing for bs={graph_bs_list}', flush=True)

    for bs in reversed(graph_bs_list):
        graph = torch.cuda.CUDAGraph()
        flat = bs * two_kp1

        # Uniform cu_seqlens_q for capture (each seq gets 2K+1 queries)
        seqlen_q = torch.full((bs,), two_kp1, dtype=torch.int32, device=model_runner.device)
        cu = cu_seqlens_q[:bs + 1]
        cu.zero_()
        cu[1:].copy_(torch.cumsum(seqlen_q, 0))

        set_context(
            is_prefill=False,
            cu_seqlens_q=cu,
            max_seqlen_q=two_kp1,
            slot_mapping=slot_mapping[:flat],
            context_lens=context_lens[:bs],
            block_tables=block_tables[:bs],
        )

        if eagle_hidden_states is not None:
            outputs[:flat] = model_runner.model(input_ids[:flat], positions[:flat], eagle_hidden_states[:flat])
        else:
            outputs[:flat] = model_runner.model(input_ids[:flat], positions[:flat])

        with torch.cuda.graph(graph, graph_pool):
            if eagle_hidden_states is not None:
                outputs[:flat] = model_runner.model(input_ids[:flat], positions[:flat], eagle_hidden_states[:flat])
            else:
                outputs[:flat] = model_runner.model(input_ids[:flat], positions[:flat])

        if graph_pool is None:
            graph_pool = graph.pool()
        graphs[bs] = graph
        torch.cuda.synchronize()
        reset_context()

    graph_vars = dict(
        input_ids=input_ids,
        positions=positions,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        cu_seqlens_q=cu_seqlens_q,
        outputs=outputs,
    )
    if eagle_hidden_states is not None:
        graph_vars["eagle_hidden_states"] = eagle_hidden_states

    return graph_vars, graph_pool, graphs, graph_bs_list


@torch.inference_mode()
def capture_fi_tree_decode_cudagraph(model_runner):
    config = model_runner.config
    hf_config = config.hf_config
    max_bs = min(model_runner.config.max_num_seqs, 512)
    MQ_LEN = sum(model_runner.config.fan_out_list)
    max_flat_batch_size = max_bs * MQ_LEN

    max_num_blocks = (config.max_model_len +
                      model_runner.block_size - 1) // model_runner.block_size
    input_ids = torch.zeros(max_flat_batch_size, dtype=torch.int64, device=model_runner.device)
    positions = torch.zeros(max_flat_batch_size, dtype=torch.int64, device=model_runner.device)
    slot_mapping = torch.zeros(max_flat_batch_size, dtype=torch.int32, device=model_runner.device)
    context_lens = torch.full((max_bs,), config.max_model_len, dtype=torch.int32, device=model_runner.device)
    block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32, device=model_runner.device)
    outputs = torch.empty(max_flat_batch_size, hf_config.hidden_size, device=model_runner.device)
    logits = torch.empty(max_flat_batch_size, hf_config.vocab_size, device=model_runner.device)

    graph_bs_list = [1]
    for bs in [2, 4, 8] + list(range(16, max_bs + 1, 16)):
        if bs <= max_bs:
            graph_bs_list.append(bs)
    if max_bs not in graph_bs_list:
        graph_bs_list.append(max_bs)
    graph_bs_list.sort()

    graphs = {}
    graph_pool = None

    fi_hidden_states = None
    if config.use_eagle_or_phoenix and model_runner.is_draft:
        fi_hidden_states = torch.zeros(
            max_flat_batch_size,
            model_runner.hidden_states_dim,
            dtype=hf_config.torch_dtype,
            device=model_runner.device,
        )

    # Pre-allocate tree_cu_seqlens_q per batch size bucket (constant values, used by FA4)
    tree_cu_seqlens_q_dict = {}
    for bs in graph_bs_list:
        tree_cu_seqlens_q_dict[bs] = torch.arange(
            bs + 1, dtype=torch.int32, device=model_runner.device) * MQ_LEN

    # Pre-allocate tree mask bias at max size (shared across all batch sizes, updated before replay)
    tree_mask_bias = torch.zeros(
        max_flat_batch_size * config.max_model_len,
        dtype=torch.float32, device=model_runner.device)

    print(f'[cuda_graph_helpers.capture_fi_tree_decode_cudagraph] About to capture FA4 tree decode cudagraphs for bs={graph_bs_list}', flush=True)

    for bs in reversed(graph_bs_list):
        graph = torch.cuda.CUDAGraph()

        # Set context with FA4 metadata
        set_context(
            is_prefill=False,
            slot_mapping=slot_mapping[:bs * MQ_LEN],
            context_lens=context_lens[:bs],
            block_tables=block_tables[:bs],
            tree_cu_seqlens_q=tree_cu_seqlens_q_dict[bs],
            tree_mask_bias=tree_mask_bias,
        )

        # Warmup run
        if fi_hidden_states is not None:
            outputs[:bs * MQ_LEN] = model_runner.model(
                input_ids[:bs * MQ_LEN], positions[:bs * MQ_LEN], fi_hidden_states[:bs * MQ_LEN])
        else:
            outputs[:bs * MQ_LEN] = model_runner.model(
                input_ids[:bs * MQ_LEN], positions[:bs * MQ_LEN])
        logits[:bs * MQ_LEN] = model_runner.model.compute_logits(outputs[:bs * MQ_LEN], False)

        # Capture both model run and logits computation
        with torch.cuda.graph(graph, graph_pool):
            if fi_hidden_states is not None:
                outputs[:bs * MQ_LEN] = model_runner.model(
                    input_ids[:bs * MQ_LEN], positions[:bs * MQ_LEN], fi_hidden_states[:bs * MQ_LEN])
            else:
                outputs[:bs * MQ_LEN] = model_runner.model(input_ids[:bs * MQ_LEN], positions[:bs * MQ_LEN])
            logits[:bs * MQ_LEN] = model_runner.model.compute_logits(outputs[:bs * MQ_LEN], False)

        if graph_pool is None:
            graph_pool = graph.pool()
        graphs[bs] = graph

        torch.cuda.synchronize()
        reset_context()

    graph_vars = dict(
        input_ids=input_ids,
        positions=positions,
        slot_mapping=slot_mapping,
        block_tables=block_tables,
        context_lens=context_lens,
        outputs=outputs,
        logits=logits,
        tree_cu_seqlens_q=tree_cu_seqlens_q_dict,
        tree_mask_bias=tree_mask_bias,
    )
    if fi_hidden_states is not None:
        graph_vars["hidden_states"] = fi_hidden_states

    return graph_vars, graph_pool, graphs, graph_bs_list
