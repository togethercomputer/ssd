import os
import time
from datetime import datetime
import torch
import torch.distributed as dist
import dataclasses

from ssd.engine.model_runner import ModelRunner
from ssd.config import Config
from ssd.utils import profile
from ssd.utils.context import set_context, reset_context
from ssd.utils.misc import compress_neg_ones_and_zeros
from ssd.utils.async_helpers.async_spec_helpers import get_forked_recovery_tokens_from_logits, make_glue_decode_input_ids
from ssd.engine.helpers.runner_helpers import PrefillRequest, SpeculationRequest, SpeculationResponse, COMMAND

NCCL_LOG = os.environ.get("SSD_NCCL_LOG", "0") == "1"
BRIEF_LOG = os.environ.get("SSD_BRIEF_LOG", "0") == "1"

def _ts():
    return f'{datetime.now().strftime("%H:%M:%S.%f")[:-3]}'

ttl = 0
ttl_hit = 0


class DraftRunner(ModelRunner):
    
    @classmethod
    def create_draft_config(cls, cfg: Config) -> Config:
        """Create a draft config from the main config without instantiating DraftRunner."""
        draft_cfg = dataclasses.replace(
            cfg,
            model=cfg.draft,
            gpu_memory_utilization = (0.75 if not cfg.draft_async else 0.8), # REMAINING SPACE if not draft_async
            tokenizer_path=cfg.model if cfg.use_eagle_or_phoenix else None,
            d_model_target=cfg.hf_config.hidden_size if cfg.use_eagle_or_phoenix and cfg.hf_config else None,
        )
        return draft_cfg

    def __init__(self, draft_cfg: Config, rank: int = 0, init_q = None):
        print(f'[DraftRunner.__init__] draft_cfg={draft_cfg}', flush=True)
        self.draft_cfg = draft_cfg
        self.is_draft = True # this is is_draft, use self.config.draft for the draft model path 
        self.prev_num_tokens = None
        super().__init__(self.draft_cfg, rank=rank, event=None, is_draft=True, num_tp_gpus=1, init_q=init_q)
        self._prefill_metadata = torch.empty(5, dtype=torch.int64, device=self.device)
        self._decode_metadata = torch.empty(4, dtype=torch.int64, device=self.device)
        self.target_rank = 0
        self.communicate_logits = self.config.communicate_logits
        self.communicate_cache_hits = self.config.communicate_cache_hits

        if self.is_draft and self.draft_async:
            self._reset_tree_cache_tensors()
            self._init_prealloc_buffers()
            self._draft_step_times = []
            self._acceptance_lengths = []
            self._cache_hits = []
            self._acceptance_rate_log_path = os.environ.get("ACCEPTANCE_RATE_LOG", None)
            if self._acceptance_rate_log_path:
                print(f'[{_ts()}] DraftRunner will log acceptance rate to: {self._acceptance_rate_log_path}', flush=True)
            print(f'[{_ts()}] DraftRunner set up, starting draft_loop', flush=True)
            self.draft_loop()

    def draft_async_prefill(self):
        assert self.draft_async and self.is_draft

        if self.config.verbose:
            print(f'[{_ts()}] [draft_async_prefill] DRAFT ASYNC PREFILL STARTING', flush=True)

        prefill_request = PrefillRequest.receive(self.async_pg, self.target_rank, self.device, metadata_buffer=self._prefill_metadata)
        total_new_tokens, batch_size, max_blocks, use_eagle_or_phoenix, eagle_phoenix_act_dim = prefill_request.metadata.tolist()
        input_ids = prefill_request.input_ids
        num_tokens = prefill_request.num_tokens
        draft_block_table = prefill_request.draft_block_table
        eagle_acts = prefill_request.eagle_acts

        if NCCL_LOG:
            sep = '=' * 80
            print(f"[{_ts()}] \n{sep}", flush=True)
            print(f"[{_ts()}] [NCCL_LOG DRAFT_RECV_PREFILL] input_ids shape={input_ids.shape}, values={input_ids.tolist()}", flush=True)
            print(f"[{_ts()}] [NCCL_LOG DRAFT_RECV_PREFILL] input_ids decoded='{self.tokenizer.decode(input_ids.cpu().tolist())}'", flush=True)
            print(f"[{_ts()}] [NCCL_LOG DRAFT_RECV_PREFILL] num_tokens={num_tokens.tolist()}", flush=True)
            draft_block_table_values_str = compress_neg_ones_and_zeros(f"{draft_block_table.tolist()}")
            print(f"[{_ts()}] [NCCL_LOG DRAFT_RECV_PREFILL] draft_block_table shape={draft_block_table.shape}, values={draft_block_table_values_str}", flush=True)
            print(f"[{_ts()}] [NCCL_LOG DRAFT_RECV_PREFILL] eagle_acts={'None' if eagle_acts is None else f'shape={eagle_acts.shape}'}", flush=True)
            print(f"[{_ts()}] {sep}\n", flush=True)

        prefill_ctxt = self.prepare_prefill_ctxt(num_tokens, draft_block_table)

        if self.config.use_eagle:
            assert eagle_phoenix_act_dim == 3 * self.config.d_model_target, (
                f"EAGLE activation dimension {eagle_phoenix_act_dim} does not match expected dimension 3 * {self.config.d_model_target}"
            )
        elif self.config.use_phoenix:
            assert eagle_phoenix_act_dim == self.config.d_model_target, (
                f"PHOENIX activation dimension {eagle_phoenix_act_dim} does not match expected dimension {self.config.d_model_target}"
            )
        if self.config.verbose:
            print(f'[{_ts()}] [draft_async_prefill] METADATA: total_new_tokens={total_new_tokens}, batch_size={batch_size}, max_blocks={max_blocks}, use_eagle_or_phoenix={use_eagle_or_phoenix}, eagle_phoenix_act_dim={eagle_phoenix_act_dim}', flush=True)


        # 5) set up context exactly like prepare_prefill() does:
        set_context(
            is_prefill=True,
            cu_seqlens_q=prefill_ctxt["cu_seqlens_q"],
            cu_seqlens_k=prefill_ctxt["cu_seqlens_k"],
            max_seqlen_q=prefill_ctxt["max_seqlen_q"],
            max_seqlen_k=prefill_ctxt["max_seqlen_k"],
            slot_mapping=prefill_ctxt["slot_map"],
            context_lens=None,
        ) # , block_tables=block_tables, commenting this out essentially removes prefix caching

        # 6) run the draft model in prefill mode
        positions = prefill_ctxt["positions"]
        self.run_model(input_ids, positions, is_prefill=True, last_only=True, hidden_states=eagle_acts)

        if self.config.verbose:
            print(f'[{_ts()}] [draft_async_prefill] DRAFT ASYNC PREFILL DONE', flush=True)
            # --- KV cache diagnostic ---
            kv = self.kv_cache  # [2, layers, blocks, block_size, heads, dim]
            prefill_slots = prefill_ctxt["slot_map"].long()
            k_norm = kv[0, 0, prefill_slots, 0, :, :].norm().item()
            v_norm = kv[1, 0, prefill_slots, 0, :, :].norm().item()
            print(f'[{_ts()}] [KV_CACHE] After prefill: K norm at slots {prefill_slots.tolist()} = {k_norm:.4f}, V norm = {v_norm:.4f}', flush=True)

        # 7) clean up
        reset_context()

    def _reset_tree_cache_tensors(self):
        """Reset tensor-backed tree cache to empty."""
        # initialize as empty keys on correct device; tokens/logits set to None until first populate
        self.tree_cache_keys = torch.empty(0, 3, dtype=torch.int64, device=self.device)
        self.tree_cache_tokens = None
        self.tree_cache_logits = None
        self.tree_cache_activations = None

    def _init_prealloc_buffers(self):
        # PERFORMANCE: pre-allocate constant tensors used every draft step to avoid repeated CUDA mallocs
        K, MQ_LEN = self.config.speculate_k, self.config.MQ_LEN
        d = self.device
        self._step_pos_offsets = torch.arange(K, device=d, dtype=torch.int64)[:, None] * MQ_LEN
        self._step_rope_offsets = torch.arange(K, device=d, dtype=torch.int64)[:, None]
        self._fan_idx_hit = torch.arange(K + 1, device=d, dtype=torch.int64).repeat_interleave(self.config.fan_out_t)
        self._fan_idx_miss = torch.arange(K + 1, device=d, dtype=torch.int64).repeat_interleave(self.config.fan_out_t_miss)
        self._arange_mq = torch.arange(MQ_LEN, device=d, dtype=torch.int64)
        self._arange_kp1 = torch.arange(K + 1, device=d, dtype=torch.int64)
        self._arange_2kp1 = torch.arange(2 * K + 1, device=d, dtype=torch.int64)

    def jit_speculate(
        self,
        request_keys: torch.Tensor,
        num_tokens: torch.Tensor,
        out_logits: torch.Tensor,
        out_tokens: torch.Tensor,
        temperatures: torch.Tensor,
        draft_block_tables: torch.Tensor,
        extend_activations: torch.Tensor = None,
        extend_counts: torch.Tensor = None,
        extend_token_ids: torch.Tensor = None,
    ):
        input_ids = request_keys[:, -1]
        positions = num_tokens - 1
        context_lens = num_tokens
        # Calculate slot mapping vectorized
        block_idx = positions // self.block_size
        pos_in_block = positions % self.block_size
        batch_indices = torch.arange(input_ids.shape[0], device=self.device)
        slot_map = draft_block_tables[batch_indices, block_idx] * self.block_size + pos_in_block

        hidden_states = None
        spec_activations = None

        if self.config.use_eagle_or_phoenix:
            assert extend_activations is not None and extend_counts is not None and extend_token_ids is not None

            # Phase 1: warm up the draft KV cache with target-activation-based KV for the
            # extend tokens (previously accepted specs from the prior round). Without this,
            # the recovery iter's attention reads stale draft prenorms left at those slots
            # by the previous round's glue decode.
            if int(extend_counts.max().item()) > 0:
                self._jit_extend_warmup(
                    extend_activations, extend_counts, extend_token_ids,
                    num_tokens, draft_block_tables,
                )

            # Recovery activation (target hidden at last accepted token) lives at
            # extend_activations[i, extend_counts[i]] in the new combined layout.
            rec_act = extend_activations[batch_indices, extend_counts.long()]
            if self.config.use_eagle:
                hidden_states = self.model.fc(rec_act.to(self.model.fc.weight.dtype))
            else:
                hidden_states = rec_act
            spec_activations = torch.empty(
                input_ids.shape[0], self.config.speculate_k,
                self.hidden_states_dim,
                dtype=self.hf_config.torch_dtype, device=self.device)

        for i in range(self.config.speculate_k): # we're going to glue after this anyways, and by sending the spec request target has verified we have K more slots left in our last page 
            set_context(
                is_prefill=False,
                slot_mapping=slot_map,
                context_lens=context_lens.to(torch.int32),
                block_tables=draft_block_tables,
                is_jit=True,
            )
            
            if self.config.use_eagle_or_phoenix:
                logits, prenorm = self.run_model(input_ids, positions, is_prefill=False, last_only=True, hidden_states=hidden_states)
                if self.config.use_eagle:
                    spec_activations[:, i] = prenorm
                    hidden_states = prenorm
                else:
                    spec_activations[:, i] = hidden_states
            else:
                logits = self.run_model(input_ids, positions, is_prefill=False, last_only=True)

            if self.config.communicate_logits:
                out_logits[:, i, :] = logits
            reset_context()
            next_tokens = self.sampler(logits, temperatures, is_tree=True)
            out_tokens[:, i] = next_tokens
            
            # Update for next iteration
            input_ids = next_tokens
            positions = positions + 1
            context_lens = context_lens + 1
            # Update slot mapping for next position
            block_idx = positions // self.block_size
            pos_in_block = positions % self.block_size
            slot_map = draft_block_tables[batch_indices, block_idx] * self.block_size + pos_in_block

        return spec_activations

    def _jit_extend_warmup(
        self,
        extend_activations: torch.Tensor,
        extend_counts: torch.Tensor,
        extend_token_ids: torch.Tensor,
        num_tokens: torch.Tensor,
        draft_block_tables: torch.Tensor,
    ):
        """Run a varlen forward pass over the extend tokens to overwrite stale draft KV
        with target-activation-based KV. Convention 1 (slot = token's sequence position):
        extend tokens occupy draft slots [num_tokens-1-n_ext, num_tokens-2] per request.
        Outputs are discarded — KV writes are the side effect.
        """
        B = num_tokens.shape[0]
        K = self.config.speculate_k
        fc_dtype = self.model.fc.weight.dtype if self.config.use_eagle else self.hf_config.torch_dtype

        n_ext = extend_counts.long()  # [B]
        cu_seqlens_q = torch.zeros(B + 1, dtype=torch.int32, device=self.device)
        cu_seqlens_q[1:] = torch.cumsum(n_ext.to(torch.int32), 0)
        total = int(cu_seqlens_q[-1].item())

        # Per-token batch index and local offset within each seq's extend run.
        batch_idx = torch.repeat_interleave(torch.arange(B, device=self.device, dtype=torch.int64), n_ext)
        local_off = torch.arange(total, device=self.device, dtype=torch.int64) - cu_seqlens_q[:-1].long().repeat_interleave(n_ext)

        base_pos = (num_tokens - 1 - n_ext).long()  # [B]
        positions = base_pos[batch_idx] + local_off  # [total]

        block_idx = (positions // self.block_size).clamp(0, draft_block_tables.shape[1] - 1).to(torch.int64)
        block_off = (positions % self.block_size).to(torch.int32)
        blk_ids = draft_block_tables[batch_idx, block_idx]
        slot_map = (blk_ids * self.block_size + block_off).to(torch.int32)

        flat_token_ids = extend_token_ids[batch_idx, local_off]
        flat_act = extend_activations[batch_idx, local_off].to(fc_dtype)
        flat_hidden = self.model.fc(flat_act) if self.config.use_eagle else flat_act

        # After writing, the last extend slot per request sits at sequence pos num_tokens - 2,
        # so context_lens (= valid KV count) is num_tokens - 1.
        context_lens = (num_tokens - 1).to(torch.int32)

        n_ext_max = int(n_ext.max().item())
        set_context(
            is_prefill=False,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=n_ext_max,
            slot_mapping=slot_map,
            context_lens=context_lens,
            block_tables=draft_block_tables,
        )
        self.run_model(flat_token_ids, positions, is_prefill=False, last_only=False, hidden_states=flat_hidden)
        reset_context()

    def hit_cache(self, request_keys, B, K, num_tokens, temperatures, draft_block_tables,
                  extend_activations=None, extend_counts=None, extend_token_ids=None):
        """Hits the cache (tensor-backed) and returns tensors to respond to the spec request."""
        global ttl
        # Draft model now returns full target vocab size logits (after d2t expansion)
        V = self.hf_config.vocab_size

        if self.config.communicate_logits:
            out_logits = torch.full((B, K, V), float('-inf'), dtype=self.hf_config.torch_dtype, device=self.device)
            out_logits[:, :, 0] = 0.0
        else:
            out_logits = None

        out_tokens = torch.zeros(B, K, dtype=torch.int64, device=self.device)
        cache_hits = torch.zeros(B, dtype=torch.bool, device=self.device)

        assert request_keys.shape == (B, 3), f"ERROR in hit_cache: request_keys should be (B, 3), got {request_keys.shape}"

        out_activations = torch.empty(
            B, K, self.hidden_states_dim,
            dtype=self.hf_config.torch_dtype, device=self.device
        ) if self.config.use_eagle_or_phoenix else None

        # Statistics
        ttl += int(B)

        if self.config.verbose:
            print(f"[{_ts()}] [hit_cache] Request keys: {request_keys}", flush=True)
            for i in range(B):
                rec_token = request_keys[i, 2].item()
                rec_text = self.tokenizer.decode([rec_token])
                print(f"[{_ts()}]   Req {i}: token={rec_token} ('{rec_text}')", flush=True)

        if self.tree_cache_keys.numel() > 0:
            # Vectorized membership: broadcast eq on [B,T,3], fuse hit+idx via max()
            eq = (request_keys.unsqueeze(1) == self.tree_cache_keys.unsqueeze(0))  # [B,T,3]
            match = torch.all(eq, dim=2)  # [B,T]
            cache_hits, idx = match.max(dim=1)  # cache_hits: [B] bool, idx: [B] first-match index.

        there_was_a_cache_miss = not cache_hits.all()
        if self.config.force_jit_speculate or (self.config.jit_speculate and there_was_a_cache_miss):
            if self.config.verbose:
                if self.config.force_jit_speculate:
                    msg = "Force JIT speculate, running JIT speculate for all"
                elif self.tree_cache_keys.numel() == 0:
                    msg = "Cache empty, running JIT speculate for all"
                else:
                    assert there_was_a_cache_miss
                    msg = "There was a cache miss, running JIT speculate for all"
                print(f"[{_ts()}] [hit_cache] {msg}", flush=True)
            jit_acts = self.jit_speculate(
                request_keys,
                num_tokens,
                out_logits,
                out_tokens,
                temperatures,
                draft_block_tables,
                extend_activations=extend_activations,
                extend_counts=extend_counts,
                extend_token_ids=extend_token_ids,
                )  # write into out_logits, out_tokens
            if self.config.use_eagle_or_phoenix:
                out_activations = jit_acts
        elif self.tree_cache_keys.numel() > 0:
            if self.config.verbose:
                print(f"[{_ts()}] [hit_cache] Cache hits: {cache_hits.sum().item()}/{B}", flush=True)
                print(f"[{_ts()}] [hit_cache] Cache: {self.tree_cache_keys.shape[0]} entries", flush=True)

                # Build set of hit cache indices for marking
                hit_indices = set()
                for i in range(B):
                    if cache_hits[i]:
                        hit_indices.add(idx[i].item())

                # Print cache entries with hit markers
                for i, key in enumerate(self.tree_cache_keys):
                    seq_id, k_idx, rec_token = key.tolist()
                    rec_text = self.tokenizer.decode([rec_token])
                    hit_marker = "[HIT]" if i in hit_indices else ""
                    print(f"[{_ts()}]     [{i}]: key=({seq_id}, {k_idx}, {rec_token}) -> value=('{rec_text}') {hit_marker}", flush=True)

            # Fill via direct indexing (miss slots get stale cache data, but that's ok since we can
            # return any tokens/logits for cache misses, as long as they are consistent with one another).
            out_tokens = self.tree_cache_tokens[idx]
            if self.config.communicate_logits:
                out_logits = self.tree_cache_logits[idx]
            if self.config.use_eagle_or_phoenix:
                out_activations = self.tree_cache_activations[idx]

        rec_toks = request_keys[:, 2]

        if self.config.verbose:
            print(f"[{_ts()}] [CACHE RESPONSE]", flush=True)
            for i in range(B):
                hit_status = "HIT" if cache_hits[i].item() == 1 else "MISS"
                print(f"[{_ts()}]   Seq {request_keys[i, 0].item()}: {hit_status}", flush=True)
                if cache_hits[i].item() == 1 or self.config.jit_speculate:
                    tokens_list = out_tokens[i, :K].tolist()
                    tokens_text = [self.tokenizer.decode([t]) for t in tokens_list]
                    print(f"[{_ts()}]     Tokens: {tokens_list}", flush=True)
                    print(f"[{_ts()}]     Detokenized: {tokens_text}", flush=True)
            print(f"[{_ts()}] ", flush=True)

        return out_tokens, out_logits, make_glue_decode_input_ids(out_tokens, rec_toks), cache_hits, out_activations

    def _service_spec_request(self):
        """Receives a speculation request, serves it from cache, and sends results back in a single response."""
        DEBUG = os.environ.get("SSD_DEBUG", "0") == "1"
        ev = profile.new_events(4)
        if ev: ev[0].record()

        speculation_request = SpeculationRequest.receive(
            async_pg=self.async_pg,
            target_rank=self.target_rank,
            device=self.device,
            draft_dtype=self.hf_config.torch_dtype,
            tokenizer=self.tokenizer,
            verbose=self.config.verbose,
        )

        B, K, _, _, _ = speculation_request.metadata.tolist()
        cache_keys, num_tokens, draft_block_tables, temperatures = (
            speculation_request.cache_keys,
            speculation_request.num_tokens,
            speculation_request.block_tables,
            speculation_request.temps,
        )
        # extend_activations is the concatenation [extend, recovery] per request:
        # the recovery activation lives at extend_activations[i, extend_counts[i]].
        extend_activations = speculation_request.extend_activations
        extend_counts = speculation_request.extend_counts
        extend_token_ids = speculation_request.extend_token_ids

        if DEBUG:
            # Sanity-check the received payload: a corrupted/torn NCCL receive will surface
            # downstream as a `torch.empty(<huge negative>)` failure inside `_build_tree_batch`.
            # Catch it here, while we still know the bad tensor is fresh off the wire.
            vocab_size = int(speculation_request.metadata[4].item())
            assert num_tokens.dtype == torch.int64, f"num_tokens dtype={num_tokens.dtype}, expected int64"
            assert num_tokens.shape == (B,), f"num_tokens shape={tuple(num_tokens.shape)}, expected ({B},)"
            nt_min = int(num_tokens.min().item())
            nt_max = int(num_tokens.max().item())
            assert nt_min > 0, f"num_tokens has non-positive value: min={nt_min}, num_tokens={num_tokens.tolist()}"
            assert nt_max < (1 << 31), f"num_tokens implausibly large: max={nt_max}, num_tokens={num_tokens.tolist()}"
            if extend_counts is not None:
                assert extend_counts.dtype == torch.int64, f"extend_counts dtype={extend_counts.dtype}, expected int64"
                assert extend_counts.shape == (B,), f"extend_counts shape={tuple(extend_counts.shape)}, expected ({B},)"
                ec_min = int(extend_counts.min().item())
                ec_max = int(extend_counts.max().item())
                assert ec_min >= 0 and ec_max <= K, (
                    f"extend_counts out of range [0,{K}]: min={ec_min}, max={ec_max}, "
                    f"extend_counts={extend_counts.tolist()}"
                )
            if extend_token_ids is not None:
                assert extend_token_ids.dtype == torch.int64, f"extend_token_ids dtype={extend_token_ids.dtype}, expected int64"
                assert extend_token_ids.shape == (B, K + 1), (
                    f"extend_token_ids shape={tuple(extend_token_ids.shape)}, expected ({B},{K+1})"
                )
                # Only the valid prefix [0 .. extend_counts[i]] per row carries meaningful token ids;
                # the rest is uninitialized scratch. Check just the valid prefix.
                if extend_counts is not None and ec_max >= 0:
                    col_idx = torch.arange(K + 1, device=extend_token_ids.device)
                    valid_mask = col_idx[None, :] <= extend_counts[:, None]
                    valid_ids = extend_token_ids[valid_mask]
                    if valid_ids.numel() > 0:
                        tid_min = int(valid_ids.min().item())
                        tid_max = int(valid_ids.max().item())
                        assert 0 <= tid_min and tid_max < vocab_size, (
                            f"extend_token_ids out of [0,{vocab_size}): min={tid_min}, max={tid_max}"
                        )

        if ev: ev[1].record()

        out_tokens, out_logits, glue_decode_input_ids, cache_hits, out_activations = self.hit_cache(
            cache_keys, B, K, num_tokens, temperatures, draft_block_tables,
            extend_activations=extend_activations,
            extend_counts=extend_counts,
            extend_token_ids=extend_token_ids,
        )

        if ev: ev[2].record()

        if self._acceptance_rate_log_path:
            # Collect per-step metrics for logging.
            # cache_keys[:, 1] is last_spec_step_accepted_len - 1 from the target;
            # first request has -1 (forced miss).
            global ttl_hit
            ttl_hit += int(cache_hits.sum().item())
            for i in range(B):
                accept_len = cache_keys[i, 1].item() + 1
                self._acceptance_lengths.append(accept_len)
                self._cache_hits.append(int(cache_hits[i].item()))

        speculation_response = SpeculationResponse(
            speculations=out_tokens.reshape(-1).to(torch.int64),
            cache_hits=cache_hits.reshape(-1).to(torch.int64) if self.communicate_cache_hits else None,
            logits_q=out_logits[:, :K, :].contiguous() if self.communicate_logits else None,
        )
        if BRIEF_LOG:
            for i in range(B):
                cache_hit = cache_hits[i].item()
                # We pretend we are actually sending it, for clarify in debugging.
                cache_hit_text = "HIT" if cache_hit == 1 else "MISS"
                print(f"[{_ts()}] [SpeculationResponse.send] req[{i}]: CACHE {cache_hit_text}", flush=True)

        # Profile-only ACK so the target can split nccl_recv into
        # wait_for_draft vs bulk_recv. Only fires when SSD_PROFILE=1; both
        # sides must agree on the gate or the wire format drifts.
        if profile.is_active():
            if not hasattr(self, "_profile_ack_tensor"):
                self._profile_ack_tensor = torch.ones(
                    1, dtype=torch.int32, device=self.device
                )
            dist.send(
                self._profile_ack_tensor,
                dst=self.target_rank,
                group=self.async_pg,
            )

        speculation_response.send(self.async_pg, self.target_rank, tokenizer=self.tokenizer)

        if ev: ev[3].record()
        # Per-iter meta for downstream analysis:
        #   cache_hit:    number of requests in this batch whose prior speculation
        #                 was found in the draft's tree cache (0..B).
        #   accepted_prev: total tokens the target accepted from the *previous*
        #                 spec iter (across the batch). Decoded from
        #                 cache_keys[:, 1], which carries (accept_length - 1)
        #                 from the target. The first iter has -1 as a sentinel
        #                 ("no prior acceptance"); we clamp to 0 in that case.
        # Gated on `ev` so the two .item() syncs (≈10-20μs total) only fire
        # when profiling is on.
        _meta = {"B": B, "K": K}
        if ev is not None:
            _meta["cache_hit"] = int(cache_hits.sum().item())
            _meta["accepted_prev"] = int(
                (cache_keys[:, 1] + 1).clamp(min=0).sum().item()
            )
        profile.emit(
            "draft._service_spec_request",
            ["receive", "hit_cache", "send"],
            ev,
            **_meta,
        )

        if NCCL_LOG:
            sep = '=' * 80
            print(f"[{_ts()}] \n{sep}", flush=True)
            for i in range(B):
                spec_ids = out_tokens[i, :K].tolist()
                spec_text = [self.tokenizer.decode([t]) for t in spec_ids]
                print(f"[{_ts()}]   req[{i}]: speculations={spec_ids}", flush=True)
                print(f"[{_ts()}]            decoded={spec_text}", flush=True)
            print(f"[{_ts()}] {sep}\n", flush=True)

        partial_tree_decode_args = {
            "num_tokens": num_tokens,
            "seq_ids": speculation_request.cache_keys[:, 0],
            "temperatures": temperatures,
            "dbt": draft_block_tables,
            "cache_hits": cache_hits,
            "returned_tokens": out_tokens,
            "previous_activations": out_activations,
            "extend_counts": extend_counts,
            # extend_eagle_acts has shape (B, K+1, dim); slot extend_counts[i] holds the recovery activation.
            "extend_eagle_acts": extend_activations,
            "extend_token_ids": extend_token_ids,
        }
        return glue_decode_input_ids, partial_tree_decode_args

    def prepare_prefill_ctxt(
        self,
        num_tokens: torch.Tensor,  # [B]
        draft_block_table: torch.Tensor,  # [B, max_blocks]
    ) -> dict:
        """
        Prepare context for prefill forward pass.
        """
        B = num_tokens.shape[0]
        total = num_tokens.sum().item()
        cu_seqlens_q = torch.zeros(B + 1, dtype=torch.int32, device=self.device)
        cu_seqlens_q[1:] = torch.cumsum(num_tokens, dim=0)
        batch_indices = torch.arange(B, device=self.device, dtype=torch.int64).repeat_interleave(num_tokens)
        positions = torch.arange(total, device=self.device, dtype=torch.int64) - cu_seqlens_q[:-1].to(torch.int64).repeat_interleave(num_tokens)
        max_seqlen_q = num_tokens.max().item()

        # Calculate block indices and offsets for ALL positions
        block_indices = (positions // self.block_size).to(torch.int64)
        offsets = (positions % self.block_size).to(torch.int32)

        # Get block IDs for each position from dbt
        block_ids = draft_block_table[batch_indices, block_indices]

        # Calculate slot_map for each position
        slot_map = (block_ids * self.block_size + offsets).to(torch.int32)

        return {
            "positions": positions,
            "slot_map": slot_map,
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_q.clone(),
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_k": max_seqlen_q,
        }

    
    def prepare_glue_decode_ctxt(self, num_tokens, input_ids, dbt, B):
        K = self.config.speculate_k
        positions_start = (num_tokens - 1).unsqueeze(-1)
        positions_grid = positions_start + self._arange_kp1

        # Calculate block indices and offsets for ALL positions
        block_indices = (positions_grid // self.block_size).to(torch.int64)
        offsets = (positions_grid % self.block_size).to(torch.int32)

        # Get block IDs for each position from dbt
        B_expanded = torch.arange(B, device=self.device).unsqueeze(-1).expand(-1, K + 1)
        blk_ids = dbt[B_expanded, block_indices]

        # Calculate slot_map for each position
        slot_map_grid = blk_ids * self.block_size + offsets

        # Flattened tensors for varlen decode
        positions_flat = positions_grid.reshape(-1).to(torch.int64)
        slot_map_flat = slot_map_grid.reshape(-1).to(torch.int32)

        context_lens = (num_tokens + K).to(torch.int32)
        seqlen_q = torch.full((B,), K + 1, dtype=torch.int32, device=self.device)
        cu_seqlens_q = torch.zeros(B + 1, dtype=torch.int32, device=self.device)
        cu_seqlens_q[1:] = torch.cumsum(seqlen_q, dim=0)

        return {
            "input_ids": input_ids,
            "positions": positions_flat,
            "slot_map": slot_map_flat,
            "cu_seqlens_q": cu_seqlens_q,
            "max_seqlen_q": K + 1,
            "context_lens": context_lens,
            "block_tables": dbt,
        }

    def prepare_glue_decode_ctxt_eagle(self, num_tokens, fused_ids, fused_hs, extend_counts, seqlens_q, cu_seqlens_q, dbt, B):
        """Prepare context for EAGLE glue decode with FA varlen causal.

        Tokens packed contiguously: [ext_0..ext_{n0-1}, rec_0, spec_0..spec_{K-1}, ext_1..., ...]
        No padding within sequences. cu_seqlens_q has variable per-seq lengths.
        """
        K = self.config.speculate_k
        total_real = int(cu_seqlens_q[-1].item())

        # Per-token batch index and local offset within each seq
        batch_idx = torch.repeat_interleave(torch.arange(B, device=self.device), seqlens_q)  # [total_real]
        local_off = torch.arange(total_real, device=self.device) - cu_seqlens_q[:-1].long().repeat_interleave(seqlens_q)

        # Positions: extend starts at num_tokens - 1 - n_ext, then rec, then spec
        # base_pos[b] = num_tokens[b] - 1 - extend_counts[b] (position of first extend token)
        base_pos = (num_tokens - 1 - extend_counts).long()  # [B]
        positions = (base_pos[batch_idx] + local_off).to(torch.int64)

        # Context lens: last token (spec K-1) at pos num_tokens+K-1, cache has 0..num_tokens+K-1
        context_lens = (num_tokens + K).to(torch.int32)

        # Slot mapping
        block_idx = (positions // self.block_size).clamp(0, dbt.shape[1] - 1).to(torch.int64)
        block_off = (positions % self.block_size).to(torch.int32)
        blk_ids = dbt[batch_idx, block_idx]
        slot_map = (blk_ids * self.block_size + block_off).to(torch.int32)

        return {
            "input_ids": fused_ids,
            "positions": positions,
            "slot_map": slot_map,
            "hidden_states": fused_hs,
            "cu_seqlens_q": cu_seqlens_q,
            "max_seqlen_q": 2 * K + 1,
            "context_lens": context_lens,
            "block_tables": dbt,
        }

    def _construct_tree_decode_args(self, partial_tree_decode_args, rec_flat, dbt):
        # tree decode needs (input_ids, positions) that are [N], wrapper plan handles batch size of attn computation 
        # rec_flat is [N]
        
        B = dbt.shape[0]
        K = self.config.speculate_k
        F = self.config.async_fan_out
        N = rec_flat.shape[0]
        cache_hits = partial_tree_decode_args["cache_hits"]

        if __debug__:
            assert N == B*self.config.MQ_LEN, f"ERROR in _construct_tree_decode_args: N should be B*self.config.MQ_LEN={B*self.config.MQ_LEN}, got {N}"

        b_flat = torch.arange(B, device=self.device, dtype=torch.int64)[:, None].expand(B, self.config.MQ_LEN).flatten()
        fkp1_flat = self._arange_mq.repeat(B)
        j_idx_flat = torch.cat([self._fan_idx_hit if hit else self._fan_idx_miss for hit in cache_hits])
        metadata = torch.tensor([B, K, F, N], dtype=torch.int64, device=self.device)

        seq_ids = partial_tree_decode_args["seq_ids"]
        seq_ids_expanded = seq_ids[b_flat]
        positions = (partial_tree_decode_args["num_tokens"][b_flat] - 1) + (K + 1) + fkp1_flat
        rope_positions = (partial_tree_decode_args["num_tokens"][b_flat] - 1) + j_idx_flat + 1
        temperatures = partial_tree_decode_args["temperatures"][b_flat]

        tree_decode_args = {
            "metadata": metadata,
            "input_ids": rec_flat,  # [N]
            "positions": positions,  # [N]
            "rope_positions": rope_positions, # [N], these are to be passed into model fwd 
            # the dbt is now [B, M] in the seq fan out codebase
            "block_tables": dbt,
            "temps": temperatures,  # [N]
            "rec_flat": rec_flat,  # [N]
            "seq_ids_expanded": seq_ids_expanded,  # [N]
            "cache_hits": cache_hits,  # [B] # we also want returned_tokens which is [B, K]
        }

        return tree_decode_args

    def _build_tree_batch(self, partial_tree_decode_args, glue_decode_input_ids):
        if self.config.verbose:
            print(f'[{_ts()}] about to build tree batch')
        K = self.config.speculate_k
        dbt = partial_tree_decode_args["dbt"]
        cache_hits = partial_tree_decode_args["cache_hits"]
        cache_hits_list = cache_hits.tolist()

        ev = profile.new_events(7)
        if ev: ev[0].record()

        if self.config.use_eagle_or_phoenix:
            B = partial_tree_decode_args["num_tokens"].shape[0]
            extend_counts = partial_tree_decode_args.get("extend_counts")
            if extend_counts is None:
                extend_counts = torch.zeros(B, dtype=torch.int64, device=self.device)
            # extend_eagle_acts_batch has shape (B, K+1, dim): the recovery activation lives
            # at extend_eagle_acts_batch[i, extend_counts[i]] (one past the original extends).
            extend_eagle_acts_batch = partial_tree_decode_args.get("extend_eagle_acts")
            extend_token_ids_batch = partial_tree_decode_args.get("extend_token_ids")
            prev_acts = partial_tree_decode_args["previous_activations"]
            hidden_size = self.hidden_states_dim
            fc_dtype = self.model.fc.weight.dtype if self.config.use_eagle else self.hf_config.torch_dtype

            gd_view = glue_decode_input_ids.view(B, K + 1)
            spec_tok_ids = gd_view[:, 1:]

            # Check if all extend counts are the same (common case) for vectorized fast path
            n_ext_0 = int(extend_counts[0].item())
            uniform_extends = (B == 1) or (extend_counts == n_ext_0).all().item()

            if uniform_extends:
                # ── Fast path: regular layout (all seqs have same length) ──
                # Layout per seq: [ext_0, ..., ext_{n-1}, rec, spec_0, ..., spec_{K-1}]
                sl = n_ext_0 + K + 1   # uniform sequence length
                total_real = B * sl
                fused_ids = torch.empty(total_real, dtype=torch.int64, device=self.device)
                fused_hs = torch.empty(total_real, hidden_size, dtype=self.hf_config.torch_dtype, device=self.device)
                fid_v = fused_ids.view(B, sl)
                fhs_v = fused_hs.view(B, sl, hidden_size)

                # Extend + recovery tokens: positions 0..n_ext_0 (need fc / target acts).
                # extend_eagle_acts_batch[:, :n_ext_0+1] and extend_token_ids_batch[:, :n_ext_0+1]
                # both hold [extend_0..extend_{n-1}, recovery].
                fid_v[:, :n_ext_0 + 1] = extend_token_ids_batch[:, :n_ext_0 + 1]
                fc_in = extend_eagle_acts_batch[:, :n_ext_0 + 1].reshape(B * (n_ext_0 + 1), -1).to(fc_dtype)
                if self.config.use_eagle:
                    fc_out = self.model.fc(fc_in)
                else:
                    fc_out = fc_in  # Phoenix: no fc, use activations directly
                fhs_v[:, :n_ext_0 + 1, :] = fc_out.view(B, n_ext_0 + 1, hidden_size)

                # Spec tokens: positions n_ext_0+1..sl-1 (no fc needed)
                fid_v[:, n_ext_0 + 1:] = spec_tok_ids
                fhs_v[:, n_ext_0 + 1:, :] = prev_acts

                # cu_seqlens_q: regular spacing
                cu_seqlens_q = (torch.arange(B + 1, device=self.device, dtype=torch.int32) * sl)
                seqlens_q = torch.full((B,), sl, device=self.device, dtype=torch.int32)

                # Positions and slot mapping via arange arithmetic (no repeat_interleave)
                tok_idx = torch.arange(total_real, device=self.device, dtype=torch.int64)
                batch_idx_fast = tok_idx // sl
                local_off_fast = tok_idx % sl
                base_pos = (partial_tree_decode_args["num_tokens"] - 1 - n_ext_0).long()
                positions = base_pos[batch_idx_fast] + local_off_fast
                context_lens = (partial_tree_decode_args["num_tokens"] + K).to(torch.int32)
                block_idx = (positions // self.block_size).clamp(0, dbt.shape[1] - 1).to(torch.int64)
                block_off = (positions % self.block_size).to(torch.int32)
                blk_ids = dbt[batch_idx_fast, block_idx]
                slot_map = (blk_ids * self.block_size + block_off).to(torch.int32)

                glue_decode_ctxt = {
                    "input_ids": fused_ids,
                    "positions": positions,
                    "slot_map": slot_map,
                    "hidden_states": fused_hs,
                    "cu_seqlens_q": cu_seqlens_q,
                    "max_seqlen_q": sl,
                    "context_lens": context_lens,
                    "block_tables": dbt,
                }
            else:
                # ── Fallback: variable-length layout (repeat_interleave + boolean masks) ──
                seqlens_q = (extend_counts + K + 1).to(torch.int32)
                cu_seqlens_q = torch.zeros(B + 1, dtype=torch.int32, device=self.device)
                cu_seqlens_q[1:] = torch.cumsum(seqlens_q, 0)
                total_real = int(cu_seqlens_q[-1].item())

                fused_ids = torch.empty(total_real, dtype=torch.int64, device=self.device)
                fused_hs = torch.empty(total_real, hidden_size, dtype=self.hf_config.torch_dtype, device=self.device)

                batch_idx = torch.repeat_interleave(torch.arange(B, device=self.device), seqlens_q)
                local_off = torch.arange(total_real, device=self.device) - cu_seqlens_q[:-1].long().repeat_interleave(seqlens_q)
                n_ext = extend_counts.long()
                n_ext_per_tok = n_ext[batch_idx]

                # is_extend now covers both the original extend slots AND the recovery slot
                # (recovery sits at local_off == n_ext_per_tok within the new combined buffers).
                is_extend = local_off <= n_ext_per_tok
                is_spec = ~is_extend

                ext_b = batch_idx[is_extend]
                ext_local = local_off[is_extend]
                tc_acts = extend_eagle_acts_batch[ext_b, ext_local].to(fc_dtype)
                fused_ids[is_extend] = extend_token_ids_batch[ext_b, ext_local]

                if self.config.use_eagle:
                    fused_hs[is_extend] = self.model.fc(tc_acts)
                elif self.config.use_phoenix:
                    fused_hs[is_extend] = tc_acts

                spec_j = local_off[is_spec] - n_ext_per_tok[is_spec] - 1
                fused_ids[is_spec] = spec_tok_ids[batch_idx[is_spec], spec_j]
                fused_hs[is_spec] = prev_acts[batch_idx[is_spec], spec_j]

                glue_decode_ctxt = self.prepare_glue_decode_ctxt_eagle(
                    num_tokens=partial_tree_decode_args["num_tokens"],
                    fused_ids=fused_ids, fused_hs=fused_hs,
                    extend_counts=extend_counts, seqlens_q=seqlens_q,
                    cu_seqlens_q=cu_seqlens_q, dbt=dbt, B=B,
                )
        else:
            # Non-EAGLE: K+1 per seq, uses verify CG path
            B = glue_decode_input_ids.shape[0] // (K + 1)
            assert B == partial_tree_decode_args["num_tokens"].shape[0]
            glue_decode_ctxt = self.prepare_glue_decode_ctxt(
                num_tokens=partial_tree_decode_args["num_tokens"],
                input_ids=glue_decode_input_ids,
                dbt=dbt, B=B,
            )

        if ev: ev[1].record()

        # Pre-compute tree decode args (overlap CPU with GPU)
        _pre_b_flat = torch.arange(B, device=self.device, dtype=torch.int64)[:, None].expand(B, self.config.MQ_LEN).flatten()
        _pre_fkp1_flat = self._arange_mq.repeat(B)
        _pre_j_idx_flat = torch.cat([self._fan_idx_hit if int(h) else self._fan_idx_miss for h in cache_hits_list])
        N_pre = _pre_b_flat.shape[0]
        _pre_metadata_ints = (B, K, self.config.async_fan_out, N_pre)
        _pre_seq_ids_expanded = partial_tree_decode_args["seq_ids"][_pre_b_flat]
        _pre_positions = (partial_tree_decode_args["num_tokens"][_pre_b_flat] - 1) + (K + 1) + _pre_fkp1_flat
        _pre_rope_positions = (partial_tree_decode_args["num_tokens"][_pre_b_flat] - 1) + _pre_j_idx_flat + 1
        _pre_temperatures = partial_tree_decode_args["temperatures"][_pre_b_flat]

        # --- Run glue decode forward ---
        set_context(
            is_prefill=False,
            cu_seqlens_q=glue_decode_ctxt["cu_seqlens_q"],
            max_seqlen_q=glue_decode_ctxt["max_seqlen_q"],
            slot_mapping=glue_decode_ctxt["slot_map"],
            context_lens=glue_decode_ctxt["context_lens"],
            block_tables=glue_decode_ctxt["block_tables"],
        )

        if ev: ev[2].record()

        glue_prenorm = None
        if self.config.use_eagle_or_phoenix:
            fused_hs_flat = glue_decode_ctxt["hidden_states"]
            glue_decode_logits_flat, glue_prenorm = self.run_model(
                glue_decode_ctxt["input_ids"], glue_decode_ctxt["positions"],
                is_prefill=False, last_only=False, hidden_states=fused_hs_flat)
        else:
            glue_decode_logits_flat = self.run_model(
                glue_decode_ctxt["input_ids"], glue_decode_ctxt["positions"],
                is_prefill=False, last_only=False)

        if ev: ev[3].record()

        if self.config.verbose:
            print(f"[{_ts()}] [GLUE DECODE] logits shape={glue_decode_logits_flat.shape}, "
                  f"max={glue_decode_logits_flat.max().item():.4f}, "
                  f"min={glue_decode_logits_flat.min().item():.4f}, "
                  f"mean={glue_decode_logits_flat.mean().item():.6f}", flush=True)

        reset_context()

        if ev: ev[4].record()

        # --- Extract K+1 logits/prenorms at rec+spec positions ---
        if self.config.use_eagle_or_phoenix:
            # Packed layout: rec at cu_seqlens_q[b] + n_ext[b], spec follows
            cu_q = glue_decode_ctxt["cu_seqlens_q"]
            rec_offsets = cu_q[:-1].long() + extend_counts.long()  # [B]
            extract_idx = rec_offsets.unsqueeze(1) + self._arange_kp1.unsqueeze(0)  # [B, K+1]
            flat_idx = extract_idx.flatten()
            glue_decode_logits = glue_decode_logits_flat[flat_idx].view(B, K + 1, -1)
            if glue_prenorm is not None:
                glue_prenorm_kp1 = glue_prenorm[flat_idx].view(B, K + 1, -1)
        else:
            glue_decode_logits = glue_decode_logits_flat.view(B, K + 1, -1)
            if glue_prenorm is not None:
                glue_prenorm_kp1 = glue_prenorm.view(B, K + 1, -1)

        # --- Build tree hidden states from K+1 prenorms ---
        tree_hidden_states = None
        if glue_prenorm is not None:
            assert self.config.use_eagle_or_phoenix, "ERROR in _build_tree_batch: use_eagle_or_phoenix must be True when glue_prenorm is not None."
            # Vectorized: for each (b, depth), repeat prenorm by fan_out[depth]
            # fan_out_t[depth] for hits, fan_out_t_miss[depth] for misses
            fan_hit = self.config.fan_out_t  # [K+1]
            fan_miss = self.config.fan_out_t_miss  # [K+1]
            # Per-batch fan_out: [B, K+1]
            per_batch_fan = torch.where(
                cache_hits.bool().unsqueeze(1).expand(B, K + 1),
                fan_hit.unsqueeze(0).expand(B, K + 1),
                fan_miss.unsqueeze(0).expand(B, K + 1),
            )  # [B, K+1]
            reps_flat = per_batch_fan.reshape(-1)  # [B*(K+1)]

            if self.config.use_eagle:
                prenorms_flat = glue_prenorm_kp1.reshape(B * (K + 1), -1)   # [B*(K+1), d]
                tree_hidden_states = torch.repeat_interleave(prenorms_flat, reps_flat, dim=0)
            else:
                assert self.config.use_phoenix
                # Phoenix conditions on target activations (recovery slot), not prenorms.
                # The recovery activation per request is at extend_eagle_acts_batch[i, extend_counts[i]].
                rec_acts = extend_eagle_acts_batch[torch.arange(B, device=self.device), extend_counts.long()]  # [B, target_dim]
                target_acts_expanded = rec_acts.unsqueeze(1).expand(B, K + 1, -1)  # [B, K+1, target_dim]
                acts_flat = target_acts_expanded.reshape(B * (K + 1), -1)  # [B*(K+1), target_dim]
                tree_hidden_states = torch.repeat_interleave(acts_flat, reps_flat, dim=0)

        # --- Fork tokens from K+1 logits ---
        # Need [B, K+1] input_ids for forking (rec + spec tokens)
        if self.config.use_eagle_or_phoenix:
            gd_for_fork = gd_view  # [B, K+1] already computed above
        else:
            gd_for_fork = glue_decode_input_ids.reshape(B, K + 1)

        if ev: ev[5].record()

        forked_rec_tokens = get_forked_recovery_tokens_from_logits(
            self.config,
            glue_decode_logits,
            cache_hits,
            gd_for_fork,
            tokenizer=self.tokenizer,
        ).view(-1)

        if ev: ev[6].record()
        profile.emit(
            "draft._build_tree_batch",
            [
                "prepare_glue_decode_ctxt",
                "set_context",
                "run_model",
                "reset_context",
                "prepare_get_forked_recovery_tokens",
                "get_forked_recovery_tokens",
            ],
            ev,
            B=B,
            K=K,
        )
        # Phoenix tree decode reuses the recovery activation per request.
        # In the new layout it lives at extend_eagle_acts_batch[i, extend_counts[i]].
        if self.config.use_phoenix:
            ext_acts_p = partial_tree_decode_args["extend_eagle_acts"]
            ext_counts_p = partial_tree_decode_args["extend_counts"]
            target_recovery_activations = ext_acts_p[
                torch.arange(ext_acts_p.shape[0], device=self.device), ext_counts_p.long()
            ]
            # Expand (B, dim) → (N=B*MQ_LEN, dim). Every branch of the tree decode
            # needs to see the same recovery activation; without this expansion,
            # _decode_tree_step stores a (B, dim) tensor into payload["hidden_states"]
            # for depths >= 1, and run_tree_decode_cudagraph silently zero-pads it
            # to (N, dim), poisoning branches 1..N-1 with zero hidden states.
            target_recovery_activations = target_recovery_activations.repeat_interleave(
                self.config.MQ_LEN, dim=0
            )
        else:
            target_recovery_activations = None
        tree_decode_args = {
            "metadata_ints": _pre_metadata_ints,
            "input_ids": forked_rec_tokens,
            "positions": _pre_positions,
            "rope_positions": _pre_rope_positions,
            "block_tables": dbt,
            "temps": _pre_temperatures,
            "rec_flat": forked_rec_tokens,
            "seq_ids_expanded": _pre_seq_ids_expanded,
            "cache_hits": cache_hits,
            "cache_hits_list": cache_hits_list,
            "target_recovery_activations": target_recovery_activations,
        }
        tree_decode_args["hidden_states"] = tree_hidden_states
        return tree_decode_args

    @torch.inference_mode()
    def _compute_step_positions_and_slot_maps(self, initial_positions, initial_rope_positions, dbt, B, K, F, N, MQ_LEN):
        # PERFORMANCE: pre-allocated _step_pos_offsets/_step_rope_offsets avoid per-step torch.arange calls
        step_positions = initial_positions[None, :] + self._step_pos_offsets
        step_rope_positions = initial_rope_positions[None, :] + self._step_rope_offsets
        step_context_lens = step_positions.view(K, B, MQ_LEN)[:, :, -1] + 1

        # Precompute slot_maps for all steps: [K, N]
        b_flat = torch.arange(B, device=self.device, dtype=torch.int64)[
            :, None].expand(B, self.config.MQ_LEN).flatten()
        batch_indices = torch.arange(N, device=self.device)
        dbt_expanded = dbt[b_flat]  # [N, M] - constant across steps

        step_offsets = (step_positions % self.block_size).to(torch.int32)  # [K, N]
        step_last_blks = (step_positions // self.block_size).to(torch.int64)  # [K, N]
        step_blk_ids = dbt_expanded[batch_indices[None, :], step_last_blks]  # [K, N]
        step_slot_maps = step_blk_ids * self.block_size + step_offsets  # [K, N]

        return step_positions, step_rope_positions, step_context_lens, step_slot_maps

    def _decode_tree_step(self, depth, current_input_ids, step_rope_positions, step_slot_maps, step_context_lens, dbt, payload, spec_tokens, spec_logits, spec_activations, target_recovery_activations):
        """Execute a single tree decode step."""
        # Use precomputed values for this step
        set_context(
            is_prefill=False,
            slot_mapping=step_slot_maps[depth],
            context_lens=step_context_lens[depth].to(torch.int32),
            block_tables=dbt,
        )

        hidden_states = payload.get("hidden_states")
        if self.config.use_eagle_or_phoenix:
            # Diagnostic: Phoenix bug was that payload["hidden_states"] was stored as
            # (B, dim) for depths >= 1, then silently zero-padded to (N, dim) by
            # run_tree_decode_cudagraph — poisoning branches 1..N-1. With the fix in
            # _build_tree_batch (target_recovery_activations is now repeat_interleaved
            # to (N, dim)), depths 0 AND >=1 should both report shape (N, dim) here.
            # Print once per unique (depth, hs_shape) so logs stay terse.
            if not hasattr(self, "_phnx_hs_shapes_logged"):
                self._phnx_hs_shapes_logged = set()
            _hs_shape = tuple(hidden_states.shape) if hidden_states is not None else None
            _key = (depth, _hs_shape, tuple(current_input_ids.shape))
            if _key not in self._phnx_hs_shapes_logged:
                self._phnx_hs_shapes_logged.add(_key)
                print(
                    f"[{_ts()}] [PHX_DIAG] _decode_tree_step depth={depth} "
                    f"current_input_ids.shape={tuple(current_input_ids.shape)} "
                    f"payload['hidden_states'].shape={_hs_shape} "
                    f"use_phoenix={self.config.use_phoenix} MQ_LEN={self.config.MQ_LEN}",
                    flush=True,
                )
            logits, prenorm = self.run_model(current_input_ids, step_rope_positions[depth], is_prefill=False, last_only=False, tree_decode_step=depth, cache_hits=payload["cache_hits"], hidden_states=hidden_states)
            assert spec_activations is not None
            if self.config.use_eagle:
                spec_activations[:, depth] = prenorm
                payload["hidden_states"] = prenorm
            else:
                spec_activations[:, depth] = target_recovery_activations
                payload["hidden_states"] = target_recovery_activations
        else:
            logits = self.run_model(current_input_ids, step_rope_positions[depth], is_prefill=False, last_only=False, tree_decode_step=depth, cache_hits=payload["cache_hits"])
        
        reset_context()
        
        V = self.hf_config.vocab_size  # Draft returns full target vocab size after d2t expansion
        logits_flat = logits.view(-1, V)  # [N, V]
        spec_logits[:, depth, :] = logits_flat
        # Inline greedy: payload["_all_greedy"] checked once in _decode_tree
        next_tokens = logits_flat.argmax(dim=-1) if payload["_all_greedy"] else self.sampler(logits_flat, payload["temps"], is_tree=True)
        spec_tokens[:, depth] = next_tokens
        
        return next_tokens

    def _decode_tree(self, payload):
        """Decodes the speculation tree, checking for interrupts at each step."""

        # setup
        B, K, F, N = payload["metadata_ints"]

        V = self.hf_config.vocab_size  # Draft returns full target vocab size after d2t expansion
        spec_tokens = torch.empty(
            N, K, dtype=torch.int64, device=self.device)
        spec_logits = torch.empty(
            N, K, V, dtype=self.hf_config.torch_dtype, device=self.device)
        spec_activations = torch.empty(
            N, K, self.hidden_states_dim,
            dtype=self.hf_config.torch_dtype, device=self.device
        ) if self.config.use_eagle_or_phoenix else None

        # Precompute all positions, context_lens, and slot_maps for all K steps
        # PERFORMANCE: no .clone() needed — these are not modified in-place
        initial_positions = payload["positions"]  # [N]
        initial_rope_positions = payload["rope_positions"]  # [N]
        current_input_ids = payload["input_ids"]  # [N], the forked tokens
        dbt = payload["block_tables"]  # [B, M] - constant across steps
        target_recovery_activations = payload["target_recovery_activations"]
        
        # Use compiled function for batch-size independent computations
        _, step_rope_positions, step_context_lens, step_slot_maps = self._compute_step_positions_and_slot_maps(
            initial_positions, initial_rope_positions, dbt, B, K, F, N, self.config.MQ_LEN
        )

        payload["_all_greedy"] = bool((payload["temps"] == 0).all())
        ev = profile.new_events(K + 1)
        if ev: ev[0].record()
        for depth in range(K):
            current_input_ids = self._decode_tree_step(
                depth, current_input_ids, step_rope_positions, step_slot_maps,
                step_context_lens, dbt, payload, spec_tokens, spec_logits, spec_activations, target_recovery_activations,
            )
            if ev: ev[depth + 1].record()
        profile.emit(
            "draft._decode_tree",
            [f"step_{i}" for i in range(K)],
            ev,
            B=B,
            K=K,
        )

        return spec_tokens, spec_logits, spec_activations

    def _populate_tree_cache(self, payload, tokens, logits, activations=None):
        """Populates the tensor-backed tree_cache with the results of the decoding.
        """
        seq_ids_expanded = payload["seq_ids_expanded"].to(torch.int64)
        rec_flat = payload["rec_flat"].to(torch.int64)

        k_flat = torch.cat([self._fan_idx_hit if hit else self._fan_idx_miss for hit in payload["cache_hits_list"]])

        assert k_flat.shape[0] == payload["block_tables"].shape[0] * self.config.MQ_LEN, f"ERROR in _populate_tree_cache: k_flat should be {payload['block_tables'].shape[0] * self.config.MQ_LEN}, got {k_flat.shape[0]}"
        
        keys = torch.stack([seq_ids_expanded, k_flat, rec_flat], dim=1).contiguous()  # [N,3]

        assert self.tree_cache_keys.numel() == 0
        self.tree_cache_keys = keys
        self.tree_cache_tokens = tokens
        self.tree_cache_logits = logits
        self.tree_cache_activations = activations
        
        # Print cache population details
        if self.config.verbose:
            N = keys.shape[0]
            print(f"[{_ts()}] \n{'='*80}", flush=True)
            print(f"[{_ts()}] [CACHE POPULATED] {N} entries", flush=True)
            
            # Show sample entries per sequence
            for seq_id in keys[:, 0].unique()[:1]:  # Just show first sequence
                seq_mask = keys[:, 0] == seq_id
                seq_entries = keys[seq_mask]
                seq_tokens = tokens[seq_mask]
                
                print(f"[{_ts()}]   Seq {seq_id.item()}: {seq_mask.sum().item()} entries", flush=True)
                
                # Show first 2 unique recovery tokens
                for rec_token in seq_entries[:, 2].unique()[:2]:
                    rec_mask = seq_entries[:, 2] == rec_token
                    if rec_mask.any():
                        idx = rec_mask.nonzero(as_tuple=True)[0][0]
                        k_idx = seq_entries[idx, 1].item()
                        
                        rec_text = self.tokenizer.decode([rec_token.item()])
                        spec_tokens = seq_tokens[idx].tolist()
                        spec_text = [self.tokenizer.decode([t]) for t in spec_tokens]
                        print(f"[{_ts()}]     k={k_idx}, rec={rec_token.item()} ('{rec_text}') -> {spec_text}", flush=True)
            print(f"[{_ts()}] {'='*80}\n", flush=True)

    def _start_interrupt_listener(self):
        """Initiates a non-blocking receive for the next command to allow interruption."""
        cmd_tensor = torch.empty(1, dtype=torch.int64, device=self.device)
        work_handle = dist.irecv(cmd_tensor, src=0, group=self.async_pg)
        # return both the handle and its tensor buffer
        return work_handle, cmd_tensor

    # new one, with true asynchrony
    def draft_loop(self):
        """
        Runs the asynchronous draft model loop. 
        Handles three commands:
          1 = prefill, 0 = spec request, 2 = exit, 3 = branch prefetch (only after a spec request).
        """
        assert self.draft_async, "draft_loop only runs in async-draft mode"

        try:
            self._draft_loop_inner()
        except (torch.distributed.DistBackendError, RuntimeError) as e:
            err = str(e)
            if "closed" in err or "Connection" in err or "NCCL" in err:
                print(f"[{_ts()}] [draft] Target disconnected, shutting down gracefully.", flush=True)
                self.exit()
                return
            print(f"[{_ts()}] [draft] Error in draft_loop: {e}", flush=True)
            raise e
        except Exception as e:
            print(f"[{_ts()}] [draft] Error in draft_loop: {e}", flush=True)
            raise e

    def _draft_loop_inner(self):
        while True:
            # 1) Wait for the next command (may be PREFILL, SPEC_REQUEST, or EXIT).
            # Bracket the CPU-blocking wait with CPU events so we can see how
            # long the draft sat idle between iters — high when target is
            # gating, ~0 when draft is gating. CUDA events would not capture
            # this region (idle stream).
            ev_idle = profile.new_cpu_events(2)
            if ev_idle: ev_idle[0].record()
            cmd, _ = self._wait_for_cmd()
            if ev_idle: ev_idle[1].record()

            # PREFILL: run the draft prefill and then loop back
            if cmd == COMMAND.PREFILL:
                self.draft_async_prefill()
                continue

            # SPECULATE request: serve out-of-cache or random speculations
            elif cmd == COMMAND.SPECULATION:
                _ds0 = time.perf_counter()
                ev = profile.new_events(5)
                if ev: ev[0].record()

                glue_decode_input_ids, partial_tree_decode_args = self._service_spec_request()
                if ev: ev[1].record()

                self._reset_tree_cache_tensors()
                tree_decode_args = self._build_tree_batch(partial_tree_decode_args, glue_decode_input_ids)
                if ev: ev[2].record()

                # Decode the branch tree
                tokens, logits, activations = self._decode_tree(tree_decode_args)
                if ev: ev[3].record()

                # Populate the local cache so future spec-requests can hit
                self._populate_tree_cache(tree_decode_args, tokens, logits, activations)
                self._draft_step_times.append(time.perf_counter() - _ds0)
                if ev: ev[4].record()

                profile.emit(
                    "draft.spec_iter",
                    ["service", "build_tree", "decode_tree", "populate"],
                    ev,
                    B=int(partial_tree_decode_args["num_tokens"].shape[0]),
                )
                profile.emit(
                    "draft.idle",
                    ["wait_for_cmd"],
                    ev_idle,
                )
                # Single sync + aggregated print for everything emitted this iteration:
                # _service_spec_request, _build_tree_batch, _decode_tree, run_*_cudagraph,
                # model forward/compute_logits (eager), etc.
                profile.flush()

                continue

            # EXIT: clean up and break out of the loop
            elif cmd == COMMAND.DRAFT_EXIT:
                if self._draft_step_times:
                    avg_ms = sum(self._draft_step_times) * 1000 / len(self._draft_step_times)
                    print(f"[{_ts()}] [metrics] Avg draft step time (ms): {avg_ms:.2f}", flush=True)
                if self._acceptance_rate_log_path and self._acceptance_lengths:
                        import json
                        avg_acc = sum(self._acceptance_lengths) / len(self._acceptance_lengths)
                        hit_rate = sum(self._cache_hits) / len(self._cache_hits) if self._cache_hits else 0
                        print(f"[{_ts()}] [metrics] Avg acceptance length: {avg_acc:.2f} ({len(self._acceptance_lengths)} steps)", flush=True)
                        print(f"[{_ts()}] [metrics] Cache hit rate: {hit_rate:.2%} ({sum(self._cache_hits)}/{len(self._cache_hits)})", flush=True)
                        print(f"[{_ts()}] [metrics] All acceptance lengths: {self._acceptance_lengths}", flush=True)
                        print(f"[{_ts()}] [metrics] All cache hits: {self._cache_hits}", flush=True)
                        print(f"[{_ts()}] [metrics] Logging acceptance lengths and cache hits to: {self._acceptance_rate_log_path}", flush=True)
                        with open(self._acceptance_rate_log_path, "w") as f:
                            json.dump({
                                "acceptance_lengths": self._acceptance_lengths,
                                "cache_hits": self._cache_hits,
                            }, f)
                self.exit()
                break

            else:
                raise RuntimeError(f"draft_loop: unknown command {cmd}")
