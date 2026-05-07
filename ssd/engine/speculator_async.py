import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from ssd.engine.helpers.speculate_types import SpeculateResult, VerifyResult, SpeculatorBase
from ssd.engine.helpers.runner_helpers import PrefillRequest, SpeculationRequest, SpeculationResponse
from ssd.engine.sequence import Sequence
from ssd.utils.misc import decode_tokens


class SpeculatorAsync(SpeculatorBase):

    def __init__(
        self,
        lookahead: int,
        device: torch.device,
        async_fan_out: int,
        max_blocks: int,
        vocab_size: int,
        draft_dtype: torch.dtype,
        kvcache_block_size: int,
        max_model_len: int,
        eagle: bool,
        eagle_act_dim: int,
        communicate_logits: bool,
        communicate_cache_hits: bool,
        async_pg: dist.ProcessGroup,
        draft_runner_rank: int,
        tokenizer: AutoTokenizer,
        verbose: bool,
    ):
        super().__init__(lookahead, device)
        self.async_fan_out = async_fan_out
        self.max_blocks = max_blocks
        self.vocab_size = vocab_size
        self.draft_dtype = draft_dtype
        self.kvcache_block_size = kvcache_block_size
        self.max_model_len = max_model_len
        self.eagle = eagle
        self.eagle_act_dim = eagle_act_dim
        self.communicate_logits = communicate_logits
        self.communicate_cache_hits = communicate_cache_hits
        self.async_pg = async_pg
        self.draft_runner_rank = draft_runner_rank
        self.target_rank = 0
        self.tokenizer = tokenizer
        self.verbose = verbose
        self.K = lookahead

        # Pre-allocate handshake send/recv buffers (reused every step)
        B = 1
        self._speculation_request = SpeculationRequest.prepare(
            batch_size=B,
            lookahead=lookahead,
            max_blocks=max_blocks,
            vocab_size=vocab_size,
            draft_dtype=draft_dtype,
            device=device,
            eagle=eagle,
            eagle_act_dim=eagle_act_dim,
        )
        self._speculation_response = SpeculationResponse.prepare(
            batch_size=B,
            lookahead=lookahead,
            device=device,
            draft_dtype=draft_dtype,
            communicate_logits=communicate_logits,
            communicate_cache_hits=communicate_cache_hits,
            vocab_size=vocab_size,
        )
        self._recovery_buf = torch.empty(B, dtype=torch.int64, device=self.device)
        self._speculations_buf = torch.empty(B, self.K + 1, dtype=torch.int64, device=self.device)

    def _prepare_prefill_request(self, seqs: list[Sequence], verify_result: VerifyResult) -> PrefillRequest:
        eagle_acts = verify_result.eagle_acts
        input_id_list = [seq.token_ids for seq in seqs]

        # EAGLE/Phoenix token-conditioning shift: we duplicate the first target activation for each sequence.
        # [t0, h0], [t1, h0], [t2, h1], [t3, h2], ...
        if eagle_acts is not None:
            sliced = []
            offset = 0
            for ids in input_id_list:
                seq_len = len(ids)
                sliced.append(eagle_acts[offset:offset + 1])
                sliced.append(eagle_acts[offset:offset + seq_len - 1])
                offset += seq_len
            eagle_acts = torch.cat(sliced, dim=0)

        max_blocks = (self.max_model_len + self.kvcache_block_size - 1) // self.kvcache_block_size
        input_ids_flat = []
        num_tokens = []
        for input_ids in input_id_list:
            input_ids_flat.extend(input_ids)
            num_tokens.append(len(input_ids))

        draft_block_tables = [seq.draft_block_table for seq in seqs]
        input_ids_flat = torch.tensor(input_ids_flat, dtype=torch.int64, device=self.device)
        num_tokens = torch.tensor(num_tokens, dtype=torch.int64, device=self.device)
        if isinstance(draft_block_tables, list):
            draft_block_table = torch.tensor(
                [dbt + [-1] * (max_blocks - len(dbt)) for dbt in draft_block_tables],
                dtype=torch.int32, device=self.device,
            )
        else:
            assert draft_block_tables.shape == (len(input_id_list), max_blocks), (
                f"draft_block_tables shape mismatch: expected ({len(input_id_list), max_blocks}), got {draft_block_tables.shape}"
            )
            draft_block_table = draft_block_tables

        return PrefillRequest.prepare(
            input_ids_flat,
            num_tokens,
            draft_block_table,
            eagle_acts,
            max_blocks,
            self.device,
        )

    def prefill(self, seqs: list[Sequence], verify_result: VerifyResult) -> SpeculateResult:
        prefill_request = self._prepare_prefill_request(seqs, verify_result)
        prefill_request.send(self.async_pg, self.draft_runner_rank)
        return SpeculateResult([], [])

    def speculate(self, seqs: list[Sequence], verify_result: VerifyResult) -> SpeculateResult:
        for seq in seqs:
            assert seq.recovery_token_id is not None
            seq.append_token(seq.recovery_token_id)

        if self.verbose:
            sep = '=' * 80
            print(f"\n{sep}", flush=True)
            print(f"[TARGET SEQUENCE TRUNK] Batch size: {len(seqs)}", flush=True)
            for i, seq in enumerate(seqs):
                trunk = seq.token_ids[-20:] if len(seq.token_ids) > 20 else seq.token_ids
                print(f"  Seq {seq.seq_id} (len={len(seq.token_ids)}):", flush=True)
                print(f"    Trunk: ...{decode_tokens(trunk, self.tokenizer)}", flush=True)
                print(f"    Recovery: {seq.recovery_token_id} ({decode_tokens([seq.recovery_token_id], self.tokenizer)})", flush=True)
            print(f"{sep}\n", flush=True)

        eagle = verify_result.eagle_acts is not None
        assert self.eagle == eagle, "Eagle status mismatch"
        speculation_response = self._make_speculation_request(seqs, eagle)
        speculation_tokens = speculation_response.speculations
        logits_q = speculation_response.logits_q
        cache_hits = speculation_response.cache_hits

        # Build speculations using pre-allocated buffers (avoids torch.tensor(device=cuda) sync)
        speculations = self._prepend_recovery_tokens(seqs, speculation_tokens)

        for i, seq in enumerate(seqs):
            seq.token_ids.extend(speculation_tokens[i].tolist())
            seq.num_tokens = len(seq.token_ids)
            seq.last_token = seq.token_ids[-1]
            seq.num_draft_cached_tokens += len(speculation_tokens[i]) + 1

        return SpeculateResult(speculations, logits_q, cache_hits)

    def _prepend_recovery_tokens(self, seqs: list[Sequence], speculation_tokens: torch.Tensor) -> torch.Tensor:
        B = len(seqs)
        if B != self._recovery_buf.shape[0]:
            self._recovery_buf = torch.empty(B, dtype=torch.int64, device=self.device)
            self._speculations_buf = torch.empty(B, self.K + 1, dtype=torch.int64, device=self.device)
        _rec_cpu = torch.tensor([seq.recovery_token_id for seq in seqs], dtype=torch.int64)
        self._recovery_buf.copy_(_rec_cpu, non_blocking=True)
        self._speculations_buf[:, 0] = self._recovery_buf
        self._speculations_buf[:, 1:] = speculation_tokens
        return self._speculations_buf

    def _prepare_speculation_request(self, seqs: list[Sequence], eagle: bool) -> SpeculationRequest:
        B = len(seqs)
        self._speculation_request.maybe_update_buffers(B)

        # Fill send buffers in-place (avoids torch.tensor from Python lists)
        for i, seq in enumerate(seqs):
            self._speculation_request.cache_keys[i, 0] = seq.seq_id
            self._speculation_request.cache_keys[i, 1] = seq.last_spec_step_accepted_len - 1
            self._speculation_request.cache_keys[i, 2] = seq.recovery_token_id
            self._speculation_request.num_tokens[i] = seq.num_tokens
            self._speculation_request.temps[i] = seq.draft_temperature if seq.draft_temperature is not None else seq.temperature
            bt = seq.draft_block_table
            bt_len = len(bt)
            if bt_len > 0:
                self._speculation_request.block_tables[i, :bt_len] = torch.tensor(bt, dtype=torch.int32, device=self.device)
            self._speculation_request.block_tables[i, bt_len:] = -1

        if eagle:
            self._prepare_eagle_payload(seqs)

        return self._speculation_request

    def _prepare_eagle_payload(self, seqs: list[Sequence]):
        # Layout: extend_activations[i, :n] / extend_token_ids[i, :n] hold the n extend entries;
        # extend_activations[i, n] / extend_token_ids[i, n] hold the recovery activation/token,
        # where n = extend_count.
        for i, seq in enumerate(seqs):
            n = seq.extend_count
            self._speculation_request.extend_counts[i] = n
            if n > 0 and seq.extend_eagle_acts is not None:
                self._speculation_request.extend_activations[i, :n] = seq.extend_eagle_acts[:n].to(self.draft_dtype)
                self._speculation_request.extend_token_ids[i, :n] = seq.extend_token_ids[:n]
            self._speculation_request.extend_activations[i, n] = seq.last_target_hidden_state.to(self.draft_dtype)
            self._speculation_request.extend_token_ids[i, n] = seq.recovery_token_id

    def _make_speculation_request(self, seqs: list[Sequence], eagle: bool):
        speculation_request = self._prepare_speculation_request(seqs, eagle)
        speculation_request.send(self.async_pg, self.draft_runner_rank)
        self._speculation_response.receive(self.async_pg, self.draft_runner_rank, batch_size=len(seqs))
        return self._speculation_response
