from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer

from ssd.engine.model_runner import ModelRunner
from ssd.engine.sequence import Sequence
from ssd.engine.scheduler import Scheduler
from ssd.engine.helpers.speculate_types import SpeculatorBase, VerifierBase, VerifyResult
from ssd.utils import profile
from ssd.utils.misc import decode_tokens


class InferenceStep(ABC):

    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler

    @abstractmethod
    def prefill(self, seqs: list[Sequence], step_num: int = 0) -> int:
        pass

    @abstractmethod
    def decode(self, seqs: list[Sequence], step_num: int = 0) -> int:
        pass


class AutoRegressiveStep(InferenceStep):

    def __init__(self, scheduler: Scheduler, model_runner: ModelRunner, tokenizer: AutoTokenizer, verbose: bool = False):
        super().__init__(scheduler)
        self.model_runner = model_runner
        self.tokenizer = tokenizer
        self.verbose = verbose

    def step(self, seqs: list[Sequence], is_prefill: bool, step_num: int = 0) -> int:
        if self.verbose:
            print(f'[auto_regressive_step] is_prefill={is_prefill}', flush=True)

        token_ids = self.model_runner.call("run", seqs, is_prefill)

        if self.verbose:
            decoded_tokens = decode_tokens(token_ids, self.tokenizer)
            print(f"[auto_regressive_step] generated tokens: {decoded_tokens}", flush=True)

        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        return len(seqs) if not is_prefill else sum(len(seq) for seq in seqs)

    def prefill(self, seqs: list[Sequence], step_num: int = 0) -> int:
        return self.step(seqs, is_prefill=True, step_num=step_num)

    def decode(self, seqs: list[Sequence], step_num: int = 0) -> int:
        return self.step(seqs, is_prefill=False, step_num=step_num)


class SpecDecodeStep(InferenceStep):

    def __init__(
        self,
        scheduler: Scheduler,
        speculator: SpeculatorBase,
        verifier: VerifierBase,
        eagle: bool,
        tokenizer: AutoTokenizer,
        async_spec: bool,
        verbose: bool = False,
    ):
        super().__init__(scheduler)
        self.speculator = speculator
        self.verifier = verifier
        self.eagle = eagle
        self.tokenizer = tokenizer
        self.async_spec = async_spec
        self.verbose = verbose

    def prefill(self, seqs: list[Sequence], step_num: int = 0) -> int:
        # When doing async speculation and not Eagle, we can do draft and target prefills in parallel.
        # TEMPORARY: Disable prefill optimization of running draft and target prefills in parallel.
        # if not self.eagle and self.async_spec:
        #     empty_verify_result = VerifyResult([], [], None)
        #     self.speculator.prefill(seqs, empty_verify_result)
        #     verify_result = self.verifier.prefill(seqs, eagle=False)
        # else:
        if self.verbose:
            print(f"[SpecDecodeStep] Verifier prefill {step_num}", flush=True)
        verify_result = self.verifier.prefill(seqs, eagle=self.eagle)

        if self.verbose:
            print(f"[SpecDecodeStep] Speculator prefill {step_num}", flush=True)
        self.speculator.prefill(seqs, verify_result)

        if self.verbose:
            print(f"[SpecDecodeStep] Prefill {step_num} complete", flush=True)

        for seq in seqs:
            assert seq.recovery_token_id is not None
            seq.num_cached_tokens = seq.num_prompt_tokens
            seq.num_draft_cached_tokens = seq.num_prompt_tokens

        return sum(len(seq) for seq in seqs)

    def decode(self, seqs: list[Sequence], step_num: int = 0) -> int:
        ev = profile.new_events(4)
        if ev: ev[0].record()

        # Save lightweight state instead of expensive clone_spec deep copy.
        # speculate() modifies: token_ids (append+extend), num_tokens, last_token, num_draft_cached_tokens
        # verify() modifies: num_cached_tokens (line 77 of verifier.py)
        # postprocess_speculate() needs the ORIGINAL state to apply new suffixes.
        saved = [(len(seq.token_ids), seq.num_tokens, seq.last_token, seq.num_draft_cached_tokens, seq.num_cached_tokens) for seq in seqs]

        eagle_sentinel = True if self.eagle else None
        in_verify_result = VerifyResult(
            new_suffixes=[],
            recovery_tokens=[],
            eagle_acts=eagle_sentinel,
        )
        #### STEP 1: SPECULATE ####
        speculate_result = self.speculator.speculate(seqs, in_verify_result)
        if ev: ev[1].record()

        if self.verbose:
            speculations = speculate_result.speculations
            print(f"[SpecDecodeStep] speculations {step_num}: {speculations}", flush=True)
            speculations_list = speculations.tolist()

            for i, speculation in enumerate(speculations_list):
                decoded_tokens = decode_tokens(speculation, self.tokenizer)
                print(f"[SpecDecodeStep] speculation {step_num},{i}: {decoded_tokens}", flush=True)

        #### STEP 2: VERIFY ####
        out_verify_result = self.verifier.verify(seqs, speculate_result, eagle=self.eagle)
        if ev: ev[2].record()

        if self.verbose:
            recovery_tokens = out_verify_result.recovery_tokens
            new_suffixes = out_verify_result.new_suffixes
            for i, new_suffix in enumerate(new_suffixes):
                decoded_tokens = decode_tokens(new_suffix + [recovery_tokens[i]], self.tokenizer)
                print(f"[SpecDecodeStep] verification {step_num},{i}: {decoded_tokens}", flush=True)

        # Restore original seq state before postprocess (undo speculate + verify modifications)
        for seq, (orig_len, orig_nt, orig_lt, orig_ndc, orig_nct) in zip(seqs, saved):
            del seq.token_ids[orig_len:]
            seq.num_tokens = orig_nt
            seq.last_token = orig_lt
            seq.num_draft_cached_tokens = orig_ndc
            seq.num_cached_tokens = orig_nct

        #### STEP 3: POSTPROCESS ####
        self.scheduler.postprocess_speculate(
            seqs,
            out_verify_result.new_suffixes,
            out_verify_result.recovery_tokens,
            eagle_acts=out_verify_result.eagle_acts if self.eagle else None,
        )

        if ev: ev[3].record()
        cache_hits = speculate_result.cache_hits
        n_hits = int(cache_hits.sum().item()) if cache_hits is not None else 0
        toks = sum(len(s) for s in out_verify_result.new_suffixes)
        profile.emit(
            "SpecDecodeStep.decode",
            ["handshake", "verify", "postprocess"],
            ev,
            step=step_num,
            hits=n_hits,
            toks=toks,
        )
        # Flush at end of step: aggregates events from this step + everything
        # called inside (verifier, CG-path replays, in-model forward/compute_logits).
        profile.flush()

        return sum(len(s) for s in out_verify_result.new_suffixes)
