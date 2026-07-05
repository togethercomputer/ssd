import os
from pathlib import Path

import pytest
import requests
import torch
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer

from ssd import LLM, SamplingParams
from .eagle3_hf import Eagle3Model, load_eagle3_specforge
from .phoenix_hf import PhoenixModel, load_phoenix_specforge
from .helpers import require_8b_target, require_eagle_llama_8b_draft, require_1b_draft, require_phoenix_llama_8b_draft, launch_tgl_server, wait_for_server, kill_server


PORT = 40023
LOGIT_GAP_THRESHOLD = 0.3
# When reconstructing the engine's speculations with the HF reference draft, the
# engine's speculated token must stay near the top of the HF draft's distribution
# (only bf16-level numerical drift should separate them). A rank above this means
# the reconstruction is conditioning on the wrong activation/position — a real bug.
SPEC_RANK_THRESHOLD = 4
# The acceptance length recomputed from the engine's dumped prefixes must match the
# engine's own reported spec_accept_length to within this tolerance.
ACCEPT_LENGTH_TOLERANCE = 0.03
EAGLE_LAYERS = [2, 16, 29]
D_MODEL = 4096
# Phoenix conditions on the target's final post-norm hidden state (the vector
# fed to the LM head). In HF's output_hidden_states tuple (length num_layers+1),
# that is the last entry, index 32 for the 32-layer Llama-3.1-8B target. Index 31
# would be the *input* to the last layer (pre-norm), which is what the engine does
# NOT use.
PHOENIX_LAYERS = [32]

ASYNC_BACKUPS = ["force-jit", "jit", "fast"]
SPECULATOR_TYPES = ["standalone", "eagle", "phoenix"]
CROSS_NODE = [True, False]

# @pytest.mark.parametrize("speculator_type", ["standalone"])
# @pytest.mark.parametrize("cross_node", [False])
# @pytest.mark.parametrize("backup", ["force-jit"])
@pytest.mark.parametrize("backup", ["fast", "jit", "force-jit"])
@pytest.mark.parametrize("speculator_type", ["phoenix", "eagle", "standalone"])
@pytest.mark.parametrize("cross_node", [False])
@pytest.mark.parametrize("engine", ["tgl"])
@pytest.mark.parametrize("max_new_tokens", [128])
def test_ssd_vs_hf_reference(backup, speculator_type, cross_node, engine, max_new_tokens, tmp_path):
    lookahead = 4
    fanout = 3
    engine_accept_length = None  # engine's reported spec_accept_length; set for tgl below
    eagle = speculator_type in ["eagle", "sync_eagle"]
    phoenix = speculator_type in ["phoenix", "sync_phoenix"]
    sync_speculator = speculator_type in ["sync_standalone", "sync_eagle", "sync_phoenix"]
    dtype = torch.bfloat16
    target_path = require_8b_target()
    if eagle:
        draft_path = require_eagle_llama_8b_draft()
    elif phoenix:
        draft_path = require_phoenix_llama_8b_draft()
    else:
        draft_path = require_1b_draft()

    trace_dir = tmp_path / "trace"
    trace_dir.mkdir(exist_ok=True)
    os.environ["SSD_DUMP_TENSORS_DIR"] = str(trace_dir)
    print(f"================================================================================")
    print(f"[{engine}] Launching {engine} engine with speculator type {speculator_type} and backup {backup}, trace directory {trace_dir}, max new tokens {max_new_tokens}, cross node {cross_node}", flush=True)
    print(f"================================================================================")

    tokenizer = AutoTokenizer.from_pretrained(target_path)
    prompt_tokens = tokenizer.apply_chat_template(
        [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Please tell me about San Francisco."}],
        add_generation_prompt=True,
    )
    if isinstance(prompt_tokens, list):
        print(f"[{engine}] BANANA: {prompt_tokens=}", flush=True)
    else:
        prompt_tokens = prompt_tokens["input_ids"]

    # For each engine, we initialize the engine, send a request to it, and then tear down the engine.
    if engine == "tgl":
        tgl_server = None
        draft_process = None
        try:
            tgl_server, draft_process = launch_tgl_server(
                speculator_type, backup, target_path, draft_path, lookahead, fanout, PORT, cross_node=cross_node,
            )

            assert wait_for_server(PORT), "tgl server failed to start"
            print(f"[{engine}] server up; sending request", flush=True)

            resp = requests.post(
                f"http://localhost:{PORT}/generate",
                json={
                    "input_ids": prompt_tokens,
                    "sampling_params": {
                        "temperature": 0.0,
                        "max_new_tokens": max_new_tokens,
                        "ignore_eos": True,
                    },
                },
            )
            # Fields in the response json:
            # 'completion_tokens': 128, 'e2e_latency': 1.4077615810092539,
            # 'spec_accept_rate': 0.8166666666666667, 'spec_accept_length': 4.266666666666667, 'spec_accept_histogram': [4, 0, 2, 2, 22], 
            # 'spec_accept_token_num': 98, 'spec_draft_token_num': 120, 'spec_verify_ct': 30, 

            assert resp.status_code == 200, "tgl server failed to generate"
            print(f"[{engine}] response received", flush=True)
            resp_json = resp.json()
            print(f"[{engine}] response json: {resp_json}", flush=True)
            # completion_text = resp_json["text"]
            completion_tokens = resp_json["output_ids"]
            engine_accept_length = resp_json["meta_info"]["spec_accept_length"]
            print(f"[{engine}] prompt tokens: {prompt_tokens}", flush=True)
            print(f"[{engine}] response tokens: {completion_tokens}", flush=True)

        except Exception as e:
            print(f"[{engine}] error: {e}", flush=True)
            pytest.fail(f"[{engine}] error: {e}")

        finally:
            # Guard for launch_tgl_server itself raising: tgl_server stays None and
            # there is nothing to kill — without the guard the UnboundLocalError here
            # masks the primary launch error in the pytest report.
            if tgl_server is not None:
                print(f"[{engine}] killing server", flush=True)
                kill_server(tgl_server)
                assert not wait_for_server(PORT, timeout=3.0), "tgl server failed to stop"
                print(f"[{engine}] server stopped", flush=True)

            if cross_node and draft_process is not None:
                print(f"[{engine}] killing draft process", flush=True)
                kill_server(draft_process)
                print(f"[{engine}] draft process stopped", flush=True)

    elif engine == "ssd":
        if eagle:
            eagle_layers = EAGLE_LAYERS
        elif phoenix:
            eagle_layers = PHOENIX_LAYERS
        else:
            eagle_layers = None
        ssd_kwargs = dict(
            enforce_eager=False,
            num_gpus=2,
            speculate=True,
            speculate_k=lookahead,
            draft_async=True,
            async_fan_out=fanout,
            verbose=True,
            draft=draft_path,
            kvcache_block_size=64,
            max_num_seqs=1,
            max_model_len=4096,
            jit_speculate=(backup == "jit" or backup == "force-jit"),
            force_jit_speculate=(backup == "force-jit"),
            communicate_cache_hits=True,
            communicate_logits=True,
            use_eagle=eagle,
            use_phoenix=phoenix,
            eagle_layers=eagle_layers,
        )
        llm = None
        try:
            llm = LLM(target_path, **ssd_kwargs)
            print(f"[{engine}] generating completion", flush=True)
            output, metrics = llm.generate(
                [prompt_tokens],
                SamplingParams(max_new_tokens=max_new_tokens, temperature=0.0, ignore_eos=True),
                use_tqdm=False,
            )
        except Exception as e:
            print(f"[{engine}] error: {e}", flush=True)
            pytest.fail(f"[{engine}] error: {e}")
        finally:
            # Clean up the engine.
            if llm is not None:
                llm.exit(hard=False)
                del llm
            # Defensive: if LLM init raised partway, llm is None and exit() never ran,
            # so the default process group set up inside ModelRunner.__init__ is still
            # alive in this process. Without this, the next parametrize case fails with
            # "trying to initialize the default process group twice".
            try:
                if torch.distributed.is_initialized():
                    torch.distributed.destroy_process_group()
            except Exception:
                pass
            import gc; gc.collect()
            torch.cuda.empty_cache()

        completion_text = output[0]["text"]
        print(f"[{engine}] completion text: {completion_text}", flush=True)
        completion_tokens = output[0]["token_ids"]
        print(f"[{engine}] completion tokens: {completion_tokens}", flush=True)
        print(f"[{engine}] generation metrics: {metrics}", flush=True)
    else:
        raise ValueError(f"Unknown engine: {engine}")

    # COMPARE TGL RESPONSE TO HF REFERENCE. Ensure that 
    target_device = "cuda:4"
    draft_device = "cuda:5"

    # Load target
    print(f"[{engine}] begin load target model", flush=True)
    target_model = AutoModelForCausalLM.from_pretrained(target_path, torch_dtype=dtype)
    print(f"[{engine}] target model loaded", flush=True)
    target_model.eval()
    target_model.to(target_device)

    # COMPARE TGL RESPONSE TO HF REFERENCE.
    print(f"====================================================")
    print(f"[{engine}] Beginning comparison of completion to hf reference ({speculator_type}, {backup})")
    print(f"=====================================================")
    gaps, full_target_logits = compare_completion_to_hf_reference(
        target_model,
        prompt_tokens,
        completion_tokens,
        0,
        tokenizer,
        engine=engine,
    )
    assert max(gaps) < LOGIT_GAP_THRESHOLD, f"COMPARE COMPLETION TO HF REFERENCE: max gap {max(gaps)} exceeds threshold {LOGIT_GAP_THRESHOLD}, {gaps=}"

    if sync_speculator:
        return

    # Load draft
    if eagle:
        draft_model = load_eagle3_specforge(
            draft_path, target_model.model.embed_tokens.weight, target_model.config.hidden_size, draft_device,
            dtype=dtype,
        )
        draft_model.eval()
    elif phoenix:
        draft_model = load_phoenix_specforge(
            draft_path, target_model.config.hidden_size, draft_device,
            dtype=dtype,
        )
        draft_model.eval()
    else:
        assert speculator_type == "standalone"
        draft_model = AutoModelForCausalLM.from_pretrained(draft_path, torch_dtype=dtype).to(draft_device)
        draft_model.eval()


    print(f"====================================================")
    print(f"[{engine}] Beginning SSD simulation ({speculator_type}, {backup})")
    print(f"=====================================================")
    full_ssd_simulation(
        target_model,
        draft_model,
        prompt_tokens,
        completion_tokens,
        backup=backup,
        eagle=eagle,
        phoenix=phoenix,
        lookahead=lookahead,
        tokenizer=tokenizer,
    )
    # COMPARE SPECULATIONS TO HF REFERENCE
    print(f"====================================================")
    print(f"[{engine}] Beginning comparison of speculations to hf reference ({speculator_type}, {backup})")
    print(f"=====================================================")
    spec_ranks, recon_accept_length = compare_speculations_to_hf_reference(
        trace_dir,
        target_model,
        draft_model,
        prompt_tokens,
        completion_tokens,
        eagle=eagle,
        phoenix=phoenix,
        backup=backup,
        tokenizer=tokenizer,
        engine=engine,
        full_target_logits=full_target_logits,
    )

    # The engine's speculated tokens must stay near the top of the HF draft's distribution.
    # A rank above SPEC_RANK_THRESHOLD means the reconstruction is conditioning on the wrong
    # activation/position rather than mere bf16 drift. (eagle/phoenix only; standalone has none.)
    if spec_ranks:
        worst_rank = max(max(r) for r in spec_ranks)
        assert worst_rank <= SPEC_RANK_THRESHOLD, (
            f"COMPARE SPECULATIONS TO HF REFERENCE: worst speculated-token rank {worst_rank} "
            f"exceeds threshold {SPEC_RANK_THRESHOLD}, {spec_ranks=}"
        )

    # The acceptance length recomputed from the engine's dumped prefixes must match the
    # engine's own reported spec_accept_length.
    if engine_accept_length is not None:
        accept_length_delta = abs(recon_accept_length - engine_accept_length)
        assert accept_length_delta <= ACCEPT_LENGTH_TOLERANCE, (
            f"COMPARE SPECULATIONS TO HF REFERENCE: reconstructed acceptance length "
            f"{recon_accept_length:.4f} differs from engine spec_accept_length "
            f"{engine_accept_length:.4f} by {accept_length_delta:.4f} > {ACCEPT_LENGTH_TOLERANCE}"
        )


def compare_completion_to_hf_reference(
    model,
    prefix: list[int],
    completion: list[int],
    request_index: int,
    tokenizer: AutoTokenizer,
    engine: str = "tgl",
    full_target_logits: torch.Tensor = None,
    verbose: bool = False,
):
    completion_length = len(completion)
    all_tokens = prefix + completion
    hf_logits_for_completion = get_hf_logits_for_completion(model, all_tokens, completion_length)
    gaps = []
    for i in range(completion_length):
        completion_token = completion[i]
        hf_logit = hf_logits_for_completion[i, completion_token]
        hf_max_logit = hf_logits_for_completion[i].max()
        gaps.append(torch.abs(hf_logit - hf_max_logit).item())

    max_gap = max(gaps)

    greedy_preds = hf_logits_for_completion.argmax(dim=-1)
    matching = tokenizer.decode(greedy_preds) == tokenizer.decode(completion)
    match_str = "YES" if matching else " NO"
    if verbose:
        print("=============")
        print(f"[{engine}][{request_index}][{match_str}] completion (hf reference): {tokenizer.decode(greedy_preds)}")
        print(f"[{engine}][{request_index}][{match_str}] completion (engine - tgl): {tokenizer.decode(completion)}")
    print(f"[{engine}][{request_index}][{match_str}] max gap: {max_gap}, gaps: {gaps}")

    if full_target_logits is not None:
        full_target_logits = full_target_logits.to(hf_logits_for_completion.device)
        norm_gaps = []
        for i in range(completion_length):
            idx = len(prefix) + i
            curr_logits = hf_logits_for_completion[i]
            if idx > full_target_logits.shape[0] - 1:
                break
            target_logits = full_target_logits[idx]
            target_probs = torch.softmax(target_logits, dim=-1)
            curr_probs = torch.softmax(curr_logits, dim=-1)
            norm_gaps.append(torch.linalg.norm(curr_probs - target_probs, ord=1).item())
        max_norm_gap = max(norm_gaps) if norm_gaps else 0.0
        print(f"[{engine}][{request_index}] max norm gap: {max_norm_gap}, norm gaps: {norm_gaps}")

    # pytest.set_trace()
    return gaps, hf_logits_for_completion


def full_ssd_simulation(
    target_model: AutoModelForCausalLM,
    draft_model: AutoModelForCausalLM | Eagle3Model | PhoenixModel,
    prompt_tokens: list[int],
    completion_tokens: list[int],
    backup: str = "force-jit",
    eagle: bool = False,
    phoenix: bool = False,
    lookahead: int = 4,
    full_target_logits: torch.Tensor = None,
    full_target_activations: torch.Tensor = None, # Note: These should already be projected into the draft space.
    duplicate_first_token: bool = True,
    tokenizer: AutoTokenizer = None,
    fan_out: int = 5,
    verbose: bool = False,
):
    all_tokens = prompt_tokens + completion_tokens
    all_tokens_tensor = torch.tensor([all_tokens], device=draft_model.device, dtype=torch.long)
    draft_device = draft_model.device
    dtype = draft_model.lm_head.weight.dtype
    if full_target_activations is None and (eagle or phoenix):
        full_target_activations = get_hf_target_activations_for_eagle_or_phoenix(target_model, all_tokens, eagle, phoenix).to(draft_model.device)
        if duplicate_first_token:
            full_target_activations = torch.cat([
                full_target_activations[:1],
                full_target_activations
            ])
            if eagle:
                full_target_activations = draft_model.fc(full_target_activations.to(dtype=dtype))
        else:
            raise ValueError("Unsupported at the moment")

    if full_target_logits is None:
        full_target_logits = get_hf_logits(target_model, all_tokens).to(draft_model.device)

    target_preds = full_target_logits.argmax(dim=-1)
    
    acceptance_lengths = []
    cache_hits = []
    probability_gaps = []

    cache_hit = False
    generated = 1  # bonus token from prefill is already generated
    while True:
        ## SPECULATE ##
        tokens_remaining = all_tokens_tensor.shape[1] - (len(prompt_tokens) + generated)
        # Need lookahead tokens for the speculation/gap loop AND one extra token
        # so the cache-hit check (which indexes at num_accepted, up to lookahead) is in range.
        if tokens_remaining < lookahead + 1:
            break

        if eagle or phoenix:
            if backup == "force-jit" or (not cache_hit and backup == "jit") or cache_hit:

                # For cache hits, we don't have the target activations from the previous round.
                if cache_hit and backup != "force-jit":
                    num_generated_last_round = acceptance_lengths[-1] + 1
                    base_len = len(prompt_tokens) + generated - num_generated_last_round
                    # We do one extra draft pass (+1) to get the logits after the last speculated token,
                    # which are needed to check for cache hits when all tokens are accepted.
                    num_draft_passes = num_generated_last_round + lookahead + 1
                else:
                    base_len = len(prompt_tokens) + generated
                    # We do one extra draft pass (+1) to get the logits after the last speculated token,
                    # which are needed to check for cache hits when all tokens are accepted.
                    num_draft_passes = lookahead + 1
                current_activations = full_target_activations[:base_len]
                for i in range(num_draft_passes):
                    curr_len = base_len + i
                    current_prefix = all_tokens_tensor[0, :curr_len]
                    # print(f"[SIMULATION] current_activations.shape: {current_activations.shape}")
                    if i > 0:
                        # print(f"[SIMULATION] draft_activations.shape: {draft_activations.shape}")
                        if phoenix:
                            current_activations = torch.cat([current_activations, full_target_activations[base_len - 1:base_len]])
                        else:
                            current_activations = torch.cat([current_activations, draft_activations[-1:]])
                    draft_activations = draft_model.forward_with_cond(current_prefix, torch.arange(curr_len, device=draft_device), current_activations)
                speculation_activations = draft_model.norm(draft_activations[-(lookahead + 1):])
                speculation_logits = draft_model.lm_head(speculation_activations)
                if eagle:
                    speculation_logits = convert_to_full_vocab_logits(draft_model, speculation_logits)
                speculation_preds = speculation_logits.argmax(dim=-1)
            else:
                # Fast speculation on cache miss: the engine sends zeros as the K speculation
                # tokens, but still runs a glue decode over the prefix so the recovery-position
                # logits are available for the next round's cache-candidate lookup. We mirror
                # that here so the cache_hit check below has real next-round logits.
                curr_len = len(prompt_tokens) + generated
                current_prefix = all_tokens_tensor[0, :curr_len]
                current_activations = full_target_activations[:curr_len]
                draft_activations = draft_model.forward_with_cond(
                    current_prefix,
                    torch.arange(curr_len, device=draft_device),
                    current_activations,
                )
                recovery_logits = draft_model.lm_head(draft_model.norm(draft_activations[-1:]))
                if eagle:
                    recovery_logits = convert_to_full_vocab_logits(draft_model, recovery_logits)

                speculation_logits = torch.full((lookahead + 1, draft_model.config.vocab_size), float("-inf"), device=draft_device, dtype=dtype)
                speculation_logits[:, 0] = 0.0
                speculation_logits[0] = recovery_logits[0]
                speculation_preds = torch.zeros(lookahead + 1, device=draft_device, dtype=torch.long)
        else:
            curr_len = len(prompt_tokens) + generated + lookahead
            current_prefix = all_tokens_tensor[:, :curr_len]
            if backup == "fast" and not cache_hit:
                # Fast speculation on cache miss: the engine sends zeros as the K speculation
                # tokens, but still runs a glue decode over the prefix so the recovery-position
                # logits are available for the next round's cache-candidate lookup. We mirror
                # that here so the cache_hit check below has real next-round logits.
                recovery_prefix_len = len(prompt_tokens) + generated
                recovery_prefix = all_tokens_tensor[:, :recovery_prefix_len]
                recovery_logits = draft_model.forward(recovery_prefix).logits[0, -1:]  # [1, vocab]

                speculation_logits = torch.full((lookahead + 1, draft_model.config.vocab_size), float("-inf"), device=draft_device, dtype=dtype)
                speculation_logits[:, 0] = 0.0
                speculation_logits[0] = recovery_logits[0]
                speculation_preds = torch.zeros(lookahead + 1, device=draft_device, dtype=torch.long)
            else:
                speculation_logits = draft_model.forward(current_prefix).logits[0]
                speculation_logits = speculation_logits[-(lookahead + 1):]
                # Note: speculation preds has an extra token at the end.
                speculation_preds = speculation_logits.argmax(dim=-1)
        ### END SPECULATE ###

        ### CHECK HOW MANY TOKENS ARE ACCEPTED ###
        num_accepted = lookahead
        for i in range(lookahead):
            curr_idx = len(prompt_tokens) + generated + i
            next_token = all_tokens[curr_idx]
            if verbose and target_preds[curr_idx - 1].item() != next_token:
                if tokenizer is not None:
                    target_pred_str = tokenizer.decode(target_preds[curr_idx - 1])
                    next_token_str = tokenizer.decode(next_token)
                    print(f"[SIMULATION] Target prediction `{target_pred_str}` != next token `{next_token_str}` at index {curr_idx}")
                else:
                    print(f"[SIMULATION] Target prediction {target_preds[curr_idx].item()} != next token {next_token} at index {curr_idx}")

            speculated_token = speculation_preds[i].item()
            if speculated_token != next_token:
                num_accepted = i
                break

        acceptance_lengths.append(num_accepted)
        ### END CHECK HOW MANY TOKENS ARE ACCEPTED ###

        ### DETERMINE IF THERE IS A CACHE HIT IN THE NEXT ROUND ###
        next_token = all_tokens[len(prompt_tokens) + generated + num_accepted]
        speculated_token = speculation_preds[num_accepted].item()
        draft_logits = speculation_logits[num_accepted].clone()
        if num_accepted != lookahead:
            draft_logits[speculated_token] = float("-inf")
        cache_hit = int(next_token in draft_logits.topk(k=fan_out).indices)
        cache_hits.append(cache_hit)
        ### END DETERMINE IF THERE IS A CACHE HIT IN THE NEXT ROUND ###

        ### MEASURE PROBABILITY DISTRIBUTION GAPS (DRAFT VS TARGET) ###
        curr_probability_gaps = []
        for i in range(lookahead):
            curr_idx = len(prompt_tokens) + generated + i
            draft_logits = speculation_logits[i]
            target_logits = full_target_logits[curr_idx - 1]
            draft_probs = torch.softmax(draft_logits, dim=-1)
            target_probs = torch.softmax(target_logits, dim=-1)
            gap = torch.linalg.norm(draft_probs - target_probs, ord=1).item()
            if verbose and gap > 0.5:
                prefix = all_tokens_tensor[0, :curr_idx]
                decoded_prefix = tokenizer.decode(prefix)
                print(f"[SIMULATION][{curr_idx}] Prefix: {decoded_prefix}")
                draft_pred = draft_logits.argmax(dim=-1)
                target_pred = target_logits.argmax(dim=-1)
                draft_pred_str = tokenizer.decode(draft_pred)
                target_pred_str = tokenizer.decode(target_pred)
                print(f"[SIMULATION][{curr_idx}] |draft_probs - target_probs| = {gap:.4f}, Draft prediction `{draft_pred_str}`. Target prediction `{target_pred_str}`.")
            curr_probability_gaps.append(gap)

        probability_gaps.append(curr_probability_gaps)
        ### END MEASURE PROBABILITY DISTRIBUTION GAPS (DRAFT VS TARGET) ###

        generated += num_accepted + 1

    acc_lengths_array = np.array(acceptance_lengths) + 1
    print(f"[SIMULATION] Acceptance lengths: {acc_lengths_array.tolist()}")
    print(f"[SIMULATION] Average acceptance length: {acc_lengths_array.mean():.4f}")
    print(f"[SIMULATION] Probability gaps: {probability_gaps}")
    print(f"[SIMULATION] Average probability gap: {np.array(probability_gaps).mean():.4f}")
    if backup != "force-jit":
        print(f"[SIMULATION] Cache hits: {cache_hits}")
        print(f"[SIMULATION] Average cache hit: {np.array(cache_hits).mean():.4f}")
    return acceptance_lengths, probability_gaps


def convert_to_full_vocab_logits(draft_model: Eagle3Model, draft_logits: torch.Tensor) -> torch.Tensor:
    full_vocab_indices = torch.arange(draft_model.d2t.shape[0], device=draft_logits.device) + draft_model.d2t
    full_vocab_logits = draft_logits.new_full((draft_logits.shape[0], draft_model.config.vocab_size), float("-inf"))
    full_vocab_logits.index_copy_(-1, full_vocab_indices, draft_logits)
    return full_vocab_logits


def compare_completion_to_hf_reference_eagle(
    draft_model: Eagle3Model | PhoenixModel,
    prefix: list[int],
    speculation: list[int],
    eagle_acts: torch.Tensor,
    eagle_activation_index: int,  # where to start forward passes from.
    request_index: int,
    extend_token_ids: list[torch.Tensor],
    extend_counts: list[int],
    extend_activations: list[torch.Tensor],
    prompt_eagle_acts: torch.Tensor,
    jit: bool,
    engine_acts: torch.Tensor,
    tokenizer: AutoTokenizer,
    engine: str = "tgl",
    funky: bool = False,
    prefixes: list[list[int]] = None,
    full_target_logits: torch.Tensor = None,
    phoenix: bool = False,
    verbose: bool = False,
):
    """Reconstruct the draft's speculation logits the way the engine produced them and
    compare argmax to the engine's dumped speculation tokens.

    Eagle and Phoenix differ in how the conditioning stream is built:
      - Eagle projects the target acts through `fc` and fills the speculated positions
        via its self-recurrence (each speculated position conditions on the draft's own
        previous prenorm), so it needs the two iterative forward-pass loops below.
      - Phoenix has no `fc` and no recurrence: it conditions on the raw target acts up to
        the recovery point, then reuses the single recovery activation for every
        speculated position. That's a fixed conditioning stream, so one causal forward
        pass over prefix+speculation suffices.
    """
    if funky and jit:
        if request_index == 0:
            eagle_activation_index = len(prefixes[0])
        else:
            eagle_activation_index = len(prefixes[request_index - 1])

    device = draft_model.device
    dtype = draft_model.lm_head.weight.dtype
    all_tokens = torch.tensor(prefix + speculation, device=device, dtype=torch.long)
    eagle_acts = eagle_acts.to(device=device, dtype=dtype)
    # eagle_acts = engine_acts.to(device=device, dtype=dtype)  # WE ARE TESTING OUT ENGINE ACTS INSTEAD OF HF ACTS
    speculation_length = len(speculation)

    if not phoenix:
        all_eagle_acts_proj = draft_model.fc(eagle_acts)
        target_eagle_acts = eagle_acts[:eagle_activation_index]
        target_eagle_acts = draft_model.fc(target_eagle_acts)

        draft_eagle_acts = torch.zeros(all_tokens.shape[0] - eagle_activation_index, target_eagle_acts.shape[1], device=device, dtype=dtype)
        joint_eagle_acts = torch.cat([target_eagle_acts, draft_eagle_acts], dim=0)
        joint_eagle_acts[:eagle_activation_index] = target_eagle_acts
        # First we do len(prefix) - eagle_activation_index steps of forward passes to catch up to the current speculation.
        for i in range(len(prefix) - eagle_activation_index):
            idx = eagle_activation_index + i
            with torch.no_grad():
                if funky and idx == len(prefix) - 1:
                    joint_eagle_acts[idx] = all_eagle_acts_proj[idx]
                else:
                    # teacher-force with the actual speculation tokens.
                    prenorm = draft_model.forward_with_cond(all_tokens[:idx], torch.arange(idx, device=device), joint_eagle_acts[:idx])
                    joint_eagle_acts[idx] = prenorm[-1]

        # Now we do the remaining steps of forward passes to get the logits for the speculation.
        for i in range(speculation_length):
            idx = len(prefix) + i
            with torch.no_grad():
                prenorm = draft_model.forward_with_cond(all_tokens[:idx], torch.arange(idx, device=device), joint_eagle_acts[:idx])
                joint_eagle_acts[idx] = prenorm[-1]

        # joint_eagle_acts[idx] holds the prenorm that predicts token idx, so the last
        # speculation_length entries predict speculation[0..S-1].
        post_norm_final_draft_acts = draft_model.norm(joint_eagle_acts[-speculation_length:])
        draft_logits = draft_model.lm_head(post_norm_final_draft_acts)

        # Scatter draft-vocab draft_logits into target-vocab space via d2t so argmax /
        # indexing by the engine's target-vocab ids is well-defined. Non-draft
        # positions stay -inf (the draft cannot produce those tokens).
        draft_logits = convert_to_full_vocab_logits(draft_model, draft_logits)
    else:
        # Phoenix: raw target acts up to the recovery point, recovery activation reused
        # afterwards, single causal forward pass over prefix+speculation.
        #
        # Build the conditioning at full (prefix+speculation) length. Positions
        # [0, activation_index) use the true target acts; everything after reuses the
        # recovery activation (the target hidden at the last real position). We must NOT
        # index eagle_acts past what's available: the final speculation extends a token or
        # two beyond the dumped completion (it speculates K past the last accepted token),
        # and those trailing positions live entirely in the recovery-reuse region anyway.
        recovery_act = eagle_acts[eagle_activation_index - 1]
        joint_eagle_acts = recovery_act.unsqueeze(0).repeat(all_tokens.shape[0], 1)
        joint_eagle_acts[:eagle_activation_index] = eagle_acts[:eagle_activation_index]
        with torch.no_grad():
            prenorm = draft_model.forward_with_cond(
                all_tokens, torch.arange(all_tokens.shape[0], device=device), joint_eagle_acts,
            )
        # prenorm[j] is the draft output at position j, which predicts token j+1, so the
        # output at position (len(prefix)-1+i) predicts speculation[i].
        start = len(prefix) - 1
        post_norm_final_draft_acts = draft_model.norm(prenorm[start : start + speculation_length])
        draft_logits = draft_model.lm_head(post_norm_final_draft_acts)

    greedy_preds = draft_logits.argmax(dim=-1)

    gaps = []
    per_token_match = []
    ranks = []
    for i in range(speculation_length):
        speculation_token = speculation[i]
        hf_logit = draft_logits[i, speculation_token]
        hf_max_logit = draft_logits[i].max()
        gaps.append(torch.abs(hf_logit - hf_max_logit).item())
        per_token_match.append(int(greedy_preds[i].item() == speculation_token))
        # Rank of the engine's speculated token within the HF draft's logits: number of
        # tokens with a strictly larger logit. 0 means it's (tied for) the HF argmax.
        # Numerical noise => near-top ranks; a real conditioning/indexing bug => the
        # engine's token lands at a much worse rank.
        ranks.append(int((draft_logits[i] > hf_logit).sum().item()))

    matching = tokenizer.decode(greedy_preds) == tokenizer.decode(speculation)
    match_str = "YES" if matching else " NO"
    if verbose:
        prefix_str = tokenizer.decode(prefix)
        print(f"[{engine}][{request_index}] prefix[-40:]: {prefix_str[-40:]}")
        print(f"[{engine}][{request_index}][{match_str}] speculation (hf reference): {tokenizer.decode(greedy_preds).replace('\n', '\\n')}")
        print(f"[{engine}][{request_index}][{match_str}] speculation (engine - tgl): {tokenizer.decode(speculation).replace('\n', '\\n')}")
    print(f"[{engine}][{request_index}][{match_str}] per-token match: {per_token_match}, ranks: {ranks}, max gap: {max(gaps):.4f}, gaps: {gaps}, engine: '{tokenizer.decode(speculation).replace('\n', '\\n')}', hf: '{tokenizer.decode(greedy_preds).replace('\n', '\\n')}'")
    return gaps, ranks


def validate_request_and_response(request, response, request_num, eagle: bool = False, phoenix: bool = False):
    assert request["cache_keys"].shape[0] == 1
    assert request["num_tokens"].shape[0] == 1
    cache_keys = request["cache_keys"][0]
    num_accepted = cache_keys[1].item()
    if request_num == 0:
        assert num_accepted <= 0
    else:
        assert num_accepted >= 0

    if eagle or phoenix:
        assert request["extend_token_ids"].shape[0] == 1
        assert request["extend_counts"].shape[0] == 1
        assert request["extend_activations"].shape[0] == 1

    assert response["cache_hits"].shape[0] == 1
    assert response["logits"].shape[0] == 1


def compare_speculations_to_hf_reference(
    trace_dir: Path,
    target_model,
    draft_model,
    prompt_tokens: list[int],
    completion_tokens: list[int],
    eagle: bool = False,
    phoenix: bool = False,
    backup: str = "force-jit",
    tokenizer: AutoTokenizer = None,
    engine: str = "tgl",
    full_target_logits: torch.Tensor = None,
    verbose: bool = False,
):
    all_tokens = prompt_tokens + completion_tokens
    prefill_request_files = list(trace_dir.glob("prefill_request_*.pt"))
    speculation_request_files = list(sorted(trace_dir.glob("speculation_request_*.pt")))
    speculation_response_files = list(sorted(trace_dir.glob("speculation_response_*.pt")))
    assert len(prefill_request_files) == 1
    assert len(speculation_request_files) == len(speculation_response_files)

    prefill_request = torch.load(prefill_request_files[0])
    speculation_requests = [torch.load(f) for f in speculation_request_files]
    speculation_responses = [torch.load(f) for f in speculation_response_files]

    if not eagle and not phoenix:
        prompt_tokens_from_prefill_request = prefill_request["input_ids"].tolist()
        assert prompt_tokens_from_prefill_request == prompt_tokens, f"{prompt_tokens_from_prefill_request=} != {prompt_tokens=}"
    else:
        hf_full_eagle_acts = get_hf_target_activations_for_eagle_or_phoenix(target_model, all_tokens, eagle, phoenix).to(draft_model.device)
        hf_full_eagle_acts = torch.cat([
            hf_full_eagle_acts[:1],
            hf_full_eagle_acts
        ])
        prompt_eagle_acts = prefill_request["eagle_acts"].to(draft_model.device)
        prompt_len = prompt_eagle_acts.shape[0]
        if verbose:
            print(f"[{engine}] hf prompt acts vs dumped eagle_acts: {torch.norm(prompt_eagle_acts - hf_full_eagle_acts[:prompt_len])}")
            print(f"[{engine}] prompt acts: {prompt_eagle_acts[:5, :5]}")
            print(f"[{engine}] full acts: {hf_full_eagle_acts[:5, :5]}")
            # print(f"[{engine}] prompt eagle acts.shape: {prompt_eagle_acts.shape}")
            # print(f"[{engine}] full eagle acts.shape: {full_eagle_acts.shape}")

    prefixes = []
    speculations = []
    num_accepted = []
    num_tokens = []
    cache_hits = []
    logits = []
    if eagle or phoenix:
        extend_token_ids = []
        extend_counts = []
        extend_activations = []
        extend_activations_accepted = []
    # TODO: Do this per request, by having a dictionary indexed by sequence ID.
    for i in range(len(speculation_requests)):
        request = speculation_requests[i]
        response = speculation_responses[i]
        validate_request_and_response(request, response, i, eagle, phoenix)

        cache_keys = request["cache_keys"][0]
        num_tokens.append(request["num_tokens"][0].item())
        num_accepted.append(cache_keys[1].item())
        rec_token = cache_keys[2].item()
        if i == 0:
            prefixes.append(prompt_tokens + [rec_token])
        else:
            # Does the speculation contain the recovery token? I think it does?
            prefixes.append(prefixes[-1] + speculations[-1][:num_accepted[-1]] + [rec_token])

        if eagle or phoenix:
            extend_token_ids.append(request["extend_token_ids"][0])
            extend_counts.append(request["extend_counts"][0].item())
            extend_activations.append(request["extend_activations"][0])
            if verbose:
                print(f"[{engine}] extend_activations.shape: {extend_activations[-1].shape}")

        # TODO: It seems speculations is shape [lookahead] instead of [batch_size, lookahead]. Fix this?
        speculations.append(response["speculations"].tolist())
        cache_hits.append(response["cache_hits"][0].item())
        logits.append(response["logits"][0].tolist())
        if verbose:
            if tokenizer is not None:
                prefix_text = tokenizer.decode(prefixes[-1])
                speculations_text = tokenizer.decode(speculations[-1])
                print(f"[{engine}] prefix text: {prefix_text}")
                print(f"[{engine}] speculations text: {speculations_text}")
                print(f"[{engine}] num accepted: {num_accepted[-1]}")
                # print(f"[{engine}] num tokens: {num_tokens[-1]}")
                print(f"[{engine}] rec token: {tokenizer.decode([rec_token])}")
            else:
                print(f"[{engine}] prefix: {prefixes[-1]}, speculation: {speculations[-1]}, num_accepted: {num_accepted[-1]}, num_tokens: {num_tokens[-1]}, rec_token: {rec_token}")

    prompt_len = len(prompt_tokens)
    engine_acts = None  # reconstructed below for eagle/phoenix from the dumped extend acts
    if eagle or phoenix:
        # act dim is len(EAGLE_LAYERS)*hidden for eagle, len(PHOENIX_LAYERS)*hidden for phoenix.
        act_dim = hf_full_eagle_acts.shape[-1]
        engine_acts = torch.zeros((len(all_tokens), act_dim), dtype=draft_model.lm_head.weight.dtype, device="cpu")
        engine_acts[:prompt_len] = prompt_eagle_acts.cpu()
        t = prompt_len
        for i in range(len(speculation_requests)):
            num_accept = extend_counts[i]
            if num_accept > 0:
                engine_acts[t: t + num_accept + 1] = extend_activations[i][:num_accept + 1].cpu()
            t += 1 + num_accept
        if verbose:
            print(f"FINAL OFFSET: {t}")
            diffs = [
                (torch.norm(hf_full_eagle_acts[i].cpu() - engine_acts[i]) / torch.norm(hf_full_eagle_acts[i].cpu())).item()
                for i in range(t)
            ]
            for i, diff in enumerate(diffs):
                print(f"DIFF {i}: {diff:.4f}")

            print(f"[{engine}] extend counts: {extend_counts}")

    # pytest.set_trace()
    all_gaps = []
    all_ranks = []  # per-token rank of the engine's speculated token in the HF draft (eagle/phoenix only)
    if verbose:
        print(f"[{engine}] prefix lengths: [{[len(p) for p in prefixes]}")
    
    for i in range(len(speculation_requests)):
        # print(f"BANANA: CHECKING SPECULATION {i}, {cache_hits[i]=}, {backup=}")
        prefix = prefixes[i]
        speculation = speculations[i]
        # num_accepted_i = num_accepted[i]


        # jit: force_jit or (cache_miss and jit)
        # random: fast and cache_miss
        # delayed: cache_hit and not force_jit
        if backup == "fast" and not cache_hits[i]:
            continue

        if not eagle and not phoenix:
            gaps, _ = compare_completion_to_hf_reference(
                draft_model,
                prefix,
                speculation,
                i,
                tokenizer,
                engine=engine,
                full_target_logits=full_target_logits,
                verbose=verbose,
            )
            all_gaps.append(gaps)
        else:
            cache_hit = bool(cache_hits[i])
            jit = backup == "force-jit" or (not cache_hit and backup == "jit")
            if jit:
                eagle_activation_index = len(prefix)
            else:
                assert cache_hit and i > 0
                eagle_activation_index = len(prefixes[i-1])
            # if i > 0:
                # assert len(prefixes[i-1]) + extend_counts[i-1] == len(prefix)

            gaps, ranks = compare_completion_to_hf_reference_eagle(
                draft_model,
                prefix,
                speculation,
                hf_full_eagle_acts,
                eagle_activation_index,
                i,
                extend_token_ids,
                extend_counts,
                extend_activations,
                prompt_eagle_acts,
                jit,
                engine_acts,
                tokenizer,
                engine=engine,
                funky=False,
                prefixes=prefixes,
                full_target_logits=full_target_logits,
                phoenix=phoenix,
                verbose=verbose,
            )
            all_gaps.append(gaps)
            all_ranks.append(ranks)

    method_str = "eagle" if eagle else ("phoenix" if phoenix else "standalone")
    print(f" ****** SUMMARY OF ALL RESULTS (engine={engine}, method={method_str}, backup={backup}) ******")

    if eagle:
        # extend counts don't include the recovery token, so we add 1 to the average.
        avg_acceptance_length = 1 + (sum(extend_counts) / (len(extend_counts) - 1))
        print(f"[{engine},{method_str},{backup}][FINAL_METRIC] Average acceptance lengths: {avg_acceptance_length:.4f}")
        print(f"[{engine},{method_str},{backup}] Full list of acceptance lengths: {[e + 1 for e in extend_counts[1:]]}")
    else:
        prefix_lengths = np.array([len(p) for p in prefixes])
        acceptance_lengths = prefix_lengths[1:] - prefix_lengths[:-1]
        avg_acceptance_length = sum(acceptance_lengths) / len(acceptance_lengths)
        print(f"[{engine},{method_str},{backup}][FINAL_METRIC] Average acceptance lengths: {avg_acceptance_length:.4f}")
        print(f"[{engine},{method_str},{backup}] Full list of acceptance lengths: {acceptance_lengths}")

    print(f"[{engine},{method_str},{backup}][FINAL_METRIC] Average cache hit rate: {sum(cache_hits) / len(cache_hits)}")
    print(f"[{engine},{method_str},{backup}] Full list of cache hits: {cache_hits}")

    print(f"[{engine},{method_str},{backup}][FINAL_METRIC] Average gap: {np.array(all_gaps).mean():.4f}")
    print(f"[{engine},{method_str},{backup}] Full list of gaps: {all_gaps}")

    max_gap = max(max(gaps) for gaps in all_gaps)
    # assert max_gap < LOGIT_GAP_THRESHOLD, f"COMPARE SPECULATIONS TO HF REFERENCE: max gap {max_gap} exceeds threshold {LOGIT_GAP_THRESHOLD}, {all_gaps=}"

    # Return the metrics the caller asserts on:
    #   all_ranks            — per-token rank of each engine-speculated token in the HF
    #                          draft's logits (eagle/phoenix only; empty otherwise).
    #   avg_acceptance_length — acceptance length recomputed from the engine's dumped prefixes.
    return all_ranks, avg_acceptance_length


def get_hf_target_activations_for_eagle_or_phoenix(target_model, all_tokens: list[int], eagle: bool = False, phoenix: bool = False) -> torch.Tensor:
    if eagle:
        assert not phoenix
        layers = EAGLE_LAYERS
    elif phoenix:
        layers = PHOENIX_LAYERS
    else:
        raise ValueError(f"Invalid speculator type: {eagle=}, {phoenix=}")
    with torch.no_grad():
        ids = torch.tensor([all_tokens], device=target_model.device, dtype=torch.long)
        out = target_model(ids, output_hidden_states=True, use_cache=False)
    acts = [out.hidden_states[li].squeeze(0).float() for li in layers]
    return torch.cat(acts, dim=-1).detach()  # [N, 3*D] 


def get_hf_logits(model, all_tokens: list[int]) -> torch.Tensor:
    with torch.no_grad():
        output = model.forward(torch.tensor([all_tokens], device=model.device), use_cache=False)
        return output.logits[0]


def get_hf_logits_for_completion(model, all_tokens: list[int], completion_length: int) -> torch.Tensor:
    with torch.no_grad():
        output = model.forward(torch.tensor([all_tokens], device=model.device), use_cache=False)
        return output.logits[0, -completion_length-1:-1]
