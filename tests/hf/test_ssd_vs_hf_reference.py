import os
from pathlib import Path

import pytest
import requests
import torch
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer

from ssd import LLM, SamplingParams
from .eagle3_hf import Eagle3Model, load_eagle3_specforge
from .helpers import require_8b_target, require_eagle_llama_8b_draft, require_1b_draft, launch_tgl_server, wait_for_server, kill_server


PORT = 40023
LOGIT_GAP_THRESHOLD = 0.3
EAGLE_LAYERS = [2, 16, 29]
D_MODEL = 4096

ASYNC_BACKUPS = ["force-jit", "jit", "fast"]
SPECULATOR_TYPES = ["standalone", "eagle"]
CROSS_NODE = [True, False]

# @pytest.mark.parametrize("speculator_type", ["standalone"])
# @pytest.mark.parametrize("cross_node", [False])
# @pytest.mark.parametrize("backup", ["force-jit"])
@pytest.mark.parametrize("backup", ["force-jit"])  # [None])
@pytest.mark.parametrize("speculator_type", ["eagle", "standalone"])
@pytest.mark.parametrize("cross_node", [False])
@pytest.mark.parametrize("engine", ["tgl"])
@pytest.mark.parametrize("max_new_tokens", [128])
def test_ssd_vs_hf_reference(backup, speculator_type, cross_node, engine, max_new_tokens, tmp_path):
    lookahead = 4
    fanout = 3
    eagle = speculator_type in ["eagle", "sync_eagle"]
    sync_speculator = speculator_type in ["sync_standalone", "sync_eagle"]
    dtype = torch.bfloat16
    target_path = require_8b_target()
    draft_path = require_eagle_llama_8b_draft() if eagle else require_1b_draft()
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir(exist_ok=True)
    os.environ["SSD_DUMP_TENSORS_DIR"] = str(trace_dir)
    print(f"================================================================================")
    print(f"[{engine}] Launching {engine} engine with speculator type {speculator_type} and backup {backup}, trace directory {trace_dir}, max new tokens {max_new_tokens}, cross node {cross_node}", flush=True)
    print(f"================================================================================")

    tokenizer = AutoTokenizer.from_pretrained(target_path)
    prompt_tokens = tokenizer.apply_chat_template(
        [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Please tell me about the capital city of France."}],
        add_generation_prompt=True,
    )
    if isinstance(prompt_tokens, list):
        print(f"[{engine}] BANANA: {prompt_tokens=}", flush=True)
    else:
        prompt_tokens = prompt_tokens["input_ids"]

    # For each engine, we initialize the engine, send a request to it, and then tear down the engine.
    if engine == "tgl":
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
            print(f"[{engine}] prompt tokens: {prompt_tokens}", flush=True)
            print(f"[{engine}] response tokens: {completion_tokens}", flush=True)

        except Exception as e:
            print(f"[{engine}] error: {e}", flush=True)
            pytest.fail(f"[{engine}] error: {e}")

        finally:
            # TODO: We currently speedup the test by not killing the server; uncomment this when done debugging.
            print(f"[{engine}] killing server", flush=True)
            kill_server(tgl_server)
            assert not wait_for_server(PORT, timeout=3.0), "tgl server failed to stop"
            print(f"[{engine}] server stopped", flush=True)

            if cross_node:
                print(f"[{engine}] killing draft process", flush=True)
                kill_server(draft_process)
                print(f"[{engine}] draft process stopped", flush=True)

    elif engine == "ssd":
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
            eagle_layers=EAGLE_LAYERS if eagle else None,
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
    print("Beginning comparison of completion to hf reference")
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
    else:
        assert speculator_type == "standalone"
        draft_model = AutoModelForCausalLM.from_pretrained(draft_path, torch_dtype=dtype).to(draft_device)
        draft_model.eval()


    print(f"====================================================")
    print("Beginning SSD simulation")
    print(f"=====================================================")
    full_ssd_simulation(
        target_model,
        draft_model,
        prompt_tokens,
        completion_tokens,
        backup=backup,
        eagle=eagle,
        lookahead=lookahead,
        tokenizer=tokenizer,
    )

    # COMPARE SPECULATIONS TO HF REFERENCE
    print(f"====================================================")
    print("Beginning comparison of speculations to hf reference")
    print(f"=====================================================")
    compare_speculations_to_hf_reference(
        trace_dir,
        target_model,
        draft_model,
        prompt_tokens,
        completion_tokens,
        eagle=eagle,
        backup=backup,
        tokenizer=tokenizer,
        engine=engine,
        full_target_logits=full_target_logits,
    )


def compare_completion_to_hf_reference(
    model,
    prefix: list[int],
    completion: list[int],
    request_index: int,
    tokenizer: AutoTokenizer,
    engine: str = "tgl",
    full_target_logits: torch.Tensor = None,
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
    print("=============")
    greedy_preds = hf_logits_for_completion.argmax(dim=-1)
    matching = tokenizer.decode(greedy_preds) == tokenizer.decode(completion)
    match_str = "YES" if matching else " NO"
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
    draft_model: AutoModelForCausalLM | Eagle3Model,
    prompt_tokens: list[int],
    completion_tokens: list[int],
    backup: str = "force-jit",
    eagle: bool = False,
    lookahead: int = 4,
    full_target_logits: torch.Tensor = None,
    full_target_activations: torch.Tensor = None, # Note: These should already be projected into the draft space.
    duplicate_first_token: bool = True,
    tokenizer: AutoTokenizer = None,
):
    assert backup == "force-jit", "SSD simulation only supports force-jit backup for now"
    all_tokens = prompt_tokens + completion_tokens
    all_tokens_tensor = torch.tensor([all_tokens], device=draft_model.device, dtype=torch.long)
    draft_device = draft_model.device
    dtype = draft_model.lm_head.weight.dtype
    if full_target_activations is None and eagle:
        full_target_activations = get_hf_target_activations_for_eagle(target_model, all_tokens).to(draft_model.device)
        if duplicate_first_token:
            full_target_activations = torch.cat([
                full_target_activations[:1],
                full_target_activations
            ])
            full_target_activations = draft_model.fc(full_target_activations.to(dtype=dtype))
            print(f"[SIMULATION] full_target_activations.shape: {full_target_activations.shape}")
        else:
            raise ValueError("Unsupported at the moment")

    if full_target_logits is None:
        full_target_logits = get_hf_logits(target_model, all_tokens).to(draft_model.device)

    target_preds = full_target_logits.argmax(dim=-1)
    
    generated = 0
    acceptance_lengths = []
    probability_gaps = []
    # current_activation_index = len(prompt_tokens)
    done_generating = False
    while not done_generating:
        if eagle:
            tokens_remaining = all_tokens_tensor.shape[1] - (len(prompt_tokens) + generated)
            effective_lookahead = min(lookahead, tokens_remaining)
            if effective_lookahead <= 0:
                done_generating = True
                break
            current_activations = full_target_activations[:len(prompt_tokens) + generated + 1]
            for i in range(effective_lookahead):
                curr_len = len(prompt_tokens) + generated + i + 1
                current_prefix = all_tokens_tensor[0, :curr_len]
                print(f"[SIMULATION] current_activations.shape: {current_activations.shape}")
                if i > 0:
                    print(f"[SIMULATION] draft_activations.shape: {draft_activations.shape}")
                    current_activations = torch.cat([current_activations, draft_activations[-1:]])
                draft_activations = draft_model.forward_with_cond(current_prefix, torch.arange(curr_len, device=draft_device), current_activations)
            speculation_activations = draft_model.norm(draft_activations[-effective_lookahead:])
            speculation_logits = draft_model.lm_head(speculation_activations)
            speculation_logits = convert_to_full_vocab_logits(draft_model, speculation_logits)
            speculation_preds = speculation_logits.argmax(dim=-1)
        else:
            curr_len = len(prompt_tokens) + generated + lookahead
            current_prefix = all_tokens_tensor[:, :curr_len]
            speculation_logits = draft_model.forward(current_prefix).logits[0]
            speculation_logits = speculation_logits[-lookahead:]
            speculation_preds = speculation_logits.argmax(dim=-1)

        num_accepted = lookahead        
        for i in range(lookahead):
            curr_idx = len(prompt_tokens) + generated + i
            if curr_idx + 1 > len(all_tokens) - 1:
                done_generating = True
                break
            next_token = all_tokens[curr_idx + 1]
            if target_preds[curr_idx].item() != next_token:
                if tokenizer is not None:
                    target_pred_str = tokenizer.decode(target_preds[curr_idx])
                    next_token_str = tokenizer.decode(next_token)
                    print(f"[SIMULATION] Target prediction `{target_pred_str}` != next token `{next_token_str}` at index {curr_idx}")
                else:
                    print(f"[SIMULATION] Target prediction {target_preds[curr_idx].item()} != next token {next_token} at index {curr_idx}")
            if speculation_preds[i].item() != next_token:
                num_accepted = i
                break

        if not done_generating:
            acceptance_lengths.append(num_accepted)
            curr_probability_gaps = []
            for i in range(lookahead):
                curr_idx = len(prompt_tokens) + generated + i
                if curr_idx > len(all_tokens) - 1:
                    done_generating = True
                    break
                draft_logits = speculation_logits[i]
                target_logits = full_target_logits[curr_idx]
                draft_probs = torch.softmax(draft_logits, dim=-1)
                target_probs = torch.softmax(target_logits, dim=-1)
                gap = torch.linalg.norm(draft_probs - target_probs, ord=1).item()
                if gap > 0.5:
                    prefix = all_tokens_tensor[0, :curr_idx + 1]
                    decoded_prefix = tokenizer.decode(prefix)
                    print(f"[SIMULATION][{curr_idx}] Prefix: {decoded_prefix}")
                    draft_pred = draft_logits.argmax(dim=-1)
                    target_pred = target_logits.argmax(dim=-1)
                    draft_pred_str = tokenizer.decode(draft_pred)
                    target_pred_str = tokenizer.decode(target_pred)
                    print(f"[SIMULATION][{curr_idx}] |draft_probs - target_probs| = {gap:.4f}, Draft prediction `{draft_pred_str}`. Target prediction `{target_pred_str}`.")
                curr_probability_gaps.append(gap)

            if not done_generating:
                probability_gaps.append(curr_probability_gaps)

        generated += num_accepted + 1

    acc_lengths_array = np.array(acceptance_lengths) + 1
    print(f"[SIMULATION] Acceptance lengths: {acc_lengths_array.tolist()}")
    print(f"[SIMULATION] Average acceptance length: {acc_lengths_array.mean():.4f}")
    print(f"[SIMULATION] Probability gaps: {probability_gaps}")
    print(f"[SIMULATION] Average probability gap: {np.array(probability_gaps).mean():.4f}")


    return acceptance_lengths, probability_gaps


def convert_to_full_vocab_logits(draft_model: Eagle3Model, draft_logits: torch.Tensor) -> torch.Tensor:
    full_vocab_indices = torch.arange(draft_model.d2t.shape[0], device=draft_logits.device) + draft_model.d2t
    full_vocab_logits = draft_logits.new_full((draft_logits.shape[0], draft_model.cfg.vocab_size), float("-inf"))
    full_vocab_logits.index_copy_(-1, full_vocab_indices, draft_logits)
    return full_vocab_logits


def compare_completion_to_hf_reference_eagle(
    draft_model: Eagle3Model,
    prefix: list[int],
    speculation: list[int],
    eagle_acts: torch.Tensor,
    eagle_activation_index: int,  # where to start forward passes from.
    request_index: int,
    extend_token_ids: list[torch.Tensor],
    extend_counts: list[int],
    extend_activations: list[torch.Tensor],
    recovery_activations: list[torch.Tensor],
    prompt_eagle_acts: torch.Tensor,
    jit: bool,
    engine_acts: torch.Tensor,
    tokenizer: AutoTokenizer,
    engine: str = "tgl",
    funky: bool = False,
    prefixes: list[list[int]] = None,
    full_target_logits: torch.Tensor = None,
):
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
    all_eagle_acts_proj = draft_model.fc(eagle_acts)

    speculation_length = len(speculation)
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

    post_norm_final_draft_acts = draft_model.norm(joint_eagle_acts[-speculation_length:])
    draft_logits = draft_model.lm_head(post_norm_final_draft_acts)

    # Scatter draft-vocab draft_logits into target-vocab space via d2t so argmax /
    # indexing by the engine's target-vocab ids is well-defined. Non-draft
    # positions stay -inf (the draft cannot produce those tokens).
    draft_logits = convert_to_full_vocab_logits(draft_model, draft_logits)

    greedy_preds = draft_logits.argmax(dim=-1)

    # print(f"[{engine}] model moved to cuda", flush=True)
    # hf_logits_for_speculation = get_hf_logits_for_speculation(model, all_tokens, speculation_length)
    # print(f"[{engine}] hf draft_logits for speculation loaded", flush=True)
    gaps = []
    for i in range(speculation_length):
        speculation_token = speculation[i]
        hf_logit = draft_logits[i, speculation_token]
        hf_max_logit = draft_logits[i].max()
        # print(f"[{engine}] hf logit {hf_logit}, hf max logit {hf_max_logit}, logit_norm {torch.norm(hf_logits_for_speculation[i])}")
        gaps.append(torch.abs(hf_logit - hf_max_logit).item())

    max_gap = max(gaps)
    print("=============")
    matching = tokenizer.decode(greedy_preds) == tokenizer.decode(speculation)
    match_str = "YES" if matching else " NO"
    prefix_str = tokenizer.decode(prefix)
    print(f"[{engine}][{request_index}] prefix[-40:]: {prefix_str[-40:]}")
    print(f"[{engine}][{request_index}][{match_str}] speculation (hf reference): {tokenizer.decode(greedy_preds)}")
    print(f"[{engine}][{request_index}][{match_str}] speculation (engine - tgl): {tokenizer.decode(speculation)}")
    print(f"[{engine}][{request_index}][{match_str}] max gap: {max_gap}, gaps: {gaps}")
    # if max_gap > 0.0:
    #     pytest.set_trace()
    return gaps



def validate_request_and_response(request, response, request_num, eagle: bool = False):
    assert request["cache_keys"].shape[0] == 1
    assert request["num_tokens"].shape[0] == 1
    cache_keys = request["cache_keys"][0]
    num_accepted = cache_keys[1].item()
    if request_num == 0:
        assert num_accepted <= 0
    else:
        assert num_accepted >= 0

    if eagle:
        assert request["extend_token_ids"].shape[0] == 1
        assert request["extend_counts"].shape[0] == 1
        assert request["extend_activations"].shape[0] == 1
        assert request["recovery_activations"].shape[0] == 1

    assert response["cache_hits"].shape[0] == 1
    assert response["logits"].shape[0] == 1


def compare_speculations_to_hf_reference(
    trace_dir: Path,
    target_model,
    draft_model,
    prompt_tokens: list[int],
    completion_tokens: list[int],
    eagle: bool = False,
    backup: str = "force-jit",
    tokenizer: AutoTokenizer = None,
    engine: str = "tgl",
    full_target_logits: torch.Tensor = None,
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

    if not eagle:
        prompt_tokens_from_prefill_request = prefill_request["input_ids"].tolist()
        assert prompt_tokens_from_prefill_request == prompt_tokens, f"{prompt_tokens_from_prefill_request=} != {prompt_tokens=}"
    else:
        hf_full_eagle_acts = get_hf_target_activations_for_eagle(target_model, all_tokens).to(draft_model.device)
        hf_full_eagle_acts = torch.cat([
            hf_full_eagle_acts[:1],
            hf_full_eagle_acts
        ])
        prompt_eagle_acts = prefill_request["eagle_acts"].to(draft_model.device)
        prompt_len = prompt_eagle_acts.shape[0]
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
    if eagle:
        extend_token_ids = []
        extend_counts = []
        extend_activations = []
        extend_activations_accepted = []
        recovery_activations = []
    # TODO: Do this per request, by having a dictionary indexed by sequence ID.
    for i in range(len(speculation_requests)):
        request = speculation_requests[i]
        response = speculation_responses[i]
        validate_request_and_response(request, response, i, eagle)

        cache_keys = request["cache_keys"][0]
        num_tokens.append(request["num_tokens"][0].item())
        num_accepted.append(cache_keys[1].item())
        rec_token = cache_keys[2].item()
        if i == 0:
            prefixes.append(prompt_tokens + [rec_token])
        else:
            # Does the speculation contain the recovery token? I think it does?
            prefixes.append(prefixes[-1] + speculations[-1][:num_accepted[-1]] + [rec_token])

        if eagle:
            extend_token_ids.append(request["extend_token_ids"][0])
            extend_counts.append(request["extend_counts"][0].item())
            extend_activations.append(request["extend_activations"][0])
            recovery_activations.append(request["recovery_activations"][0])
            print(f"[{engine}] extend_activations.shape: {extend_activations[-1].shape}")

        # TODO: It seems speculations is shape [lookahead] instead of [batch_size, lookahead]. Fix this?
        speculations.append(response["speculations"].tolist())
        cache_hits.append(response["cache_hits"][0].item())
        logits.append(response["logits"][0].tolist())
        # if tokenizer is not None:
        #     prefix_text = tokenizer.decode(prefixes[-1])
        #     speculations_text = tokenizer.decode(speculations[-1])
        #     print(f"[{engine}] prefix text: {prefix_text}")
        #     print(f"[{engine}] speculations text: {speculations_text}")
        #     print(f"[{engine}] num accepted: {num_accepted[-1]}")
        #     # print(f"[{engine}] num tokens: {num_tokens[-1]}")
        #     print(f"[{engine}] rec token: {tokenizer.decode([rec_token])}")
        # else:
        #     print(f"[{engine}] prefix: {prefixes[-1]}, speculation: {speculations[-1]}, num_accepted: {num_accepted[-1]}, num_tokens: {num_tokens[-1]}, rec_token: {rec_token}")

    prompt_len = len(prompt_tokens)
    if eagle:
        engine_acts = torch.zeros((len(all_tokens), 4096*3), dtype=draft_model.lm_head.weight.dtype, device="cpu")
        engine_acts[:prompt_len] = prompt_eagle_acts.cpu()
        t = prompt_len
        for i in range(len(speculation_requests)):
            num_accept = extend_counts[i]
            if num_accept > 0:
                engine_acts[t: t + num_accept] = extend_activations[i][:num_accept].cpu()
            engine_acts[t + num_accept] = recovery_activations[i].cpu()
            t += 1 + num_accept
        print(f"FINAL OFFSET: {t}")
        diffs = [
            (torch.norm(hf_full_eagle_acts[i].cpu() - engine_acts[i]) / torch.norm(hf_full_eagle_acts[i].cpu())).item()
            for i in range(t)
        ]
        for i, diff in enumerate(diffs):
            print(f"DIFF {i}: {diff:.4f}")

        print(f"[{engine}] eagle extend counts: {extend_counts}")

    # pytest.set_trace()
    all_gaps = []
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

        if not eagle:
            gaps, _ = compare_completion_to_hf_reference(
                draft_model,
                prefix,
                speculation,
                i,
                tokenizer,
                engine=engine,
                full_target_logits=full_target_logits,
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

            gaps = compare_completion_to_hf_reference_eagle(
                draft_model,
                prefix,
                speculation,
                hf_full_eagle_acts,
                eagle_activation_index,
                i,
                extend_token_ids,
                extend_counts,
                extend_activations,
                recovery_activations,
                prompt_eagle_acts,
                jit,
                engine_acts,
                tokenizer,
                engine=engine,
                funky=False,
                prefixes=prefixes,
                full_target_logits=full_target_logits,
            )
            all_gaps.append(gaps)

    method_str = "eagle" if eagle else "standalone"
    print(f" ****** SUMMARY OF ALL RESULTS (engine={engine}, method={method_str}, backup={backup}) ******")

    if eagle:
        # extend counts don't include the recovery token, so we add 1 to the average.
        print(f"[{engine},{method_str},{backup}][FINAL_METRIC] Average acceptance lengths: {1 + (sum(extend_counts) / (len(extend_counts) - 1)):.4f}")
        print(f"[{engine},{method_str},{backup}] Full list of acceptance lengths: {extend_counts}")
    else:
        prefix_lengths = np.array([len(p) for p in prefixes])
        acceptance_lengths = prefix_lengths[1:] - prefix_lengths[:-1]   
        print(f"[{engine},{method_str},{backup}][FINAL_METRIC] Average acceptance lengths: {sum(acceptance_lengths) / len(acceptance_lengths):.4f}")
        print(f"[{engine},{method_str},{backup}] Full list of acceptance lengths: {acceptance_lengths}")

    print(f"[{engine},{method_str},{backup}][FINAL_METRIC] Average cache hit rate: {sum(cache_hits) / len(cache_hits)}")
    print(f"[{engine},{method_str},{backup}] Full list of cache hits: {cache_hits}")
    
    print(f"[{engine},{method_str},{backup}][FINAL_METRIC] Average gap: {np.array(all_gaps).mean():.4f}")
    print(f"[{engine},{method_str},{backup}] Full list of gaps: {all_gaps}")

    max_gap = max(max(gaps) for gaps in all_gaps)
    assert max_gap < LOGIT_GAP_THRESHOLD, f"COMPARE SPECULATIONS TO HF REFERENCE: max gap {max_gap} exceeds threshold {LOGIT_GAP_THRESHOLD}, {all_gaps=}"


def get_hf_target_activations_for_eagle(target_model, all_tokens: list[int]) -> torch.Tensor:
    with torch.no_grad():
        ids = torch.tensor([all_tokens], device=target_model.device, dtype=torch.long)
        out = target_model(ids, output_hidden_states=True, use_cache=False)
    acts = [out.hidden_states[li].squeeze(0).float() for li in EAGLE_LAYERS]
    return torch.cat(acts, dim=-1).detach()  # [N, 3*D] 


def get_hf_logits(model, all_tokens: list[int]) -> torch.Tensor:
    with torch.no_grad():
        output = model.forward(torch.tensor([all_tokens], device=model.device), use_cache=False)
        return output.logits[0]


def get_hf_logits_for_completion(model, all_tokens: list[int], completion_length: int) -> torch.Tensor:
    with torch.no_grad():
        output = model.forward(torch.tensor([all_tokens], device=model.device), use_cache=False)
        return output.logits[0, -completion_length-1:-1]
