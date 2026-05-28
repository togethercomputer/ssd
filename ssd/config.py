import os
from dataclasses import dataclass
from transformers import AutoConfig
import torch
from ssd.paths import DEFAULT_TARGET, DEFAULT_DRAFT


@dataclass
class Config:
    model: str = DEFAULT_TARGET
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 1
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.7
    num_gpus: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 1
    num_kvcache_blocks: int = -1
    device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # spec config args
    draft_hf_config: AutoConfig | None = None
    speculate: bool = False 
    draft: str = DEFAULT_DRAFT
    speculate_k: int = 1
    draft_async: bool = False

    # async spec only
    async_fan_out: int = 3
    fan_out_list: list[int] | None = None
    fan_out_list_miss: list[int] | None = None
    sampler_x: float | None = None 
    jit_speculate: bool = False
    force_jit_speculate: bool = False
    async_nccl_port: int | None = None
    async_nccl_host: str = "127.0.0.1"
    communicate_logits: bool = False
    communicate_cache_hits: bool = False

    # eagle3 / phoenix
    use_eagle: bool = False 
    use_phoenix: bool = False
    eagle_layers: list[int] | None = None   
    d_model_target: int | None = None
    tokenizer_path: str | None = None

    # Runtime dtype for both target and draft. When set, overrides whatever
    # the checkpoints' config.json files say. FA4 only supports fp16/bf16, so
    # callers (e.g. the sglang scheduler) should pass the target server's
    # resolved dtype here. Left as None for standalone callers that are happy
    # to inherit dtype from the checkpoints.
    dtype: torch.dtype | None = None

    # Debugging
    verbose: bool = False
    debug_mode: bool = False
    max_steps: int | None = None

    @property
    def max_blocks(self): 
        return (self.max_model_len + self.kvcache_block_size - 1) // self.kvcache_block_size

    @property
    def use_eagle_or_phoenix(self):
        return self.use_eagle or self.use_phoenix

    def __post_init__(self):
        model = self.model
        assert os.path.isdir(model)

        assert 1 <= self.num_gpus <= 8 # this codebase only works on one node
        self.hf_config = AutoConfig.from_pretrained(model)

        # Multimodal targets (e.g. Kimi K2.5) nest the LM config under `text_config`. The rest of this
        # file expects a flat LM-style config (num_hidden_layers, hidden_size, rope_theta, ...), so
        # collapse to the text sub-config when present. We don't use vision/quant fields here.
        if hasattr(self.hf_config, "text_config"):
            self.hf_config = self.hf_config.text_config

        if not self.speculate:
            if self.max_model_len:
                self.max_model_len = min(
                    self.max_model_len, self.hf_config.max_position_embeddings)
            else:
                self.max_model_len = self.hf_config.max_position_embeddings
        else:
            draft = self.draft
            self.draft_hf_config = AutoConfig.from_pretrained(draft)
            if self.max_model_len:
                self.max_model_len = min(
                    self.max_model_len, self.draft_hf_config.max_position_embeddings)
            else:
                self.max_model_len = self.draft_hf_config.max_position_embeddings

            if self.draft_async:
                if self.fan_out_list is None: 
                    self.fan_out_list = [self.async_fan_out] * (self.speculate_k + 1)
                    self.MQ_LEN = sum(self.fan_out_list)
                if not self.jit_speculate:
                    print(f'[Config] Setting fan_out_list_miss to [sum(fan_out_list)] + [0] * speculate_k because jit_speculate is False', flush=True)
                    self.fan_out_list_miss = [sum(self.fan_out_list)] + [0] * self.speculate_k
                elif self.fan_out_list_miss is None:
                    # If you are jit speculating, always use the same fan_out_list for misses as for hits.
                    self.fan_out_list_miss = self.fan_out_list

                assert sum(self.fan_out_list_miss) == sum(self.fan_out_list), "ERROR in Config: fan_out_list_miss must be the same as fan_out_list"

        if self.use_eagle_or_phoenix:
            if self.use_eagle and self.eagle_layers is None:
                # Note: Currently we don't support Phoenix2, so only Eagle3 uses the `eagle_layers` config.
                L = self.hf_config.num_hidden_layers
                self.eagle_layers = [2, L//2, L-3]
                print(f'[Config] just set eagle_layers={self.eagle_layers}', flush=True)
            # Eagle draft must use target's rope_theta (draft config may default to wrong value)
            if self.speculate and self.draft_hf_config is not None:
                target_rope_theta = getattr(self.hf_config, 'rope_theta', 500000.0)
                draft_rope_theta = getattr(self.draft_hf_config, 'rope_theta', 10000.0)
                if target_rope_theta != draft_rope_theta:
                    print(f'[Config] Overriding eagle draft rope_theta: {draft_rope_theta} -> {target_rope_theta}', flush=True)
                    self.draft_hf_config.rope_theta = target_rope_theta
                # Also override max_position_embeddings for correct RoPE cache size
                # NOTE: Do NOT change max_model_len here - it was already correctly capped.
                # Only change draft_hf_config.max_position_embeddings for RoPE.
                target_max_pos = getattr(self.hf_config, 'max_position_embeddings', 8192)
                draft_max_pos = getattr(self.draft_hf_config, 'max_position_embeddings', 2048)
                if target_max_pos != draft_max_pos:
                    print(f'[Config] Overriding eagle draft max_position_embeddings: {draft_max_pos} -> {target_max_pos}', flush=True)
                    self.draft_hf_config.max_position_embeddings = target_max_pos

        if self.dtype is not None:
            assert self.dtype in (torch.float16, torch.bfloat16), (
                f"[Config] dtype={self.dtype} is not supported; FA4 requires "
                f"float16 or bfloat16. Pass a supported dtype from the caller."
            )
            self.hf_config.torch_dtype = self.dtype
            if self.draft_hf_config is not None:
                self.draft_hf_config.torch_dtype = self.dtype

        if self.sampler_x is not None and not self.communicate_cache_hits:
            self.communicate_cache_hits = True
            print(f'[Config] Setting communicate_cache_hits to True because sampler_x is not None', flush=True)

        # assert self.max_num_batched_tokens >= self.max_model_len
        if self.max_num_batched_tokens < self.max_model_len:
            print(f'[Config] Warning: max_num_batched_tokens ({self.max_num_batched_tokens}) is less than max_model_len ({self.max_model_len})', flush=True)
            print(f'[Config] Setting max_num_batched_tokens to max_model_len', flush=True)
            self.max_num_batched_tokens = self.max_model_len
