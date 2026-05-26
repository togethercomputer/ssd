import torch
from torch import nn
import torch.distributed as dist
from transformers import LlamaConfig
from ssd.layers.activation import SiluAndMul
from ssd.layers.attention import Attention
from ssd.layers.layernorm import RMSDNorm
from ssd.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from ssd.layers.rotary_embedding import get_rope
from ssd.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from ssd.models.llama3 import LlamaAttention, LlamaMLP
from ssd.utils import profile


class Eagle3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int,
        rms_norm_eps: float,
        head_dim: int | None,
        rope_theta: float,
        rope_scaling: dict | None,
        draft: bool,
        speculate: bool,
        spec_k: int,
        async_fan_out: int,
        draft_async: bool,
        tp_group: dist.ProcessGroup | None,
        tp_size: int,
    ):
        super().__init__()
        self.draft = draft
        self.draft_async = draft_async
        self.tp_group = tp_group
        self.tp_size = tp_size
        
        self.total_num_heads = num_heads
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        
        self.qkv_proj = QKVParallelLinear(
            2 * hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            tp_group=self.tp_group,
            tp_size=self.tp_size,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            tp_group=self.tp_group,
            tp_size=self.tp_size,
        )
        
        if rope_scaling is not None:
            rope_scaling = None
        
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            draft=draft,
            speculate=speculate,
            draft_async=draft_async,
            use_eagle=True,
            F=async_fan_out,
            K=spec_k,
        )

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # Ensure all inputs to attn are contiguous
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o)
        return output


class Eagle3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        draft: bool,
        speculate: bool,
        spec_k: int,
        async_fan_out: int,
        draft_async: bool,
        tp_group: dist.ProcessGroup | None = None,
        tp_size: int = 1,
    ):
        super().__init__()
        self.self_attn = Eagle3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 500000),
            rope_scaling=getattr(config, "rope_scaling", None),
            draft=draft,
            speculate=speculate,
            spec_k=spec_k,
            async_fan_out=async_fan_out,
            draft_async=draft_async,
            tp_group=tp_group,
            tp_size=tp_size,
        )
        self.mlp = LlamaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            tp_group=tp_group,
            tp_size=tp_size,
        )
        self.input_layernorm = RMSDNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.conditioning_feature_ln = RMSDNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSDNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        token_embeddings: torch.Tensor,
        conditioning_features: torch.Tensor,
    ) -> torch.Tensor:
        normed_tokens = self.input_layernorm(token_embeddings)
        normed_conditioning = self.conditioning_feature_ln(conditioning_features)
        hidden_states = torch.cat([normed_tokens, normed_conditioning], dim=-1)
        
        hidden_states = self.self_attn(positions, hidden_states) 
        # use conditioning features as residual stream, not token embeddings, as per SAFEAILab ref impl
        hidden_states, residual = self.post_attention_layernorm(hidden_states, conditioning_features) 
        hidden_states = self.mlp(hidden_states) + residual
        return hidden_states 

class Eagle3DraftModel(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,        draft: bool = False,
        speculate: bool = False,
        spec_k: int = 1,
        async_fan_out: int = 1,
        draft_async: bool = False,
        use_eagle: bool = False,
        eagle_layers: list[int] | None = None,
        tp_group: dist.ProcessGroup | None = None,
        tp_size: int = 1,
    ) -> None:
        super().__init__()
        self.draft = draft
        self.speculate = speculate
        self.spec_k = spec_k
        self.async_fan_out = async_fan_out
        self.draft_async = draft_async
        self.use_eagle = use_eagle
        self.eagle_layers = eagle_layers
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            draft_async=self.draft_async,
            tp_group=tp_group,
            tp_size=tp_size,
        )
        assert config.num_hidden_layers == 1, "ERROR in Eagle3DraftModel: config.num_hidden_layers must be 1"
        self.layer = Eagle3DecoderLayer(
            config,
            draft=self.draft,
            speculate=self.speculate,
            spec_k=self.spec_k,
            async_fan_out=self.async_fan_out,
            draft_async=self.draft_async,
            tp_group=tp_group,
            tp_size=tp_size,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        target_hidden_states_projected: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        token_embeddings = self.embed_tokens(input_ids)
        hidden_states = self.layer(positions, token_embeddings, target_hidden_states_projected)
        return hidden_states

class Eagle3DraftForCausalLM(nn.Module):
    packed_modules_mapping = {
        "midlayer.self_attn.q_proj": ("model.layer.self_attn.qkv_proj", "q"),
        "midlayer.self_attn.k_proj": ("model.layer.self_attn.qkv_proj", "k"),
        "midlayer.self_attn.v_proj": ("model.layer.self_attn.qkv_proj", "v"),
        "midlayer.mlp.gate_proj": ("model.layer.mlp.gate_up_proj", 0),
        "midlayer.mlp.up_proj": ("model.layer.mlp.gate_up_proj", 1),
    }

    def __init__(
        self,
        config: LlamaConfig,  
        draft: bool = False,
        speculate: bool = False,
        use_eagle: bool = False,
        eagle_layers: list[int] | None = None,
        d_model_target: int = 4096,
        spec_k: int = 1,
        async_fan_out: int = 1,
        draft_async: bool = False,
        tp_group: dist.ProcessGroup | None = None,
        tp_size: int = 1,
        debug_mode: bool = False,
    ) -> None:
        super().__init__()

        assert draft, "ERROR in Eagle3DraftForLlama3: draft must be True"
        assert use_eagle, "ERROR in Eagle3DraftForLlama3: config.use_eagle must be True"
        assert eagle_layers is not None, "ERROR in Eagle3DraftForLlama3: eagle_layers must be set"

        # this will be the draft that does tree decode, just needs a modified fwd pass that takes in hidden states and uses fc and dicts to sample, etc 
        self.config = config
        self.draft = draft
        self.async_fan_out = async_fan_out
        self.draft_async = draft_async
        self.tp_group = tp_group
        self.tp_size = tp_size
        self.use_eagle = use_eagle
        self.eagle_layers = eagle_layers
        self.d_model_target = d_model_target
        self.d2t = {}  # loaded by loader.py, converted to tensor after load_model
        self.t2d = {}  # loaded by loader.py, converted to tensor after load_model
        self.d2t_tensor = None  # will be set after load_model
        self.t2d_tensor = None  # will be set after load_model
        self.draft_vocab_size = config.draft_vocab_size if hasattr(config, 'draft_vocab_size') else config.vocab_size
        self.vocab_trim = self.draft_vocab_size != self.config.vocab_size
        self.debug_mode = debug_mode
        self._debug_saved = False  # Track if we've already saved debug data
        assert not (tp_group is None and self.tp_size > 1), "ERROR in LlamaForCausalLM: tp_group is None and tp_size > 1"

        print(f'Starting Eagle3DraftForCausalLM init, draft={draft}, speculate={speculate}, spec_k={spec_k}')
        self.fc = nn.Linear(len(self.eagle_layers) * d_model_target, config.hidden_size, bias=False)
        self.final_norm = RMSDNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.model = Eagle3DraftModel(config, draft, speculate, spec_k, async_fan_out, draft_async, use_eagle=use_eagle, eagle_layers=eagle_layers, tp_group=tp_group, tp_size=self.tp_size)
        self.lm_head = ParallelLMHead(
            self.draft_vocab_size,  # LM head size (subset of tokens draft can propose)
            config.hidden_size,
            draft_async=draft_async,
            tp_group=tp_group,
            tp_size=self.tp_size,
        )
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data
        print(f'Finishing Eagle3DraftForCausalLM init, draft={draft}, speculate={speculate}, spec_k={spec_k}') 

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Events fire only on eager paths (prefill / enforce_eager). On CG paths
        # this body is captured into the graph — replay timing is collected at
        # the cudagraph_helpers level instead.
        ev = profile.new_events(3)
        if ev: ev[0].record()

        # Only project if this is target hidden states (3 * d_model_target dimension)
        if hidden_states.shape[-1] == 3 * self.d_model_target:
            # This is the first prefill with target activations
            if self.debug_mode and not self._debug_saved and input_ids.shape[0] != 2048:
                self._save_debug_inputs(input_ids, positions, hidden_states)
                self._debug_saved = True

            hidden_states_projected = self.fc(hidden_states.to(self.fc.weight.dtype))  # [num_tokens, d_model_draft]
            had_fc = True
        else:
            hidden_states_projected = hidden_states # draft self-conditioning output, already d_model_draft from prenorm
            had_fc = False

        if ev: ev[1].record()

        # Forward through draft model with conditioning
        prenorm = self.model(input_ids, hidden_states_projected, positions)

        if ev: ev[2].record()

        profile.emit(
            "Eagle3DraftForCausalLM.forward",
            ["fc", "decoder_layer"],
            ev,
            n_tokens=input_ids.shape[0],
            had_fc=had_fc,
        )
        return prenorm
    
    def _save_debug_inputs(self, input_ids: torch.Tensor, positions: torch.Tensor, target_hidden_states: torch.Tensor):
        """Save draft prefill inputs for debugging."""
        import os
        # Get token embeddings
        with torch.no_grad():
            token_embeddings = self.model.embed_tokens(input_ids)
        
        debug_data = {
            'input_ids': input_ids.cpu(),
            'positions': positions.cpu(),
            'target_hidden_states': target_hidden_states.cpu(),  # [num_tokens, 3 * d_model_target]
            'token_embeddings': token_embeddings.cpu(),
            'd_model_target': self.d_model_target,
            'eagle_layers': self.eagle_layers,
        }
        
        os.makedirs('debug_outputs', exist_ok=True)
        save_path = 'debug_outputs/draft_prefill_inputs.pt'
        torch.save(debug_data, save_path)
        print(f"[DEBUG] Saved draft prefill inputs to {save_path}")
        print(f"[DEBUG] Shapes: input_ids={input_ids.shape}, positions={positions.shape}, target_hidden_states={target_hidden_states.shape}, token_embeddings={token_embeddings.shape}")


    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        last_only: bool = True,
    ) -> torch.Tensor:
        # compute_logits is always called outside the CUDA graph (after replay
        # in cudagraph_helpers), so timing always works.
        n_in = hidden_states.shape[0]
        ev = profile.new_events(4 if self.vocab_trim else 3)
        if ev: ev[0].record()

        hidden_states = self.final_norm(hidden_states)
        if ev: ev[1].record()

        logits = self.lm_head(hidden_states, last_only=last_only)  # [B, draft_vocab_size]

        if logits.dim() == 3:
            logits = logits.view(-1, logits.shape[-1])

        if ev: ev[2].record()

        B = logits.shape[0]
        if self.vocab_trim:
            # Expand draft vocab logits to full target vocab using d2t mapping.
            # Map draft indices to target vocab positions: target_idx = draft_idx + d2t_tensor[draft_idx]
            assert self.d2t_tensor is not None, "d2t_tensor must be loaded before inference"
            assert self.t2d_tensor is not None, "t2d_tensor must be loaded before inference"
            base = torch.arange(self.draft_vocab_size, device=logits.device)
            target_indices = base + self.d2t_tensor  # [draft_vocab_size]
            logits_full = logits.new_full((B, self.config.vocab_size), float('-inf'))
            logits_full[:, target_indices] = logits
            logits = logits_full

            if ev: ev[3].record()

        profile.emit(
            "Eagle3DraftForCausalLM.compute_logits",
            ["final_norm", "lm_head"] + (["d2t_scatter"] if self.vocab_trim else []),
            ev,
            n_tokens=n_in,
            last_only=last_only,
        )
        return logits
