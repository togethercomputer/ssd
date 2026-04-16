import torch
import torch.distributed as dist
from transformers import LlamaConfig

from ssd.layers.linear import RowParallelLinear
from ssd.models.llama3 import LlamaForCausalLM


class PhoenixLlamaForCausalLM(LlamaForCausalLM):
    def __init__(
        self,
        config: LlamaConfig,
        draft: bool = True,
        speculate: bool = True,
        use_eagle: bool = False,
        use_phoenix: bool = True,
        eagle_layers: list[int] | None = None,
        d_model_target: int = 4096,
        spec_k: int = 1,
        async_fan_out: int = 1,
        draft_async: bool = False,
        tp_group: dist.ProcessGroup | None = None,
        tp_size: int = 1,
        debug_mode: bool = False,
    ) -> None:
        assert draft, "ERROR in PhoenixLlamaForCausalLM: draft must be True"
        assert use_phoenix, "ERROR in PhoenixLlamaForCausalLM: config.use_phoenix must be True"
        assert not use_eagle, "ERROR in PhoenixLlamaForCausalLM: config.use_eagle must be False"
        super().__init__(
            config,
            draft=True,
            speculate=True,
            use_eagle=False,
            use_phoenix=True,
            eagle_layers=None,
            spec_k=spec_k,
            async_fan_out=async_fan_out,
            draft_async=draft_async,
            tp_group=tp_group,
            tp_size=tp_size,
        )
        self.d_model_target = d_model_target
        self.debug_mode = debug_mode
        self.eh_proj = RowParallelLinear(
            self.d_model_target + config.hidden_size,
            config.hidden_size,
            bias=True,
            tp_group=tp_group,
            tp_size=tp_size,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        input_embeds = self.model.embed_tokens(input_ids)
        hidden_states = torch.cat((input_embeds, hidden_states), dim=-1)
        hidden_states = self.eh_proj(hidden_states.to(self.eh_proj.weight.dtype))
        out = self.model(input_ids, positions, hidden_states)
        return out

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        last_only: bool = True, 
    ) -> torch.Tensor:
        logits = self.lm_head(hidden_states, last_only=last_only)

        if logits.dim() == 3:
            logits = logits.view(-1, logits.shape[-1])

        return logits
