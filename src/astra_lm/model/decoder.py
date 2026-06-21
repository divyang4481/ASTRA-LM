import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from .config import ModelConfig
from .embeddings import TokenEmbedding
from .rope import RotaryEmbedding
from .norms import RMSNorm
from .block import DecoderBlock

class DecoderModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embeddings = TokenEmbedding(config.vocab_size, config.d_model)
        self.rope = RotaryEmbedding(config.head_dim, config.max_seq_len, config.rope_base)

        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
        return_attn_weights: bool = False,
        return_diagnostics: bool = False,
    ) -> Dict[str, Any]:
        batch_size, seq_len = input_ids.shape

        if seq_len > self.config.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.config.max_seq_len}")

        hidden_states = self.embeddings(input_ids)
        cos, sin = self.rope(seq_len)

        all_hidden_states = [hidden_states] if return_hidden_states else None
        all_diagnostics = []

        for block in self.blocks:
            hidden_states, diagnostics = block(
                hidden_states=hidden_states,
                cos=cos,
                sin=sin,
                attention_mask=attention_mask,
                return_attn_weights=return_attn_weights,
                return_diagnostics=return_diagnostics,
            )

            if return_hidden_states:
                all_hidden_states.append(hidden_states)
            all_diagnostics.append(diagnostics)

        hidden_states = self.norm(hidden_states)

        return {
            "hidden_states": hidden_states,
            "all_hidden_states": all_hidden_states,
            "diagnostics": all_diagnostics,
        }

class DecoderForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = DecoderModel(config)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embeddings.embedding.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_hidden_states: bool = False,
        return_attn_weights: bool = False,
        return_diagnostics: bool = False,
    ) -> Dict[str, Any]:

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_hidden_states=return_hidden_states,
            return_attn_weights=return_attn_weights,
            return_diagnostics=return_diagnostics,
        )

        hidden_states = outputs["hidden_states"]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift logits and labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        result = {
            "logits": logits,
            "diagnostics": outputs["diagnostics"],
        }
        if loss is not None:
            result["loss"] = loss
        if return_hidden_states:
            result["hidden_states"] = outputs["all_hidden_states"]

        return result
