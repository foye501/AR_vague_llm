from __future__ import annotations

from typing import Any


def load_tokenizer_with_diffusion_tokens(
    AutoTokenizer,
    *,
    tokenizer_name: str,
    pad_token: str = "[PAD]",
    mask_token: str = "[MASK]",
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": pad_token})
    if tokenizer.mask_token_id is None:
        tokenizer.add_special_tokens({"mask_token": mask_token})
    return tokenizer


def configure_qwen_config_for_bidirectional_denoising(config: Any, tokenizer) -> None:
    config.vocab_size = len(tokenizer)
    config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        config.eos_token_id = tokenizer.eos_token_id
    if tokenizer.bos_token_id is not None:
        config.bos_token_id = tokenizer.bos_token_id
    config.use_cache = False
    config.is_decoder = False
    config.tie_word_embeddings = False


def build_bidirectional_qwen_for_masked_lm_class():
    import torch
    from torch import nn
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Model, Qwen2PreTrainedModel

    class BidirectionalQwenForMaskedLM(Qwen2PreTrainedModel):
        _tied_weights_keys: list[str] = []

        def __init__(self, config):
            super().__init__(config)
            config.use_cache = False
            config.is_decoder = False
            config.tie_word_embeddings = False
            self.model = Qwen2Model(config)
            self.vocab_size = config.vocab_size
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self._disable_causal_attention_flags()
            self.post_init()

        def _disable_causal_attention_flags(self) -> None:
            self.model.has_sliding_layers = False
            for layer in self.model.layers:
                layer.attention_type = "full_attention"
                layer.self_attn.is_causal = False
                layer.self_attn.sliding_window = None

        def get_input_embeddings(self):
            return self.model.embed_tokens

        def set_input_embeddings(self, value):
            self.model.embed_tokens = value

        def get_output_embeddings(self):
            return self.lm_head

        def set_output_embeddings(self, value):
            self.lm_head = value

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            inputs_embeds=None,
            labels=None,
            **kwargs,
        ):
            if input_ids is None and inputs_embeds is None:
                raise ValueError("input_ids or inputs_embeds is required")
            batch_size, sequence_length = input_ids.shape if input_ids is not None else inputs_embeds.shape[:2]
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            dtype = self.model.embed_tokens.weight.dtype
            full_attention_mask = _make_bidirectional_attention_mask(
                attention_mask=attention_mask,
                batch_size=batch_size,
                sequence_length=sequence_length,
                device=device,
                dtype=dtype,
            )
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=full_attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                use_cache=False,
                **kwargs,
            )
            logits = self.lm_head(outputs.last_hidden_state)

            loss = None
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(logits.reshape(-1, self.config.vocab_size), labels.reshape(-1))

            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

    return BidirectionalQwenForMaskedLM


def load_bidirectional_qwen_for_masked_lm(
    AutoConfig,
    *,
    model_name: str,
    tokenizer,
    from_scratch: bool,
):
    model_cls = build_bidirectional_qwen_for_masked_lm_class()
    if from_scratch:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        configure_qwen_config_for_bidirectional_denoising(config, tokenizer)
        model = model_cls(config)
    else:
        model = model_cls.from_pretrained(model_name, trust_remote_code=True)
        configure_qwen_config_for_bidirectional_denoising(model.config, tokenizer)
        if _embedding_vocab_size(model) != len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))
        model._disable_causal_attention_flags()
    return model


def _make_bidirectional_attention_mask(
    *,
    attention_mask,
    batch_size: int,
    sequence_length: int,
    device,
    dtype,
):
    import torch

    if attention_mask is None:
        attention_mask = torch.ones((batch_size, sequence_length), device=device, dtype=torch.long)
    key_mask = attention_mask.to(device=device)
    blocked = (1.0 - key_mask[:, None, None, :].to(dtype)) * torch.finfo(dtype).min
    full_mask = blocked.expand(batch_size, 1, sequence_length, sequence_length)
    return {
        "full_attention": full_mask,
        "sliding_attention": full_mask,
    }


def _embedding_vocab_size(model) -> int | None:
    embeddings = model.get_input_embeddings()
    if embeddings is None or not hasattr(embeddings, "weight"):
        return None
    return int(embeddings.weight.shape[0])
