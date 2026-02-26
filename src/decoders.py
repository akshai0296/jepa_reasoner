"""
Output decoders ("Talkers") that convert predicted latent embeddings into
human-readable text (code, math solutions, or natural language explanations).

The decoder is conditioned on the predicted latent s_y_hat and generates
tokens auto-regressively. It is trained separately (or jointly) with
standard cross-entropy on the ground-truth solution tokens.
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel


class LatentConditionedDecoder(nn.Module):
    """
    A small GPT-2-style causal language model conditioned on a latent vector.

    The latent vector is projected into a prefix sequence of "latent tokens"
    that the decoder attends to before generating output tokens. This is
    similar to prompt-tuning / prefix-tuning but with learned latent inputs.
    """

    def __init__(
        self,
        latent_dim: int = 768,
        vocab_size: int = 50257,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        max_seq_len: int = 512,
        num_latent_tokens: int = 8,
        tokenizer_name: str = "gpt2",
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_latent_tokens = num_latent_tokens
        self.hidden_dim = hidden_dim

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=hidden_dim,
            n_layer=num_layers,
            n_head=num_heads,
            n_positions=max_seq_len + num_latent_tokens,
            resid_pdrop=0.1,
            attn_pdrop=0.1,
        )
        self.transformer = GPT2LMHeadModel(config)

        self.latent_to_prefix = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * num_latent_tokens),
            nn.GELU(),
            nn.LayerNorm(hidden_dim * num_latent_tokens),
        )

    def _build_prefix(self, s_y_hat: torch.Tensor) -> torch.Tensor:
        """Convert latent vector to prefix token embeddings."""
        prefix = self.latent_to_prefix(s_y_hat)  # [B, hidden*N]
        return prefix.view(-1, self.num_latent_tokens, self.hidden_dim)

    def forward(
        self,
        s_y_hat: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Training forward pass.

        Args:
            s_y_hat:    predicted latent [batch, latent_dim]
            target_ids: ground-truth token ids [batch, seq_len]
            target_mask: attention mask [batch, seq_len]

        Returns:
            dict with 'loss' and 'logits'
        """
        batch_size = s_y_hat.size(0)
        prefix_embeds = self._build_prefix(s_y_hat)  # [B, N_prefix, hidden]

        token_embeds = self.transformer.transformer.wte(target_ids)  # [B, seq, hidden]
        inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)

        prefix_mask = torch.ones(
            batch_size, self.num_latent_tokens, device=s_y_hat.device, dtype=target_mask.dtype
        )
        full_mask = torch.cat([prefix_mask, target_mask], dim=1)

        labels = torch.cat(
            [
                torch.full(
                    (batch_size, self.num_latent_tokens),
                    -100,
                    device=target_ids.device,
                    dtype=target_ids.dtype,
                ),
                target_ids,
            ],
            dim=1,
        )
        labels = labels.masked_fill(full_mask == 0, -100)

        outputs = self.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            labels=labels,
        )

        return {"loss": outputs.loss, "logits": outputs.logits}

    @torch.no_grad()
    def generate(
        self,
        s_y_hat: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> list[str]:
        """Decode a latent vector into text."""
        self.eval()
        batch_size = s_y_hat.size(0)
        prefix_embeds = self._build_prefix(s_y_hat)

        bos_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        start_tokens = torch.full(
            (batch_size, 1), bos_id, device=s_y_hat.device, dtype=torch.long
        )
        start_embeds = self.transformer.transformer.wte(start_tokens)
        inputs_embeds = torch.cat([prefix_embeds, start_embeds], dim=1)

        generated = start_tokens
        past_key_values = None

        for step in range(max_new_tokens):
            if step == 0:
                outputs = self.transformer(
                    inputs_embeds=inputs_embeds,
                    use_cache=True,
                )
            else:
                outputs = self.transformer(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :] / max(temperature, 1e-8)

            if do_sample:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[mask] = float("-inf")
                probs = torch.softmax(sorted_logits, dim=-1)
                next_idx = torch.multinomial(probs, 1)
                next_token = sorted_indices.gather(-1, next_idx)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)

            if (next_token == self.tokenizer.eos_token_id).all():
                break

        texts = self.tokenizer.batch_decode(generated[:, 1:], skip_special_tokens=True)
        return texts


class PretrainedDecoder(nn.Module):
    """
    Uses a pretrained causal LM (e.g. GPT-2, CodeGen) and conditions
    it on the latent via prefix embeddings.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        latent_dim: int = 768,
        num_latent_tokens: int = 8,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.num_latent_tokens = num_latent_tokens
        hidden_dim = self.model.config.n_embd

        self.latent_to_prefix = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * num_latent_tokens),
            nn.GELU(),
            nn.LayerNorm(hidden_dim * num_latent_tokens),
        )

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

    def _build_prefix(self, s_y_hat: torch.Tensor) -> torch.Tensor:
        prefix = self.latent_to_prefix(s_y_hat)
        hidden_dim = self.model.config.n_embd
        return prefix.view(-1, self.num_latent_tokens, hidden_dim)

    def forward(self, s_y_hat, target_ids, target_mask):
        batch_size = s_y_hat.size(0)
        prefix_embeds = self._build_prefix(s_y_hat)
        token_embeds = self.model.transformer.wte(target_ids)
        inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)

        prefix_mask = torch.ones(
            batch_size, self.num_latent_tokens, device=s_y_hat.device, dtype=target_mask.dtype
        )
        full_mask = torch.cat([prefix_mask, target_mask], dim=1)

        labels = torch.cat(
            [
                torch.full(
                    (batch_size, self.num_latent_tokens),
                    -100,
                    device=target_ids.device,
                    dtype=target_ids.dtype,
                ),
                target_ids,
            ],
            dim=1,
        )
        labels = labels.masked_fill(full_mask == 0, -100)

        outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=full_mask, labels=labels)
        return {"loss": outputs.loss, "logits": outputs.logits}

    @torch.no_grad()
    def generate(self, s_y_hat, max_new_tokens=256, temperature=0.7, top_p=0.9, do_sample=True):
        self.eval()
        batch_size = s_y_hat.size(0)
        prefix_embeds = self._build_prefix(s_y_hat)

        bos_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        start_tokens = torch.full(
            (batch_size, 1), bos_id, device=s_y_hat.device, dtype=torch.long
        )
        start_embeds = self.model.transformer.wte(start_tokens)
        inputs_embeds = torch.cat([prefix_embeds, start_embeds], dim=1)

        generated = start_tokens
        past_key_values = None

        for step in range(max_new_tokens):
            if step == 0:
                outputs = self.model(inputs_embeds=inputs_embeds, use_cache=True)
            else:
                outputs = self.model(
                    input_ids=next_token, past_key_values=past_key_values, use_cache=True
                )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :] / max(temperature, 1e-8)

            if do_sample:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[mask] = float("-inf")
                probs = torch.softmax(sorted_logits, dim=-1)
                next_idx = torch.multinomial(probs, 1)
                next_token = sorted_indices.gather(-1, next_idx)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)
            if (next_token == self.tokenizer.eos_token_id).all():
                break

        return self.tokenizer.batch_decode(generated[:, 1:], skip_special_tokens=True)


DECODER_REGISTRY = {
    "scratch": LatentConditionedDecoder,
    "pretrained": PretrainedDecoder,
}


def build_decoder(decoder_type: str = "scratch", **kwargs) -> nn.Module:
    if decoder_type not in DECODER_REGISTRY:
        raise ValueError(f"Unknown decoder type: {decoder_type}")
    return DECODER_REGISTRY[decoder_type](**kwargs)
