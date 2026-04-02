"""
V2 Decoder: Pretrained LM conditioned on a SEQUENCE of latent chunks.

Key changes from v1:
  1. Uses a pretrained decoder (TinyLlama, Phi-2, Qwen2.5) that already
     knows language, so it can focus on learning the latent→text mapping.
  2. Conditions on a sequence of latent chunks (not a single vector),
     giving the decoder much richer information about the solution.
  3. Latent chunks are projected to match the LM's hidden dim, then
     prepended as a "soft prompt" the decoder cross-attends to.
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class ChunkedPretrainedDecoder(nn.Module):
    """
    Pretrained causal LM conditioned on a sequence of latent chunks.

    The latent chunks are projected into the LM's embedding space and
    prepended as prefix tokens. The LM then generates conditioned on
    both the latent prefix and any text tokens.
    """

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        latent_dim: int = 768,
        num_chunks: int = 8,
        freeze_backbone: bool = True,
        train_lora: bool = True,
        lora_r: int = 16,
    ):
        super().__init__()
        self.num_chunks = num_chunks
        self.latent_dim = latent_dim
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32
        )
        self.hidden_dim = self.model.config.hidden_size

        self.chunk_proj = nn.Sequential(
            nn.Linear(latent_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

            if train_lora:
                self._add_lora(lora_r)

    def _add_lora(self, r: int = 16):
        """Add simple LoRA adapters to attention layers."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and any(
                k in name for k in ["q_proj", "v_proj", "k_proj"]
            ):
                module.weight.requires_grad = False
                in_f, out_f = module.in_features, module.out_features
                lora_A = nn.Linear(in_f, r, bias=False)
                lora_B = nn.Linear(r, out_f, bias=False)
                nn.init.kaiming_uniform_(lora_A.weight)
                nn.init.zeros_(lora_B.weight)

                original_forward = module.forward
                def make_lora_forward(orig, A, B):
                    def forward(x):
                        return orig(x) + B(A(x))
                    return forward

                module.forward = make_lora_forward(original_forward, lora_A, lora_B)
                self._lora_params = getattr(self, "_lora_params", [])
                self._lora_params.extend([lora_A, lora_B])

        if hasattr(self, "_lora_params"):
            self.lora_modules = nn.ModuleList(self._lora_params)

    def _get_embed_fn(self):
        """Get the token embedding function from the model."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
            return self.model.model.embed_tokens
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "wte"):
            return self.model.transformer.wte
        else:
            raise RuntimeError(f"Cannot find embedding layer in {type(self.model)}")

    def forward(
        self,
        latent_chunks: torch.Tensor,
        target_ids: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> dict:
        """
        Training forward.

        Args:
            latent_chunks: [B, num_chunks, latent_dim]
            target_ids:    [B, seq_len] solution tokens
            target_mask:   [B, seq_len]
        """
        batch_size = latent_chunks.size(0)

        prefix_embeds = self.chunk_proj(latent_chunks)  # [B, K, hidden]

        embed_fn = self._get_embed_fn()
        token_embeds = embed_fn(target_ids)  # [B, seq, hidden]

        inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)

        prefix_mask = torch.ones(
            batch_size, self.num_chunks, device=target_ids.device, dtype=target_mask.dtype
        )
        full_mask = torch.cat([prefix_mask, target_mask], dim=1)

        # Labels: -100 for prefix positions (no loss), then target tokens
        labels = torch.cat([
            torch.full((batch_size, self.num_chunks), -100, device=target_ids.device, dtype=target_ids.dtype),
            target_ids,
        ], dim=1)
        labels = labels.masked_fill(full_mask == 0, -100)

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            labels=labels,
        )

        return {"loss": outputs.loss, "logits": outputs.logits}

    @torch.no_grad()
    def generate(
        self,
        latent_chunks: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> list[str]:
        """Generate text from latent chunks."""
        self.eval()
        batch_size = latent_chunks.size(0)

        prefix_embeds = self.chunk_proj(latent_chunks)  # [B, K, hidden]

        embed_fn = self._get_embed_fn()
        bos_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        start_tokens = torch.full((batch_size, 1), bos_id, device=latent_chunks.device, dtype=torch.long)
        start_embeds = embed_fn(start_tokens)

        inputs_embeds = torch.cat([prefix_embeds, start_embeds], dim=1)

        generated = start_tokens
        past_key_values = None

        for step in range(max_new_tokens):
            if step == 0:
                outputs = self.model(inputs_embeds=inputs_embeds, use_cache=True)
            else:
                outputs = self.model(
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

        return self.tokenizer.batch_decode(generated[:, 1:], skip_special_tokens=True)


class ChunkedSmallDecoder(nn.Module):
    """
    Lightweight alternative: small transformer decoder (no pretrained LM)
    that attends to a sequence of latent chunks via cross-attention.

    Use when you don't want the weight of TinyLlama/Phi-2.
    """

    def __init__(
        self,
        latent_dim: int = 768,
        vocab_size: int = 32000,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        num_chunks: int = 8,
        max_seq_len: int = 512,
        tokenizer_name: str = "gpt2",
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_chunks = num_chunks

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        vocab_size = len(self.tokenizer)

        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(max_seq_len, hidden_dim)

        self.chunk_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, latent_chunks, target_ids, target_mask):
        batch_size, seq_len = target_ids.shape

        memory = self.chunk_proj(latent_chunks)  # [B, K, hidden]

        positions = torch.arange(seq_len, device=target_ids.device).unsqueeze(0)
        tgt = self.token_embed(target_ids) + self.pos_embed(positions)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=target_ids.device)
        tgt_key_padding_mask = (target_mask == 0)

        output = self.transformer(
            tgt=tgt,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        output = self.norm(output)
        logits = self.lm_head(output)

        # Shift for next-token prediction
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = target_ids[:, 1:].contiguous()
        shift_labels = shift_labels.masked_fill(target_mask[:, 1:] == 0, -100)

        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def generate(self, latent_chunks, max_new_tokens=256, temperature=0.7, top_p=0.9, do_sample=True):
        self.eval()
        batch_size = latent_chunks.size(0)
        device = latent_chunks.device

        memory = self.chunk_proj(latent_chunks)

        bos_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        generated = torch.full((batch_size, 1), bos_id, device=device, dtype=torch.long)

        for step in range(max_new_tokens):
            seq_len = generated.size(1)
            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            tgt = self.token_embed(generated) + self.pos_embed(positions)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)

            output = self.transformer(tgt=tgt, memory=memory, tgt_mask=causal_mask)
            output = self.norm(output)
            logits = self.lm_head(output[:, -1, :]) / max(temperature, 1e-8)

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
