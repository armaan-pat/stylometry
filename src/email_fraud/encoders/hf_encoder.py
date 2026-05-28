"""HuggingFace AutoModel encoder with optional LoRA and three pooling modes.

pooling: "mean" (default), "cls", or "luar_episode".
luar_episode pooling expects (B, K, L) input; see Rivera et al. EMNLP 2021 (arXiv:2107.10882).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from email_fraud.config import EncoderConfig
from email_fraud.encoders.base import BaseEncoder

# @register("encoder", "hf") — Rivera et al. LUAR (arXiv:2107.10882) and
# any AutoModel-compatible checkpoint (RoBERTa, ModernBERT, CANINE, etc.)
from email_fraud.registry import register


class _LUARPeftAdapter(nn.Module):
    """Absorbs the inputs_embeds=None that PEFT injects before LUAR's forward.

    LUAR.forward() only accepts (input_ids, attention_mask, ...) — PEFT's
    PeftModelForFeatureExtraction unconditionally passes inputs_embeds=None.
    This adapter swallows that kwarg. LoRA target modules are still found
    because PEFT scans named_modules() which reaches through _luar.
    """

    def __init__(self, luar_model: nn.Module) -> None:
        super().__init__()
        self._luar = luar_model
        self.config = luar_model.config

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        return self._luar(input_ids=input_ids, attention_mask=attention_mask)


def _patch_luar_meta_device() -> None:
    # transformers 5.x builds models in a meta-device context, but LUAR's __init__
    # does a nested AutoModel.from_pretrained that can't materialize on meta.
    # Force the nested load to run under "cpu" instead. One-time, no runtime cost.
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    LUAR = get_class_from_dynamic_module("model.LUAR", "rrivera1849/LUAR-MUD")
    if getattr(LUAR, "_meta_device_patched", False):
        return
    original = LUAR.create_transformer

    def create_transformer(self, revision=None):  # type: ignore[no-untyped-def]
        with torch.device("cpu"):
            return original(self, revision=revision)

    LUAR.create_transformer = create_transformer

    # transformers 5.x reads `all_tied_weights_keys` (dict of missing→source) during
    # _finalize_model_loading. LUAR has no tied weights; an empty dict is correct.
    if not hasattr(LUAR, "all_tied_weights_keys"):
        LUAR.all_tied_weights_keys = {}

    LUAR._meta_device_patched = True


@register("encoder", "hf")
class HFEncoder(BaseEncoder):
    """HuggingFace AutoModel encoder with optional LoRA and mean/cls/luar_episode pooling."""

    MODEL_TYPE = "subword"

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path, trust_remote_code=True
        )

        if config.pooling == "luar_episode":
            _patch_luar_meta_device()

        backbone = AutoModel.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
        )
        # transformers 5.x may leave token_type_ids buffers uninitialized after
        # meta→device promotion; re-zero them for correct RoBERTa-family inference.
        for module in backbone.modules():
            if hasattr(module, "token_type_ids") and isinstance(module.token_type_ids, torch.Tensor):
                module.token_type_ids.zero_()

        if config.lora is not None:
            from peft import LoraConfig, TaskType, get_peft_model

            # Wrap LUAR before get_peft_model so the adapter absorbs inputs_embeds.
            if config.pooling == "luar_episode":
                backbone = _LUARPeftAdapter(backbone)

            lora_cfg = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=config.lora.r,
                lora_alpha=config.lora.alpha,
                target_modules=config.lora.target_modules,
                lora_dropout=config.lora.dropout,
                bias="none",
            )
            backbone = get_peft_model(backbone, lora_cfg)

        if config.freeze_backbone and config.lora is None:
            for param in backbone.parameters():
                param.requires_grad = False

        self.backbone = backbone
        # LUAR uses embedding_size; standard transformers use hidden_size.
        backbone_dim: int = getattr(backbone.config, "hidden_size", None) or getattr(backbone.config, "embedding_size")

        if config.projection_dim is not None:
            self.projection: nn.Linear | None = nn.Linear(backbone_dim, config.projection_dim, bias=False)
            self._embedding_dim = config.projection_dim
        else:
            self.projection = None
            self._embedding_dim = backbone_dim

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def episode_k(self) -> int | None:
        """Number of emails per LUAR episode; None for non-episode-pooling encoders."""
        if self.config.pooling == "luar_episode":
            return self.config.episode_k
        return None

    def tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        """Return (B, d) L2-normalized embeddings. input_ids is (B, K, L) for luar_episode."""
        pooling = self.config.pooling

        if pooling == "luar_episode":
            if self.config.episode_k is None:
                raise ValueError(
                    "encoder.episode_k must be set when pooling='luar_episode'. "
                    "Set it to the number of emails per episode in your experiment config."
                )
            episode_k = self.config.episode_k
            n_total = input_ids.size(0)
            if n_total % episode_k != 0:
                raise ValueError(
                    f"Batch size {n_total} is not divisible by episode_k={episode_k}. "
                    "Ensure batch_size // emails_per_sender_k == P and "
                    "emails_per_sender_k // episode_k is a whole number."
                )
            n_episodes = n_total // episode_k
            # PKSampler lays out K contiguous emails per sender, so consecutive
            # episode_k rows always belong to the same sender.
            input_ids = input_ids.view(n_episodes, episode_k, -1)
            attention_mask = attention_mask.view(n_episodes, episode_k, -1)
            return self._encode_luar_episode(input_ids, attention_mask, **kwargs)

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        last_hidden = outputs.last_hidden_state  # (B, L, d)

        if pooling == "cls":
            pooled = last_hidden[:, 0, :]
        elif pooling == "mean":
            pooled = self._mean_pool(last_hidden, attention_mask)
        else:
            raise ValueError(
                f"Unknown pooling strategy '{pooling}'. "
                "Choose from: 'mean', 'cls', 'luar_episode'."
            )

        if self.projection is not None:
            pooled = self.projection(pooled)

        return F.normalize(pooled, p=2, dim=-1)

    @staticmethod
    def _mean_pool(
        last_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Attention-masked mean pooling; excludes padding tokens from the average."""
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def _encode_luar_episode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        """LUAR episode pooling: (B, K, L) → (B, d)."""
        episode_embs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        if self.projection is not None:
            episode_embs = self.projection(episode_embs)
        return F.normalize(episode_embs, p=2, dim=-1)
