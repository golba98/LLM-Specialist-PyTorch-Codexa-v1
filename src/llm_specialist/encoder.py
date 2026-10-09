"""Lazy, frozen, text-only EmbeddingGemma 2 adapter."""

from dataclasses import asdict
import importlib.metadata
import os
import random
from typing import Any, Callable

import numpy as np
import torch

from llm_specialist.config import CLASSIFICATION_PREFIX, EMBEDDING_DIM, EncoderConfig


def resolve_device(requested: str) -> torch.device:
    """Choose CUDA when available, otherwise CPU, without silent CUDA fallback."""
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    return torch.device("cuda" if requested != "cpu" and torch.cuda.is_available() else "cpu")


def deterministic_seed(seed: int) -> None:
    """Seed independent head training and request deterministic operations."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def dependency_versions() -> dict[str, str]:
    """Return versions relevant to embedding reproducibility."""
    result: dict[str, str] = {}
    for name in ("torch", "transformers", "sentence-transformers", "tokenizers", "numpy"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not installed"
    return result


def encoder_runtime(config: EncoderConfig) -> tuple[torch.device, torch.dtype]:
    """Resolve allowed execution settings without loading model weights."""
    device = resolve_device(config.device)
    supported = device.type == "cuda" and torch.cuda.is_bf16_supported()
    if config.precision == "bf16" and not supported:
        raise ValueError("Explicit BF16 requires a CUDA device with native BF16 support.")
    dtype = torch.bfloat16 if config.precision != "fp32" and supported else torch.float32
    return device, dtype


def encoder_metadata(config: EncoderConfig) -> dict[str, Any]:
    """Compute an execution identity before loading an otherwise reusable cache."""
    device, dtype = encoder_runtime(config)
    return {
        "config": asdict(config), "prefix": CLASSIFICATION_PREFIX,
        "dimension": EMBEDDING_DIM,
        "precision": "bf16" if dtype == torch.bfloat16 else "fp32",
        "device_type": device.type, "versions": dependency_versions(),
    }


def validate_embeddings(value: torch.Tensor) -> torch.Tensor:
    """Require finite, nonzero, unit-length 768-dimensional float embeddings."""
    if value.ndim != 2 or value.shape[1] != EMBEDDING_DIM:
        raise ValueError(f"Embeddings must have shape [batch, {EMBEDDING_DIM}].")
    if not value.is_floating_point() or value.dtype == torch.float16 or not torch.isfinite(value).all():
        raise ValueError("Embeddings must be finite BF16/FP32 values; FP16 is forbidden.")
    result = value.detach().float().cpu()
    norms = torch.linalg.vector_norm(result, dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=2e-3, rtol=2e-3):
        raise ValueError("Embeddings must have unit L2 norm.")
    # BF16 normalization can round the norm; persist FP32 unit vectors.
    return torch.nn.functional.normalize(result, p=2, dim=1)


class FrozenEncoder:
    """Load only the text model and preserve its published embedding pipeline."""

    def __init__(
        self,
        config: EncoderConfig,
        *,
        factory: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config
        self.device, self.dtype = encoder_runtime(config)
        if factory is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise RuntimeError(
                    "Install requirements-specialist.txt in .venv-specialist and run its Python."
                ) from error
            factory = SentenceTransformer
        self.model = factory(
            config.model_id,
            revision=config.revision,
            device=str(self.device),
            config_kwargs={"vision_config": None, "audio_config": None},
            model_kwargs={"torch_dtype": self.dtype},
            trust_remote_code=False,
        )
        self.model.requires_grad_(False)
        self.model.eval()
        self.model.max_seq_length = config.max_tokens
        if self.model.get_embedding_dimension() != EMBEDDING_DIM:
            raise ValueError("Encoder does not produce 768-dimensional embeddings.")
        if self.model.prompts.get("Classification") != CLASSIFICATION_PREFIX:
            raise ValueError("Encoder Classification prompt differs from the approved prefix.")
        backbone = self.model[0].auto_model
        for name in ("vision_config", "audio_config"):
            if getattr(backbone.config, name, None) is not None:
                raise ValueError(f"Text-only encoder unexpectedly retains {name}.")
        for name in ("vision_tower", "audio_tower"):
            if getattr(backbone, name, None) is not None:
                raise ValueError(f"Text-only encoder unexpectedly loaded {name}.")
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise ValueError("Encoder weights must be frozen.")

    def metadata(self) -> dict[str, Any]:
        """Describe the exact encoder execution identity for cache/checkpoint use."""
        return encoder_metadata(self.config)

    def token_counts(self, texts: list[str], *, task: str = "Classification") -> list[int]:
        """Validate text and count all prefixed tokens without truncation."""
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Encoder inputs must be nonempty text strings.")
        if not texts:
            return []
        prefixes = {"Classification": CLASSIFICATION_PREFIX, "SearchQuery": "task: search result | query: ", "Document": "title: none | text: "}
        if task not in prefixes or self.model.prompts.get(task) != prefixes[task]:
            raise ValueError(f"Unsupported or incompatible encoder prompt: {task}.")
        encoded = self.model.tokenizer(
            [prefixes[task] + text.strip() for text in texts],
            truncation=False,
            padding=False,
            add_special_tokens=True,
        )
        counts = [len(ids) for ids in encoded["input_ids"]]
        for index, count in enumerate(counts):
            if count > self.config.max_tokens:
                raise ValueError(
                    f"Input {index} has {count} prefixed tokens; limit is {self.config.max_tokens}. "
                    "Shorten the request or increase max_tokens; silent truncation is disabled."
                )
        return counts

    def encode(self, texts: list[str], *, task: str = "Classification") -> tuple[torch.Tensor, list[int]]:
        """Return normalized FP32 CPU embeddings and actual prefixed token counts."""
        counts = self.token_counts(texts, task=task)
        if not texts:
            return torch.empty((0, EMBEDDING_DIM), dtype=torch.float32), counts
        with torch.inference_mode():
            values = self.model.encode(
                [text.strip() for text in texts],
                prompt_name=task,
                normalize_embeddings=True,
                convert_to_tensor=True,
                batch_size=self.config.batch_size,
                show_progress_bar=False,
            )
        return validate_embeddings(values), counts
