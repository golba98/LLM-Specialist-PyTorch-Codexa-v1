"""Typed, strict YAML settings for the specialist only."""

from dataclasses import asdict, dataclass, field, fields
import math
from pathlib import Path
from typing import Any

import yaml


LABELS = ("Debugging", "Refactoring", "Testing", "Implementation", "Explanation")
MODEL_ID = "google/embeddinggemma-2"
MODEL_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
EMBEDDING_DIM = 768
CLASSIFICATION_PREFIX = "task: classification | query: "


def positive_int(name: str, value: object, *, minimum: int = 1) -> None:
    """Reject booleans, fractional values, and integers below minimum."""
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}; got {value!r}.")


def finite_number(name: str, value: object, *, minimum: float = 0.0) -> float:
    """Validate a finite numeric setting."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number.")
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}.")
    return float(value)


def validate_device(value: str) -> None:
    """Limit device selection to supported CPU/CUDA modes."""
    if value not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda.")


@dataclass(frozen=True)
class EncoderConfig:
    """Pinned text-only encoder settings."""

    model_id: str = MODEL_ID
    revision: str = MODEL_REVISION
    precision: str = "auto"
    device: str = "auto"
    batch_size: int = 16
    max_tokens: int = 2048

    def __post_init__(self) -> None:
        if self.model_id != MODEL_ID:
            raise ValueError(f"Stage 1 requires {MODEL_ID}.")
        if not isinstance(self.revision, str) or len(self.revision) != 40:
            raise ValueError("revision must be a pinned 40-character commit SHA.")
        if any(c not in "0123456789abcdef" for c in self.revision):
            raise ValueError("revision must be a hexadecimal commit SHA.")
        if self.precision not in ("auto", "bf16", "fp32"):
            raise ValueError("encoder precision must be auto, bf16, or fp32; FP16 is forbidden.")
        validate_device(self.device)
        positive_int("encoder batch_size", self.batch_size)
        positive_int("max_tokens", self.max_tokens)
        if self.max_tokens > 8192:
            raise ValueError("max_tokens must not exceed the 8192-token model limit.")


@dataclass(frozen=True)
class HeadConfig:
    """Configurable MLP dimensions; the encoder dimension is fixed."""

    hidden_size: int = 256
    dropout: float = 0.1

    def __post_init__(self) -> None:
        positive_int("hidden_size", self.hidden_size)
        if finite_number("dropout", self.dropout) >= 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1.")


@dataclass(frozen=True)
class TrainingConfig:
    """Settings shared by both trainable heads."""

    seed: int = 42
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    max_epochs: int = 50
    patience: int = 5
    min_delta: float = 0.0001
    device: str = "auto"
    loss: str = "cross_entropy"
    class_weights: tuple[float, ...] | None = None
    label_smoothing: float = 0.0

    def __post_init__(self) -> None:
        positive_int("seed", self.seed, minimum=0)
        if self.seed >= 2**64:
            raise ValueError("seed must be smaller than 2**64 for PyTorch reproducibility.")
        for name in ("batch_size", "max_epochs", "patience"):
            positive_int(name, getattr(self, name))
        for name in ("learning_rate", "gradient_clip"):
            if finite_number(name, getattr(self, name)) == 0.0:
                raise ValueError(f"{name} must be positive.")
        for name in ("weight_decay", "min_delta", "label_smoothing"):
            finite_number(name, getattr(self, name))
        if self.label_smoothing >= 1.0:
            raise ValueError("label_smoothing must satisfy 0 <= value < 1.")
        if self.loss != "cross_entropy":
            raise ValueError("Stage 1 supports only cross_entropy loss.")
        validate_device(self.device)
        if self.class_weights is not None:
            if not isinstance(self.class_weights, (tuple, list)) or not self.class_weights:
                raise ValueError("class_weights must be a nonempty sequence or null.")
            for value in self.class_weights:
                if finite_number("class weight", value) == 0.0:
                    raise ValueError("class weights must be positive.")
            object.__setattr__(self, "class_weights", tuple(self.class_weights))


@dataclass(frozen=True)
class SpecialistConfig:
    """Complete independent classifier configuration."""

    labels: tuple[str, ...] = LABELS
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.labels, (tuple, list)) or len(self.labels) < 2:
            raise ValueError("labels must contain at least two distinct names.")
        if any(not isinstance(label, str) or not label.strip() for label in self.labels):
            raise ValueError("labels must contain nonempty strings.")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("labels must be distinct.")
        object.__setattr__(self, "labels", tuple(self.labels))
        weights = self.training.class_weights
        if weights is not None and len(weights) != len(self.labels):
            raise ValueError("class_weights must have one entry per label.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-compatible configuration."""
        result = asdict(self)
        result["labels"] = list(self.labels)
        if self.training.class_weights is not None:
            result["training"]["class_weights"] = list(self.training.class_weights)
        return result


def config_from_dict(value: object) -> SpecialistConfig:
    """Load serialized configuration with unknown-key rejection."""
    if not isinstance(value, dict):
        raise ValueError("specialist configuration must be a mapping.")
    expected = {"labels", "encoder", "head", "training"}
    if set(value) != expected:
        raise ValueError(f"specialist configuration requires exactly {sorted(expected)}.")
    objects: dict[str, Any] = {}
    for name, cls in (("encoder", EncoderConfig), ("head", HeadConfig), ("training", TrainingConfig)):
        section = value[name]
        if not isinstance(section, dict):
            raise ValueError(f"{name} must be a mapping.")
        unknown = set(section) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown {name} settings: {sorted(repr(key) for key in unknown)}.")
        objects[name] = cls(**section)
    return SpecialistConfig(labels=value["labels"], **objects)


def load_config(path: str | Path) -> SpecialistConfig:
    """Read a specialist YAML configuration."""
    try:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"Malformed specialist YAML: {error}") from error
    return config_from_dict(value)
