"""Head-only training with early stopping and isolated run artifacts."""

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader, TensorDataset

from llm_specialist.artifacts import atomic_json, identity, run_path, save_checkpoint
from llm_specialist.config import SpecialistConfig, config_from_dict
from llm_specialist.data import load_prepared
from llm_specialist.encoder import dependency_versions, deterministic_seed, resolve_device
from llm_specialist.models import ClassificationHead, loss_function
from llm_specialist.viewer import start_viewer


@dataclass
class EarlyStopping:
    """Select true best validation loss and independently track patience."""

    patience: int
    min_delta: float
    best_loss: float = math.inf
    patience_loss: float = math.inf
    bad_epochs: int = 0

    def update(self, loss: float) -> tuple[bool, bool]:
        """Return whether to save a new best checkpoint and whether to stop."""
        if not math.isfinite(loss):
            raise ValueError("Validation loss is not finite.")
        improved = loss < self.best_loss
        if improved:
            self.best_loss = loss
        if loss < self.patience_loss - self.min_delta:
            self.patience_loss = loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return improved, self.bad_epochs >= self.patience


def cuda_memory(device: torch.device) -> dict[str, int | None]:
    """Return process-local peak CUDA allocations, or null on CPU."""
    return {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
    }


def synchronize(device: torch.device) -> None:
    """Wait for CUDA work when measuring durations."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cpu_snapshot(value: Any) -> Any:
    """Detach checkpoint tensors so the prior head cannot retain CUDA allocations."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_snapshot(item) for item in value)
    return value


def _loss_sum(
    logits: torch.Tensor, targets: torch.Tensor, criterion: torch.nn.CrossEntropyLoss,
) -> tuple[torch.Tensor, torch.Tensor]:
    # CrossEntropyLoss's weighted mean divides by target weights, not batch size.
    numerator = torch.nn.functional.cross_entropy(
        logits, targets, weight=criterion.weight,
        label_smoothing=criterion.label_smoothing, reduction="sum",
    )
    denominator = (
        torch.tensor(float(len(targets)), device=targets.device)
        if criterion.weight is None else criterion.weight[targets].sum()
    )
    return numerator, denominator


def evaluate_loss(
    model: ClassificationHead,
    loader: DataLoader,
    criterion: torch.nn.CrossEntropyLoss,
    device: torch.device,
) -> float:
    """Calculate sample/weight-correct validation cross-entropy."""
    model.eval()
    numerator = 0.0
    denominator = 0.0
    with torch.inference_mode():
        for embeddings, labels, *_ in loader:
            logits = model(embeddings.to(device))
            value, weight = _loss_sum(logits, labels.to(device), criterion)
            numerator += float(value)
            denominator += float(weight)
    if denominator <= 0:
        raise ValueError("Validation dataset must not be empty.")
    loss = numerator / denominator
    if not math.isfinite(loss):
        raise ValueError("Validation loss is not finite.")
    return loss


def train_batch(
    model: ClassificationHead,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.CrossEntropyLoss,
    gradient_clip: float,
) -> tuple[float, float]:
    """Perform one tested head-only AdamW update with finite gradient clipping."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(embeddings)
    numerator, denominator = _loss_sum(logits, labels, criterion)
    loss = numerator / denominator
    if not torch.isfinite(loss):
        raise ValueError("Training loss is not finite.")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip, error_if_nonfinite=True)
    optimizer.step()
    return float(numerator.detach()), float(denominator.detach())


def _validate_training_config(manifest: dict[str, Any], config: SpecialistConfig) -> None:
    prepared_config = config_from_dict(manifest["config"])
    if config.labels != prepared_config.labels or config.encoder != prepared_config.encoder:
        raise ValueError("Training labels and encoder settings must match the prepared dataset.")
    if config.training.seed != prepared_config.training.seed:
        raise ValueError("Training seed must match the prepared split seed.")


def train_run(
    prepared_dir: Path,
    run_name: str,
    config: SpecialistConfig,
    kind: str = "both",
    *,
    viewer_factory: Callable[..., Any] = start_viewer,
) -> dict[str, Any]:
    """Train fresh independent heads; every launch must attach a visible viewer."""
    if kind not in ("linear", "mlp", "both"):
        raise ValueError("model must be linear, mlp, or both.")
    manifest, embeddings, token_counts = load_prepared(prepared_dir)
    _validate_training_config(manifest, config)
    checkpoint_root = run_path(run_name, "checkpoints")
    log_root = run_path(run_name, "logs")
    if checkpoint_root.exists() or log_root.exists():
        raise FileExistsError("Specialist run already exists; choose a new run name.")
    device = resolve_device(config.training.device)
    targets = torch.tensor([
        config.labels.index(record["label"]) if record["label"] in config.labels else -1
        for record in manifest["records"]
    ], dtype=torch.long)
    train_indices = torch.tensor(manifest["splits"]["train"], dtype=torch.long)
    validation_indices = torch.tensor(manifest["splits"]["validation"], dtype=torch.long)
    train_data = TensorDataset(embeddings[train_indices], targets[train_indices], token_counts[train_indices])
    validation_data = TensorDataset(embeddings[validation_indices], targets[validation_indices])
    checkpoint_root.mkdir(parents=True, exist_ok=False)
    log_root.mkdir(parents=True, exist_ok=False)
    metadata: dict[str, Any] = {
        "format": "specialist-run-v1", "run_name": run_name,
        "prepared_dir": str(prepared_dir.resolve()), "prepared_identity": manifest["identity"],
        "encoder_identity": manifest["encoder_identity"], "config": config.to_dict(),
        "versions": dependency_versions(), "device": str(device), "state": "STARTING", "heads": {},
        "throughput_definition": "prefixed tokens in cached requests presented per head update; encoder is not trained",
    }
    atomic_json(log_root / "run.json", metadata)
    for head_kind in (("linear", "mlp") if kind == "both" else (kind,)):
        head_logs = log_root / head_kind
        head_checkpoints = checkpoint_root / head_kind
        head_logs.mkdir()
        head_checkpoints.mkdir()
        status_path = head_logs / "progress.json"
        steps_per_epoch = math.ceil(len(train_data) / config.training.batch_size)
        status: dict[str, Any] = {
            "state": "STARTING", "kind": head_kind, "pid": os.getpid(),
            "step": 0, "total_steps": steps_per_epoch * config.training.max_epochs,
            "total_tokens_seen": 0, "tokens_per_second": 0.0,
            "training_loss": None, "validation_loss": None, "eta_seconds": 0,
            "checkpoint_path": str(head_checkpoints / "latest.pt"),
        }
        atomic_json(status_path, status)
        try:
            viewer = viewer_factory(status_path)
            metadata["state"] = "RUNNING"
            atomic_json(log_root / "run.json", metadata, replace=True)
            deterministic_seed(config.training.seed)
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            model = ClassificationHead(config, head_kind).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=config.training.learning_rate,
                weight_decay=config.training.weight_decay,
            )
            criterion = loss_function(config, device)
            loader = DataLoader(
                train_data, batch_size=config.training.batch_size, shuffle=True, num_workers=0,
                generator=torch.Generator().manual_seed(config.training.seed),
            )
            validation_loader = DataLoader(validation_data, batch_size=config.training.batch_size, num_workers=0)
            stopping = EarlyStopping(config.training.patience, config.training.min_delta)
            history: list[dict[str, Any]] = []
            elapsed_steps = 0.0
            with (head_logs / "train_metrics.jsonl").open("x", encoding="utf-8") as metrics:
                for epoch in range(1, config.training.max_epochs + 1):
                    total_numerator = 0.0
                    total_denominator = 0.0
                    for features, labels, counts in loader:
                        if viewer.poll() is not None:
                            raise RuntimeError("Visible training viewer closed; stopping head training.")
                        synchronize(device)
                        started = time.perf_counter()
                        numerator, denominator = train_batch(
                            model, features.to(device), labels.to(device), optimizer,
                            criterion, config.training.gradient_clip,
                        )
                        synchronize(device)
                        duration = time.perf_counter() - started
                        elapsed_steps += duration
                        total_numerator += numerator
                        total_denominator += denominator
                        status["step"] += 1
                        status["state"] = "RUNNING"
                        status["total_tokens_seen"] += int(counts.sum())
                        status["tokens_per_second"] = int(counts.sum()) / max(duration, 1e-9)
                        status["training_loss"] = numerator / denominator
                        status["eta_seconds"] = elapsed_steps / status["step"] * (status["total_steps"] - status["step"])
                        record = {
                            **status, "epoch": epoch, "step_time_seconds": duration,
                            **cuda_memory(device),
                        }
                        metrics.write(json.dumps(record, allow_nan=False) + "\n")
                        metrics.flush()
                        atomic_json(status_path, status, replace=True)
                    validation_loss = evaluate_loss(model, validation_loader, criterion, device)
                    status["validation_loss"] = validation_loss
                    history.append({
                        "epoch": epoch, "step": status["step"],
                        "training_loss": total_numerator / total_denominator,
                        "validation_loss": validation_loss,
                    })
                    improved, stop = stopping.update(validation_loss)
                    payload = {
                        "format": "specialist-v1", "kind": head_kind,
                        "config": config.to_dict(), "epoch": epoch, "step": status["step"],
                        "model_state": _cpu_snapshot(model.state_dict()),
                        "optimizer_state": _cpu_snapshot(optimizer.state_dict()),
                        "rng_state": torch.get_rng_state(),
                        "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                        "shuffle_rng_state": loader.generator.get_state(),
                        "early_stopping": {"best_loss": stopping.best_loss, "patience_loss": stopping.patience_loss, "bad_epochs": stopping.bad_epochs},
                        "prepared_identity": manifest["identity"],
                        "encoder_identity": manifest["encoder_identity"],
                        "encoder_metadata": manifest["encoder"], "history": history.copy(),
                        "versions": dependency_versions(), "training_device": str(device),
                    }
                    save_checkpoint(head_checkpoints / "latest.pt", payload)
                    if improved:
                        save_checkpoint(head_checkpoints / "best.pt", payload)
                    atomic_json(head_logs / "history.json", history, replace=True)
                    atomic_json(status_path, status, replace=True)
                    if stop:
                        break
            status["state"] = "EARLY_STOPPED" if stop else "COMPLETED"
            status["eta_seconds"] = 0
            atomic_json(status_path, status, replace=True)
            metadata["heads"][head_kind] = {
                "state": status["state"], "epochs": epoch, "steps": status["step"],
                "best_validation_loss": stopping.best_loss, "history": history,
                **cuda_memory(device),
            }
            atomic_json(log_root / "run.json", metadata, replace=True)
            del model, optimizer, criterion
        except BaseException as error:
            state = "INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED"
            status.update(state=state, error=str(error), eta_seconds=0)
            metadata.update(state=state, error=str(error))
            atomic_json(status_path, status, replace=True)
            atomic_json(log_root / "run.json", metadata, replace=True)
            raise
    metadata["state"] = "COMPLETED"
    metadata["identity"] = identity({key: value for key, value in metadata.items() if key != "identity"})
    atomic_json(log_root / "run.json", metadata, replace=True)
    return metadata
