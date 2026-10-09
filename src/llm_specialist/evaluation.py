"""Held-out comparison, challenge probes, and trained classifier inference."""

from dataclasses import replace
import json
from pathlib import Path
import statistics
import time
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

from llm_specialist.artifacts import atomic_json, load_checkpoint, run_path
from llm_specialist.config import SpecialistConfig, config_from_dict
from llm_specialist.data import load_prepared, read_records
from llm_specialist.encoder import FrozenEncoder, deterministic_seed, resolve_device
from llm_specialist.models import ClassificationHead, loss_function
from llm_specialist.training import cuda_memory, evaluate_loss, synchronize


def classification_metrics(
    targets: list[int], predictions: list[int], labels: tuple[str, ...],
) -> dict[str, Any]:
    """Compute explicit full-label metrics, including zero-support classes."""
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

    if not targets or len(targets) != len(predictions):
        raise ValueError("Metrics require equally sized nonempty targets and predictions.")
    if any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < len(labels) for value in targets + predictions):
        raise ValueError("Metric class indices are invalid.")
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, labels=list(range(len(labels))), zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1.mean()),
        "per_class": {
            label: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])}
            for i, label in enumerate(labels)
        },
        "confusion_matrix": confusion_matrix(targets, predictions, labels=list(range(len(labels)))).tolist(),
        "confusion_matrix_axes": {"rows": "actual", "columns": "predicted", "labels": list(labels)},
    }


def load_head(checkpoint_path: Path, device: torch.device) -> tuple[ClassificationHead, dict, SpecialistConfig]:
    """Load only verified specialist head weights and their class mapping."""
    payload = load_checkpoint(checkpoint_path)
    config = config_from_dict(payload["config"])
    model = ClassificationHead(config, payload["kind"]).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError("Checkpoint contains nonfinite head parameters.")
    model.eval()
    return model, payload, config


def predict_scores(model: ClassificationHead, embeddings: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Return deterministic uncalibrated softmax scores from an evaluation head."""
    model.eval()
    with torch.inference_mode():
        values = torch.softmax(model(embeddings.to(device)), dim=-1).cpu()
    if not torch.isfinite(values).all():
        raise ValueError("Classifier produced nonfinite scores.")
    return values


def score_records(scores: torch.Tensor, records: list[dict], labels: tuple[str, ...]) -> list[dict]:
    """Associate predictions with provenance, explicitly identifying raw scores."""
    return [
        {
            "id": record["id"], "source": record["source"], "status": record["status"],
            "predicted_label": labels[int(row.argmax())],
            "scores": {label: float(row[i]) for i, label in enumerate(labels)},
            "scores_calibrated": False,
        }
        for row, record in zip(scores, records, strict=True)
    ]


def benchmark_head(model: ClassificationHead, embedding: torch.Tensor, device: torch.device) -> dict[str, Any]:
    """Measure warmed batch-size-one head latency independently of the encoder."""
    durations: list[float] = []
    with torch.inference_mode():
        features = embedding[:1].to(device)
        for _ in range(3):
            model(features)
        synchronize(device)
        for _ in range(20):
            synchronize(device)
            started = time.perf_counter()
            torch.softmax(model(features), dim=-1)
            synchronize(device)
            durations.append((time.perf_counter() - started) * 1000)
    return {
        "batch_size": 1, "warmup_iterations": 3, "measured_iterations": 20,
        "median_ms": statistics.median(durations), "min_ms": min(durations),
        "scope": "head forward and softmax; excludes encoder, loading, and host transfers",
    }


def evaluate_run(prepared_dir: Path, run_name: str, device_name: str = "auto") -> dict[str, Any]:
    """Compare available validation-selected heads on one untouched test split."""
    manifest, embeddings, _ = load_prepared(prepared_dir)
    checkpoint_root = run_path(run_name, "checkpoints")
    log_root = run_path(run_name, "logs")
    run = json.loads((log_root / "run.json").read_text(encoding="utf-8"))
    if run.get("state") != "COMPLETED" or run.get("prepared_identity") != manifest["identity"]:
        raise ValueError("Evaluation requires a completed run for this exact prepared dataset.")
    device = resolve_device(device_name)
    test_indices = manifest["splits"]["test"]
    records = [manifest["records"][i] for i in test_indices]
    test_features = embeddings[test_indices]
    report: dict[str, Any] = {
        "run_name": run_name, "prepared_identity": manifest["identity"],
        "test_records": len(records), "device": str(device), "heads": {},
        "encoder_latency": None, "encoder_memory": None,
        "measurement_note": "Encoder is absent during cached-head evaluation; use smoke-encoder for separate encoder measurements.",
        "limitations": ["Scores are uncalibrated.", "Five-class prediction cannot reliably detect out-of-domain inputs."],
    }
    for kind in ("linear", "mlp"):
        if kind not in run["heads"]:
            continue
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        model, checkpoint, config = load_head(checkpoint_root / kind / "best.pt", device)
        if checkpoint["prepared_identity"] != manifest["identity"] or checkpoint["encoder_identity"] != manifest["encoder_identity"]:
            raise ValueError("Checkpoint belongs to a different prepared dataset or encoder.")
        if config.labels != tuple(manifest["config"]["labels"]):
            raise ValueError("Checkpoint label order differs from the prepared dataset.")
        targets = [config.labels.index(record["label"]) for record in records]
        chunks = [
            predict_scores(model, test_features[start : start + config.training.batch_size], device)
            for start in range(0, len(records), config.training.batch_size)
        ]
        scores = torch.cat(chunks)
        predictions = scores.argmax(dim=1).tolist()
        result = classification_metrics(targets, predictions, config.labels)
        criterion = loss_function(config, device)
        loader = DataLoader(TensorDataset(test_features, torch.tensor(targets)), batch_size=config.training.batch_size)
        result["test_loss"] = evaluate_loss(model, loader, criterion, device)
        result["selected_epoch"] = checkpoint["epoch"]
        result["loss_history"] = run["heads"][kind]["history"]
        result["training_memory"] = {
            key: run["heads"][kind].get(key) for key in ("peak_allocated_bytes", "peak_reserved_bytes")
        }
        result["latency"] = benchmark_head(model, test_features, device)
        result["inference_memory"] = cuda_memory(device)
        details = score_records(scores, records, config.labels)
        for detail, target in zip(details, targets, strict=True):
            detail["actual_label"] = config.labels[target]
        result["misclassifications"] = [detail for detail in details if detail["actual_label"] != detail["predicted_label"]]
        challenge = manifest["splits"]["challenge"]
        result["challenge_predictions"] = []
        for start in range(0, len(challenge), config.training.batch_size):
            indices = challenge[start : start + config.training.batch_size]
            result["challenge_predictions"].extend(score_records(
                predict_scores(model, embeddings[indices], device),
                [manifest["records"][i] for i in indices], config.labels,
            ))
        result["challenge_note"] = "No challenge data available." if not challenge else "Review/OOD predictions have no invented single-label accuracy."
        report["heads"][kind] = result
        del model, criterion
    if not report["heads"]:
        raise ValueError("No trained specialist heads found.")
    # Keep comparisons descriptive; do not promote a head by test-set performance.
    if set(report["heads"]) == {"linear", "mlp"}:
        report["mlp_minus_linear"] = {
            metric: report["heads"]["mlp"][metric] - report["heads"]["linear"][metric]
            for metric in ("accuracy", "macro_f1")
        }
    # Repeated evaluation creates a new report, never overwriting an earlier result.
    stamp = time.time_ns()
    output_path = log_root / f"evaluation-{stamp}.json"
    atomic_json(output_path, report)
    report["report_path"] = str(output_path)
    return report


class SpecialistClassifier:
    """Text inference that requires a trained, checksummed specialist head."""

    def __init__(self, checkpoint_path: Path, device_name: str = "auto") -> None:
        self.device = resolve_device(device_name)
        self.model, self.checkpoint, self.config = load_head(checkpoint_path, self.device)
        deterministic_seed(self.config.training.seed)
        encoder_config = replace(self.config.encoder, device=device_name)
        # Portable inference uses FP32 on CPU even when the training cache was BF16.
        if self.device.type == "cpu":
            encoder_config = replace(encoder_config, precision="fp32")
        self.encoder = FrozenEncoder(encoder_config)

    def predict(self, texts: list[str]) -> list[dict[str, Any]]:
        """Return five-class predictions and measured latency, without OOD claims."""
        synchronize(self.device)
        started = time.perf_counter()
        embeddings, _ = self.encoder.encode(texts)
        synchronize(self.device)
        encoded_at = time.perf_counter()
        scores = predict_scores(self.model, embeddings, self.device)
        synchronize(self.device)
        finished = time.perf_counter()
        return [
            {
                "predicted_label": self.config.labels[int(row.argmax())],
                "scores": {label: float(row[i]) for i, label in enumerate(self.config.labels)},
                "scores_calibrated": False,
                "encoder_identity": self.checkpoint["encoder_identity"],
                "runtime_encoder": self.encoder.metadata(),
                "batch_latency_ms": {
                    "encoder": (encoded_at - started) * 1000,
                    "head_including_transfer": (finished - encoded_at) * 1000,
                    "end_to_end": (finished - started) * 1000,
                    "batch_size": len(texts), "warmed": False,
                },
            }
            for row in scores
        ]

    def challenge(self, dataset: Path) -> list[dict[str, Any]]:
        """Probe review/OOD records without assigning fabricated gold labels."""
        records = read_records(dataset, self.config.labels)
        if any(record.status == "labeled" for record in records):
            raise ValueError("Challenge data must contain review/OOD records only.")
        predictions = self.predict([record.text for record in records])
        return [dict(result, id=record.id, status=record.status) for result, record in zip(predictions, records, strict=True)]
