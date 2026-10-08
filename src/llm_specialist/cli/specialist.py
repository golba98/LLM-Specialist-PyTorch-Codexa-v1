"""Prepare data, train heads, evaluate, or inspect the isolated specialist."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import sys
import time


from llm_specialist.artifacts import atomic_json, run_path
from llm_specialist.config import load_config


def build_parser() -> argparse.ArgumentParser:
    """Define explicit data and artifact inputs for every specialist action."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Validate, split, and cache a real labeled dataset.")
    prepare.add_argument("--config", type=Path, default=Path("configs/specialist.yaml"))
    prepare.add_argument("--dataset", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    train = commands.add_parser("train", help="Train frozen-embedding heads with an automatic Kitty viewer.")
    train.add_argument("--config", type=Path, default=Path("configs/specialist.yaml"))
    train.add_argument("--prepared-dir", type=Path, required=True)
    train.add_argument("--run-name", required=True)
    train.add_argument("--model", choices=("linear", "mlp", "both"), default="both")
    evaluate = commands.add_parser("evaluate", help="Compare selected heads on the held-out test partition.")
    evaluate.add_argument("--prepared-dir", type=Path, required=True)
    evaluate.add_argument("--run-name", required=True)
    evaluate.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    infer = commands.add_parser("infer", help="Predict using an existing trained specialist checkpoint.")
    infer.add_argument("--checkpoint", type=Path, required=True)
    infer.add_argument("--text", action="append", required=True)
    infer.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    challenge = commands.add_parser("challenge", help="Inspect additional reviewed ambiguity/OOD probes.")
    challenge.add_argument("--checkpoint", type=Path, required=True)
    challenge.add_argument("--dataset", type=Path, required=True)
    challenge.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    smoke = commands.add_parser("smoke-encoder", help="Verify real text-only encoder loading; no head training.")
    smoke.add_argument("--config", type=Path, default=Path("configs/specialist.yaml"))
    smoke.add_argument("--device", choices=("auto", "cpu", "cuda"))
    viewer = commands.add_parser("view", help="Internal viewer, opened automatically by training.")
    viewer.add_argument("--status", type=Path, required=True)
    return parser


def smoke_encoder(config_path: Path, device_name: str | None = None) -> dict:
    """Run a real, unlabeled encoder check with synchronized warmed timings."""
    import torch

    from llm_specialist.encoder import FrozenEncoder, deterministic_seed, resolve_device
    from llm_specialist.training import cuda_memory, synchronize

    config = load_config(config_path)
    settings = config.encoder if device_name is None else replace(config.encoder, device=device_name)
    device = resolve_device(settings.device)
    deterministic_seed(config.training.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    encoder = FrozenEncoder(settings)
    synchronize(device)
    initialization_seconds = time.perf_counter() - started
    texts = ["A text-only encoder smoke test.", "The input includes code: return value + 1."]
    first, counts = encoder.encode(texts)
    second, _ = encoder.encode(texts)
    if not torch.allclose(first, second, atol=1e-6, rtol=1e-5):
        raise ValueError("Repeated encoder inference is not deterministic within tolerance.")
    durations = []
    for _ in range(5):
        synchronize(device)
        started = time.perf_counter()
        encoder.encode(texts[:1])
        synchronize(device)
        durations.append((time.perf_counter() - started) * 1000)
    report = {
        "status": "passed", "encoder": encoder.metadata(),
        "embedding_shape": list(first.shape), "norms": first.norm(dim=1).tolist(),
        "token_counts": counts, "deterministic": True,
        "trainable_encoder_parameters": sum(p.numel() for p in encoder.model.parameters() if p.requires_grad),
        "loaded_parameters": sum(p.numel() for p in encoder.model.parameters()),
        "initialization_seconds": initialization_seconds,
        "initialization_scope": "model setup including Hub/cache waits and device transfer; not inference latency",
        "latency": {"batch_size": 1, "warmup_calls": 2, "measured_calls": 5, "median_ms": statistics.median(durations), "scope": "encoder including tokenization and CPU embedding transfer"},
        "memory": cuda_memory(device), "labeled_dataset_available": False,
        "task_accuracy": None, "note": "Unlabeled encoder smoke verification only; no classifier was trained.",
    }
    output = run_path(f"encoder-smoke-{time.time_ns()}", "logs") / "report.json"
    report["report_path"] = str(output)
    atomic_json(output, report)
    return report


def main() -> int:
    """Dispatch one specialist action without touching the generative pipeline."""
    arguments = build_parser().parse_args()
    if arguments.command == "prepare":
        from llm_specialist.data import prepare_dataset

        result = prepare_dataset(arguments.dataset, arguments.output_dir, load_config(arguments.config))
        result = {"prepared_dir": str(arguments.output_dir), "identity": result["identity"], "split_counts": {k: len(v) for k, v in result["splits"].items()}}
    elif arguments.command == "train":
        from llm_specialist.training import train_run

        result = train_run(arguments.prepared_dir, arguments.run_name, load_config(arguments.config), arguments.model)
    elif arguments.command == "evaluate":
        from llm_specialist.evaluation import evaluate_run

        result = evaluate_run(arguments.prepared_dir, arguments.run_name, arguments.device)
    elif arguments.command in ("infer", "challenge"):
        from llm_specialist.evaluation import SpecialistClassifier

        classifier = SpecialistClassifier(arguments.checkpoint, arguments.device)
        result = classifier.predict(arguments.text) if arguments.command == "infer" else classifier.challenge(arguments.dataset)
    elif arguments.command == "smoke-encoder":
        result = smoke_encoder(arguments.config, arguments.device)
    else:
        from llm_specialist.viewer import view_status

        view_status(arguments.status)
        return 0
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
