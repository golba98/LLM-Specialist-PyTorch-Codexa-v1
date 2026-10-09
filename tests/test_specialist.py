"""Offline specialist tests with schema-only fixtures and synthetic tensors.

Nothing in this module is a programming-intent training corpus or task result.
"""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest
import torch
from torch import nn

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_specialist import artifacts, data
from llm_specialist.artifacts import atomic_json, file_sha256, identity, load_checkpoint, save_checkpoint
from llm_specialist.config import (
    CLASSIFICATION_PREFIX, EMBEDDING_DIM, EncoderConfig, HeadConfig,
    SpecialistConfig, TrainingConfig, config_from_dict, load_config,
)
from llm_specialist.data import RequestRecord, load_prepared, read_records, split_records
from llm_specialist.encoder import FrozenEncoder, deterministic_seed, validate_embeddings
from llm_specialist.evaluation import classification_metrics, load_head, predict_scores
from llm_specialist.models import ClassificationHead, loss_function
from llm_specialist.training import EarlyStopping, evaluate_loss, train_batch, train_run
from llm_specialist.viewer import render_progress, start_viewer


class FakeSentenceTransformer(nn.Module):
    """Tiny injected module that tests adapter contracts without a download."""

    def __init__(self, model_id: str, **kwargs) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.arguments = {"model_id": model_id, **kwargs}
        self.prompts = {"Classification": CLASSIFICATION_PREFIX}
        self.backbone = SimpleNamespace(
            config=SimpleNamespace(vision_config=None, audio_config=None),
            vision_tower=None, audio_tower=None,
        )
        self.last_encode = None

    def __getitem__(self, index: int):
        return SimpleNamespace(auto_model=self.backbone)

    def get_embedding_dimension(self) -> int:
        return EMBEDDING_DIM

    def tokenizer(self, texts: list[str], **kwargs) -> dict:
        assert kwargs["truncation"] is False
        return {"input_ids": [[0] + list(range(len(text.split()))) + [1] for text in texts]}

    def encode(self, texts: list[str], **kwargs) -> torch.Tensor:
        self.last_encode = (texts, kwargs)
        values = torch.zeros((len(texts), EMBEDDING_DIM))
        for i, text in enumerate(texts):
            column = int.from_bytes(hashlib.sha256(text.encode()).digest()[:2]) % EMBEDDING_DIM
            values[i, column] = 1.0
        return values


def cpu_config(**settings) -> SpecialistConfig:
    return SpecialistConfig(
        encoder=EncoderConfig(device="cpu", precision="fp32"),
        training=TrainingConfig(device="cpu", **settings),
    )


def records_fixture() -> list[RequestRecord]:
    config = cpu_config()
    return [
        RequestRecord(
            id=f"fixture-{label_index}-{index}", text=f"schema-only-{label_index}-{index}",
            source="unit-test-only", label=label, group_id=f"group-{label_index}-{index}",
        )
        for label_index, label in enumerate(config.labels) for index in range(20)
    ]


def prepared_fixture(root: Path, config: SpecialistConfig) -> Path:
    """Write an isolated synthetic bundle for mechanical training tests only."""
    pytest.importorskip("sklearn")
    from dataclasses import asdict

    records = records_fixture()
    assignments = split_records(records, config)
    values = torch.nn.functional.normalize(
        torch.randn((len(records), EMBEDDING_DIM), generator=torch.Generator().manual_seed(12)), dim=1,
    )
    counts = [10] * len(records)
    root.mkdir()
    artifacts.atomic_bytes(root / "embeddings.npz", data._npz_bytes(values, counts))
    metadata = {"config": config.to_dict()["encoder"], "unit_test_only": True}
    manifest = {
        "format": "specialist-prepared-v1", "config": config.to_dict(),
        "records": [asdict(record) for record in records], "splits": assignments,
        "encoder": metadata, "encoder_identity": identity(metadata),
        "dataset_sha256": "fixture", "cache_identity": "fixture",
        "embeddings_sha256": file_sha256(root / "embeddings.npz"),
    }
    manifest["identity"] = identity(manifest)
    atomic_json(root / "manifest.json", manifest)
    return root


def test_specialist_config_loading_and_isolation() -> None:
    config = load_config("configs/specialist.yaml")
    assert config.head.hidden_size == 256 and config.head.dropout == 0.1
    assert config.encoder.max_tokens == 2048
    assert config_from_dict(config.to_dict()) == config
    changed = config.to_dict()
    changed["head"]["unknown"] = 1
    with pytest.raises(ValueError, match="Unknown head"):
        config_from_dict(changed)


@pytest.mark.parametrize("kwargs", [
    {"precision": "fp16"}, {"max_tokens": 8193}, {"batch_size": False},
    {"revision": "main"}, {"model_id": "google/embeddinggemma-300m"}, {"device": "mps"},
])
def test_encoder_config_rejects_invalid_settings(kwargs) -> None:
    with pytest.raises(ValueError):
        EncoderConfig(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"learning_rate": 0}, {"weight_decay": float("nan")}, {"seed": True},
    {"gradient_clip": -1}, {"label_smoothing": 1}, {"loss": "mse"}, {"class_weights": [0]},
])
def test_training_config_rejects_invalid_settings(kwargs) -> None:
    with pytest.raises(ValueError):
        TrainingConfig(**kwargs)
    with pytest.raises(ValueError):
        HeadConfig(dropout=1.0)


def test_encoder_initialization_and_embedding_contract() -> None:
    encoder = FrozenEncoder(EncoderConfig(device="cpu"), factory=FakeSentenceTransformer)
    model = encoder.model
    assert model.arguments["config_kwargs"] == {"vision_config": None, "audio_config": None}
    assert model.arguments["model_kwargs"]["torch_dtype"] == torch.float32
    assert model.arguments["trust_remote_code"] is False
    assert not model.training and not model.weight.requires_grad
    values, counts = encoder.encode(["unlabeled fixture", "second fixture"])
    assert values.shape == (2, 768) and values.dtype == torch.float32
    assert torch.equal(values.norm(dim=1), torch.ones(2))
    assert all(count > 0 for count in counts)
    assert model.last_encode[1]["prompt_name"] == "Classification"
    assert not model.last_encode[0][0].startswith(CLASSIFICATION_PREFIX)
    repeated, _ = encoder.encode(["unlabeled fixture", "second fixture"])
    assert torch.equal(values, repeated)
    empty, counts = encoder.encode([])
    assert empty.shape == (0, 768) and counts == []


def test_encoder_rejects_bad_inputs_and_modalities() -> None:
    encoder = FrozenEncoder(EncoderConfig(device="cpu", max_tokens=10), factory=FakeSentenceTransformer)
    with pytest.raises(ValueError, match="nonempty"):
        encoder.encode([""])
    with pytest.raises(ValueError, match="silent truncation"):
        encoder.encode(["word " * 20])
    with pytest.raises(ValueError, match="BF16"):
        FrozenEncoder(EncoderConfig(device="cpu", precision="bf16"), factory=FakeSentenceTransformer)

    def unwanted_modality(*args, **kwargs):
        model = FakeSentenceTransformer(*args, **kwargs)
        model.backbone.vision_tower = nn.Linear(1, 1)
        return model

    with pytest.raises(ValueError, match="vision_tower"):
        FrozenEncoder(EncoderConfig(device="cpu"), factory=unwanted_modality)


def test_embedding_validation_rejects_bad_outputs() -> None:
    for value in [
        torch.zeros((2, 768)), torch.ones((1, 256)),
        torch.full((1, 768), float("nan")),
        torch.nn.functional.normalize(torch.ones((1, 768)), dim=1).half(),
    ]:
        with pytest.raises(ValueError):
            validate_embeddings(value)


@pytest.mark.parametrize("kind", ["linear", "mlp"])
def test_head_forward_loss_gradient_and_determinism(kind: str) -> None:
    config = cpu_config()
    deterministic_seed(42)
    head = ClassificationHead(config, kind)
    values = torch.nn.functional.normalize(torch.randn(8, 768), dim=1)
    labels = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2])
    logits = head(values)
    assert logits.shape == (8, 5)
    criterion = loss_function(config, torch.device("cpu"))
    assert torch.isfinite(criterion(logits, labels))
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.01)
    before = [parameter.detach().clone() for parameter in head.parameters()]
    numerator, denominator = train_batch(head, values, labels, optimizer, criterion, 1.0)
    assert numerator > 0 and denominator == len(labels)
    assert any(not torch.equal(original, updated) for original, updated in zip(before, head.parameters()))
    first = predict_scores(head, values, torch.device("cpu"))
    second = predict_scores(head, values, torch.device("cpu"))
    assert torch.equal(first, second)
    assert torch.allclose(first.sum(dim=1), torch.ones(8))
    with pytest.raises(ValueError, match="shape"):
        head(torch.randn(2, 5))
    with pytest.raises(ValueError, match="FP32"):
        head(values.half())


def test_weighted_loss_is_batch_partition_independent() -> None:
    from torch.utils.data import DataLoader, TensorDataset

    config = cpu_config(class_weights=(1, 2, 3, 4, 5), label_smoothing=0.1)
    model = ClassificationHead(config, "linear").eval()
    features = torch.randn(9, 768)
    labels = torch.tensor([0, 0, 1, 1, 1, 2, 3, 4, 4])
    criterion = loss_function(config, torch.device("cpu"))
    expected = float(criterion(model(features), labels).detach())
    actual = evaluate_loss(model, DataLoader(TensorDataset(features, labels), batch_size=4), criterion, torch.device("cpu"))
    assert actual == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("row, message", [
    ({"id": "x", "text": "", "source": "fixture", "label": "Debugging"}, "text"),
    ({"id": "x", "text": "opaque", "source": "fixture", "label": "Other"}, "label"),
    ({"id": "x", "text": "opaque", "source": "fixture"}, "label"),
    ({"id": "x", "text": "opaque", "source": "fixture", "label": ["Testing"]}, "label"),
    ({"id": "x", "text": "opaque", "source": "fixture", "status": "needs_review", "label": "Testing"}, "null"),
    ({"id": "x", "text": "opaque", "source": "fixture", "group_id": 123}, "group_id"),
])
def test_dataset_malformed_records(row: dict, message: str) -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "records.jsonl"
        path.write_text(json.dumps(row) + "\n")
        with pytest.raises(ValueError, match=message):
            read_records(path, cpu_config().labels)


def test_dataset_deduplication_and_review_policy() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "records.jsonl"
        rows = [
            {"id": "a", "text": "opaque", "source": "fixture", "label": "Testing"},
            {"id": "b", "text": "opaque", "source": "fixture", "label": "Testing"},
            {"id": "c", "text": "review", "source": "fixture", "status": "needs_review"},
            {"id": "d", "text": "ood", "source": "fixture", "status": "out_of_domain"},
        ]
        path.write_text("\n".join(json.dumps(row) for row in rows))
        records = read_records(path, cpu_config().labels)
        assert len(records) == 3 and records[1].label is None
        rows[1]["label"] = "Debugging"
        path.write_text("\n".join(json.dumps(row) for row in rows))
        with pytest.raises(ValueError, match="conflicting duplicate"):
            read_records(path, cpu_config().labels)
        path.write_text('{"id":')
        with pytest.raises(ValueError, match="records.jsonl:1"):
            read_records(path, cpu_config().labels)
        path.write_text("")
        with pytest.raises(ValueError, match="empty"):
            read_records(path, cpu_config().labels)


def test_splits_are_reproducible_group_disjoint_and_complete() -> None:
    pytest.importorskip("sklearn")
    config = cpu_config()
    records = records_fixture()
    # Add distinct text from an existing conversation; its group must stay together.
    records.append(replace(records[0], id="extra", text="extra-schema-only"))
    first = split_records(records, config)
    assert first == split_records(records, config)
    seen_groups: dict[str, str] = {}
    for split, indices in first.items():
        if split != "challenge":
            assert {records[index].label for index in indices} == set(config.labels)
        for index in indices:
            group = records[index].group_id
            assert seen_groups.setdefault(group, split) == split
    assert sorted(index for indices in first.values() for index in indices) == list(range(len(records)))
    with pytest.raises(ValueError, match="10 independent"):
        split_records(records[:4], config)
    with pytest.raises(ValueError, match="human-reviewed"):
        split_records([RequestRecord("a", "opaque", "fixture", status="needs_review")], config)


def test_early_stopping_and_finite_validation() -> None:
    stopping = EarlyStopping(2, 0.1)
    assert stopping.update(2.0) == (True, False)
    assert stopping.update(1.95) == (True, False)  # Save true best despite min_delta.
    assert stopping.update(1.96) == (False, True)
    with pytest.raises(ValueError, match="not finite"):
        stopping.update(float("nan"))


def test_checkpoint_round_trip_checksum_and_safe_load() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "best.pt"
        config = cpu_config()
        model = ClassificationHead(config).eval()
        payload = {
            "format": "specialist-v1", "kind": "mlp", "config": config.to_dict(),
            "model_state": model.state_dict(), "prepared_identity": "fixture",
            "encoder_identity": "fixture", "epoch": 1, "step": 1,
        }
        save_checkpoint(path, payload)
        restored, checkpoint, _ = load_head(path, torch.device("cpu"))
        values = torch.randn(2, 768)
        assert torch.equal(model(values), restored(values))
        assert checkpoint["epoch"] == 1
        path.write_bytes(path.read_bytes() + b"corrupted")
        with pytest.raises(ValueError, match="checksum"):
            load_checkpoint(path)


def test_text_inference_requires_checkpoint_and_is_deterministic(monkeypatch) -> None:
    from llm_specialist import evaluation

    with TemporaryDirectory() as directory:
        path = Path(directory) / "trained-fixture.pt"
        config = cpu_config()
        deterministic_seed(42)
        model = ClassificationHead(config).eval()
        save_checkpoint(path, {
            "format": "specialist-v1", "kind": "mlp", "config": config.to_dict(),
            "model_state": model.state_dict(), "prepared_identity": "fixture",
            "encoder_identity": "fixture", "epoch": 1, "step": 1,
        })
        monkeypatch.setattr(evaluation, "FrozenEncoder", lambda settings: FrozenEncoder(settings, factory=FakeSentenceTransformer))
        classifier = evaluation.SpecialistClassifier(path, "cpu")
        first = classifier.predict(["unlabeled schema-only input"])
        second = classifier.predict(["unlabeled schema-only input"])
        assert first[0]["scores"] == second[0]["scores"]
        assert len(first[0]["scores"]) == 5 and first[0]["scores_calibrated"] is False
        assert first[0]["predicted_label"] in config.labels
        with pytest.raises(ValueError, match="checksum"):
            evaluation.SpecialistClassifier(Path(directory) / "missing.pt", "cpu")


def test_artifact_namespaces_and_exclusive_writes(monkeypatch) -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        monkeypatch.setattr(artifacts, "REPO_ROOT", root)
        with pytest.raises(ValueError, match="below"):
            artifacts.namespace_path(root / "checkpoints/legacy/latest.pt", "checkpoints")
        with pytest.raises(ValueError, match="path-safe"):
            artifacts.run_path("../legacy", "logs")
        output = root / "exclusive.json"
        atomic_json(output, {"first": True})
        with pytest.raises(FileExistsError):
            atomic_json(output, {"second": True})
        assert json.loads(output.read_text()) == {"first": True}


def test_metric_values_and_confusion_axes() -> None:
    pytest.importorskip("sklearn")
    metrics = classification_metrics([0, 1, 1], [0, 0, 1], ("A", "B"))
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["confusion_matrix"] == [[1, 0], [1, 1]]
    assert metrics["per_class"]["B"]["recall"] == 0.5
    assert metrics["confusion_matrix_axes"]["rows"] == "actual"


def test_viewer_size_attachment_and_progress_fields(monkeypatch) -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        monkeypatch.setattr(artifacts, "REPO_ROOT", root)
        monkeypatch.setenv("DISPLAY", ":unit-test")
        monkeypatch.setattr("llm_specialist.viewer.shutil.which", lambda _: "/usr/bin/kitty")
        status_path = root / "logs/specialist/fixture/progress.json"
        atomic_json(status_path, {"state": "STARTING"})
        arguments = []

        def fake_popen(args):
            arguments.extend(args)
            atomic_json(status_path.with_suffix(".attached"), {"unit_test_only": True})
            return SimpleNamespace(poll=lambda: None)

        monkeypatch.setattr("llm_specialist.viewer.subprocess.Popen", fake_popen)
        start_viewer(status_path)
        assert "initial_window_width=100c" in arguments
        assert "initial_window_height=22c" in arguments
        text = render_progress({"step": 5, "total_steps": 10, "eta_seconds": 3661, "checkpoint_path": "fixture/best.pt"})
        for expected in ("50.0%", "ETA 01:01:01", "tok/s", "TRAIN LOSS", "VAL LOSS", "TOKENS", "fixture/best.pt"):
            assert expected in text


def test_cli_help_does_not_import_sentence_transformers() -> None:
    result = subprocess.run([sys.executable, "-m", "llm_specialist.cli.specialist", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "smoke-encoder" in result.stdout and "prepare" in result.stdout


def test_specialist_imports_do_not_load_generative_pipeline() -> None:
    code = (
        "import sys, llm_specialist.training, llm_specialist.evaluation; "
        "assert not {'llm_architecture.model', 'llm_training.training', 'llm_tokenizer.tokenizer'} & set(sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_synthetic_training_checkpoint_evaluation_and_reproducibility(monkeypatch) -> None:
    pytest.importorskip("sklearn")
    from llm_specialist.evaluation import evaluate_run

    with TemporaryDirectory() as directory:
        root = Path(directory)
        monkeypatch.setattr(artifacts, "REPO_ROOT", root)
        config = cpu_config(max_epochs=2, batch_size=32)
        prepared = prepared_fixture(root / "prepared", config)
        launches = []

        def fake_viewer(path):
            launches.append(path)

            def check_running():
                run_metadata = json.loads((path.parent.parent / "run.json").read_text())
                assert run_metadata["state"] == "RUNNING"
                return None

            return SimpleNamespace(poll=check_running)

        metadata = train_run(prepared, "test-only-first", config, viewer_factory=fake_viewer)
        assert metadata["state"] == "COMPLETED" and len(launches) == 2
        first = load_checkpoint(root / "checkpoints/specialist/test-only-first/mlp/best.pt")
        train_run(prepared, "test-only-second", config, viewer_factory=fake_viewer)
        second = load_checkpoint(root / "checkpoints/specialist/test-only-second/mlp/best.pt")
        assert all(torch.equal(value, second["model_state"][key]) for key, value in first["model_state"].items())
        report = evaluate_run(prepared, "test-only-first", "cpu")
        assert set(report["heads"]) == {"linear", "mlp"}
        assert report["heads"]["mlp"]["selected_epoch"] == first["epoch"]
        assert len(report["heads"]["linear"]["confusion_matrix"]) == 5
        assert report["encoder_latency"] is None
        with pytest.raises(FileExistsError):
            train_run(prepared, "test-only-first", config, viewer_factory=fake_viewer)
        manifest, _, _ = load_prepared(prepared)
        manifest["splits"]["test"].append(manifest["splits"]["train"][0])
        manifest["identity"] = identity({key: value for key, value in manifest.items() if key != "identity"})
        atomic_json(prepared / "manifest.json", manifest, replace=True)
        with pytest.raises(ValueError, match="overlap"):
            load_prepared(prepared)


def test_viewer_failure_prevents_training_and_preserves_failure_record(monkeypatch) -> None:
    pytest.importorskip("sklearn")
    with TemporaryDirectory() as directory:
        root = Path(directory)
        monkeypatch.setattr(artifacts, "REPO_ROOT", root)
        config = cpu_config(max_epochs=1)
        prepared = prepared_fixture(root / "prepared", config)

        def missing_viewer(path):
            raise RuntimeError("no graphical viewer")

        with pytest.raises(RuntimeError, match="viewer"):
            train_run(prepared, "viewer-failure", config, "linear", viewer_factory=missing_viewer)
        assert not list((root / "checkpoints/specialist/viewer-failure").rglob("*.pt"))
        report = json.loads((root / "logs/specialist/viewer-failure/run.json").read_text())
        assert report["state"] == "FAILED"


def test_prepare_cache_reuse_invalidation_and_corruption(monkeypatch) -> None:
    pytest.importorskip("sklearn")
    from dataclasses import asdict

    with TemporaryDirectory() as directory:
        root = Path(directory)
        monkeypatch.setattr(artifacts, "REPO_ROOT", root)
        monkeypatch.setattr(data, "REPO_ROOT", root)
        config = cpu_config()
        source = root / "schema-fixture.jsonl"
        source.write_text("\n".join(json.dumps(asdict(record)) for record in records_fixture()))
        calls = []

        class CountingEncoder(FrozenEncoder):
            def __init__(self, settings):
                super().__init__(settings, factory=FakeSentenceTransformer)

            def encode(self, texts):
                calls.append(len(texts))
                return super().encode(texts)

        first_dir = root / "data/processed/specialist/first"
        first = data.prepare_dataset(source, first_dir, config, encoder_factory=CountingEncoder)
        assert calls and load_prepared(first_dir)[1].shape == (100, 768)
        calls.clear()
        second = data.prepare_dataset(source, root / "data/processed/specialist/second", config, encoder_factory=CountingEncoder)
        assert not calls and second["cache_identity"] == first["cache_identity"]
        with pytest.raises(FileExistsError):
            data.prepare_dataset(source, first_dir, config, encoder_factory=CountingEncoder)
        changed_config = replace(config, encoder=replace(config.encoder, max_tokens=1024))
        third = data.prepare_dataset(source, root / "data/processed/specialist/third", changed_config, encoder_factory=CountingEncoder)
        assert calls and third["cache_identity"] != first["cache_identity"]
        cache = root / "data/processed/specialist/embedding-cache" / first["cache_identity"] / "embeddings.npz"
        cache.write_bytes(cache.read_bytes() + b"corrupt")
        with pytest.raises(ValueError, match="checksum"):
            data.prepare_dataset(source, root / "data/processed/specialist/fourth", config, encoder_factory=CountingEncoder)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
