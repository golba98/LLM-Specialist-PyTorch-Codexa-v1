"""Validated request records, group-disjoint splits, and embedding caches."""

from dataclasses import asdict, dataclass
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from llm_specialist.artifacts import (
    REPO_ROOT, atomic_bytes, atomic_json, file_sha256, identity, namespace_path,
)
from llm_specialist.config import SpecialistConfig, config_from_dict
from llm_specialist.encoder import FrozenEncoder, encoder_metadata, validate_embeddings


@dataclass(frozen=True)
class RequestRecord:
    """One reviewed programming request with provenance and optional label."""

    id: str
    text: str
    source: str
    label: str | None = None
    group_id: str | None = None
    status: str = "labeled"


def read_records(path: str | Path, labels: tuple[str, ...]) -> list[RequestRecord]:
    """Read strict JSONL, deduplicating equivalent text and rejecting conflicts."""
    ids: set[str] = set()
    text_records: dict[str, RequestRecord] = {}
    groups: dict[str, str] = {}
    allowed = set(RequestRecord.__dataclass_fields__)
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or not set(row) <= allowed:
                    raise ValueError("record must be an object with only documented fields")
                for name in ("id", "text", "source"):
                    if not isinstance(row.get(name), str) or not row[name].strip():
                        raise ValueError(f"{name} must be a nonempty string")
                    row[name] = row[name].strip()
                if row["id"] in ids:
                    raise ValueError(f"duplicate id {row['id']!r}")
                ids.add(row["id"])
                if row.get("group_id") is not None:
                    if not isinstance(row["group_id"], str) or not row["group_id"].strip():
                        raise ValueError("group_id must be a nonempty string or null")
                    row["group_id"] = row["group_id"].strip()
                status = row.get("status", "labeled")
                if status not in ("labeled", "needs_review", "out_of_domain"):
                    raise ValueError("unknown record status")
                label = row.get("label")
                if status == "labeled" and (not isinstance(label, str) or label not in labels):
                    raise ValueError(f"label must be one of {list(labels)}")
                if status != "labeled" and label is not None:
                    raise ValueError("review/OOD records must have null or absent label")
                record = RequestRecord(**row)
                previous = text_records.get(record.text)
                if previous is not None:
                    if (previous.label, previous.status) != (record.label, record.status):
                        raise ValueError("conflicting duplicate annotations")
                    # Do not lose grouping constraints while deduplicating.
                    if previous.group_id != record.group_id:
                        raise ValueError("duplicate text has inconsistent group_id")
                    if record.id < previous.id:
                        text_records[record.text] = record
                else:
                    text_records[record.text] = record
                if record.group_id is not None:
                    existing_source = groups.setdefault(record.group_id, record.source)
                    if existing_source != record.source:
                        raise ValueError("group_id must be globally unique across sources")
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}:{number}: {error}") from error
    if not text_records:
        raise ValueError("Dataset must not be empty.")
    return sorted(text_records.values(), key=lambda record: record.id)


def split_records(records: list[RequestRecord], config: SpecialistConfig) -> dict[str, list[int]]:
    """Create reproducible roughly 80/10/10 stratified, group-disjoint splits."""
    from sklearn.model_selection import StratifiedGroupKFold

    labeled = [index for index, record in enumerate(records) if record.status == "labeled"]
    challenge = [index for index, record in enumerate(records) if record.status != "labeled"]
    if not labeled:
        raise ValueError("Training data requires human-reviewed labels; none are available.")
    targets = np.array([config.labels.index(records[index].label) for index in labeled])
    groups = np.array([
        "group:" + records[index].group_id if records[index].group_id is not None
        else "record:" + records[index].id
        for index in labeled
    ])
    for target, label in enumerate(config.labels):
        if len(set(groups[targets == target])) < 10:
            raise ValueError(
                f"Label {label!r} requires at least 10 independent groups for 80/10/10 splitting."
            )
    # Review examples from a conversation must not overlap supervised groups.
    challenge_groups = {record.group_id for record in records if record.status != "labeled" and record.group_id}
    labeled_groups = {records[index].group_id for index in labeled if records[index].group_id}
    if challenge_groups & labeled_groups:
        raise ValueError("Challenge groups must not overlap supervised groups.")
    dummy = np.zeros(len(labeled))
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config.training.seed)
    train, held_out = next(splitter.split(dummy, targets, groups))
    held_splitter = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=config.training.seed)
    for target, label in enumerate(config.labels):
        if len(set(groups[held_out][targets[held_out] == target])) < 2:
            raise ValueError(f"Held-out label {label!r} has insufficient groups; supply more data.")
    validation, test = next(held_splitter.split(dummy[held_out], targets[held_out], groups[held_out]))
    assignments = {
        "train": [labeled[int(index)] for index in train],
        "validation": [labeled[int(held_out[index])] for index in validation],
        "test": [labeled[int(held_out[index])] for index in test],
        "challenge": challenge,
    }
    for name in ("train", "validation", "test"):
        observed = {records[index].label for index in assignments[name]}
        if observed != set(config.labels):
            raise ValueError(f"Split {name} lacks class coverage; supply more independent groups.")
    return assignments


def _npz_bytes(embeddings: torch.Tensor, counts: list[int]) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, embeddings=embeddings.numpy(), token_counts=np.asarray(counts, dtype=np.int64))
    return buffer.getvalue()


def load_embedding_file(path: Path, checksum: str, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate cached arrays, their checksum, shape, and token counts."""
    if file_sha256(path) != checksum:
        raise ValueError(f"Embedding checksum mismatch: {path}.")
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) != {"embeddings", "token_counts"}:
            raise ValueError("Unexpected embedding cache fields.")
        values = arrays["embeddings"]
        counts = arrays["token_counts"]
        if values.dtype != np.float32 or len(values) != count:
            raise ValueError("Embedding cache dtype or record count mismatch.")
        if counts.dtype != np.int64 or counts.shape != (count,) or np.any(counts <= 0):
            raise ValueError("Invalid cached token counts.")
        embeddings = validate_embeddings(torch.from_numpy(values.copy()))
        return embeddings, torch.from_numpy(counts.copy())


def prepare_dataset(
    dataset: Path,
    output_dir: Path,
    config: SpecialistConfig,
    *,
    encoder_factory: Any = FrozenEncoder,
) -> dict[str, Any]:
    """Validate, split, and embed data without overwriting existing bundles."""
    output_dir = namespace_path(output_dir, "data/processed")
    if output_dir.exists():
        raise FileExistsError(f"Prepared output already exists: {output_dir}.")
    dataset_digest = file_sha256(dataset)
    records = read_records(dataset, config.labels)
    assignments = split_records(records, config)
    metadata = encoder_metadata(config.encoder)
    record_values = [asdict(record) for record in records]
    key_metadata = {"records": record_values, "encoder": metadata, "dataset_sha256": dataset_digest}
    key = identity(key_metadata)
    cache_dir = namespace_path(REPO_ROOT / "data/processed/specialist/embedding-cache" / key, "data/processed")
    cache_path = cache_dir / "embeddings.npz"
    metadata_path = cache_dir / "manifest.json"
    if cache_dir.exists():
        cached = json.loads(metadata_path.read_text())
        if cached.get("identity") != key or cached.get("key_metadata") != key_metadata:
            raise ValueError("Existing embedding cache is incompatible.")
        embeddings, counts = load_embedding_file(cache_path, cached["sha256"], len(records))
    else:
        encoder = encoder_factory(config.encoder)
        if encoder.metadata() != metadata:
            raise ValueError("Loaded encoder does not match the requested cache identity.")
        cache_dir.mkdir(parents=True, exist_ok=False)
        chunks: list[torch.Tensor] = []
        token_counts: list[int] = []
        for start in range(0, len(records), config.encoder.batch_size):
            texts = [record.text for record in records[start : start + config.encoder.batch_size]]
            encoded, batch_counts = encoder.encode(texts)
            chunks.append(encoded)
            token_counts.extend(batch_counts)
        embeddings = torch.cat(chunks)
        counts = torch.tensor(token_counts, dtype=torch.int64)
        atomic_bytes(cache_path, _npz_bytes(embeddings, token_counts))
        atomic_json(metadata_path, {"identity": key, "key_metadata": key_metadata, "sha256": file_sha256(cache_path)})
    if file_sha256(dataset) != dataset_digest:
        raise ValueError("Dataset changed during preparation; use an immutable input file.")
    output_dir.mkdir(parents=True, exist_ok=False)
    atomic_bytes(output_dir / "embeddings.npz", _npz_bytes(embeddings, counts.tolist()))
    manifest: dict[str, Any] = {
        "format": "specialist-prepared-v1",
        "config": config.to_dict(),
        "records": record_values,
        "splits": assignments,
        "dataset_sha256": dataset_digest,
        "encoder": metadata,
        "encoder_identity": identity(metadata),
        "cache_identity": key,
        "embeddings_sha256": file_sha256(output_dir / "embeddings.npz"),
    }
    manifest["identity"] = identity(manifest)
    atomic_json(output_dir / "manifest.json", manifest)
    return manifest


def load_prepared(directory: str | Path) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    """Load an immutable bundle and verify provenance and split integrity."""
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("format") != "specialist-prepared-v1":
        raise ValueError("Not a specialist-prepared-v1 dataset.")
    stored_identity = manifest.get("identity")
    if identity({key: value for key, value in manifest.items() if key != "identity"}) != stored_identity:
        raise ValueError("Prepared manifest identity mismatch.")
    config = config_from_dict(manifest["config"])
    records = manifest["records"]
    if not isinstance(records, list) or not records:
        raise ValueError("Prepared records must be a nonempty list.")
    ids: set[str] = set()
    texts: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != set(RequestRecord.__dataclass_fields__):
            raise ValueError("Prepared record fields are invalid.")
        for key in ("id", "text", "source"):
            if not isinstance(record[key], str) or not record[key].strip():
                raise ValueError(f"Prepared record {key} is invalid.")
        if record["id"] in ids or record["text"] in texts:
            raise ValueError("Prepared records contain duplicate IDs or texts.")
        ids.add(record["id"])
        texts.add(record["text"])
        if record["status"] not in ("labeled", "needs_review", "out_of_domain"):
            raise ValueError("Prepared record status is invalid.")
        if record["group_id"] is not None and (not isinstance(record["group_id"], str) or not record["group_id"].strip()):
            raise ValueError("Prepared group_id is invalid.")
    embeddings, counts = load_embedding_file(root / "embeddings.npz", manifest["embeddings_sha256"], len(records))
    if torch.any(counts > config.encoder.max_tokens):
        raise ValueError("Prepared token counts exceed the configured limit.")
    if identity(manifest["encoder"]) != manifest["encoder_identity"]:
        raise ValueError("Prepared encoder identity mismatch.")
    seen: set[int] = set()
    group_split: dict[str, str] = {}
    if set(manifest["splits"]) != {"train", "validation", "test", "challenge"}:
        raise ValueError("Prepared split names are invalid.")
    for name, indices in manifest["splits"].items():
        if not isinstance(indices, list):
            raise ValueError("Split indices must be lists.")
        for index in indices:
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(records):
                raise ValueError("Prepared split index is invalid.")
            if index in seen:
                raise ValueError("Prepared splits overlap.")
            seen.add(index)
            record = records[index]
            if name != "challenge" and record["label"] not in config.labels:
                raise ValueError("Supervised split contains an unlabeled record.")
            if name == "challenge" and (record["status"] == "labeled" or record["label"] is not None):
                raise ValueError("Challenge split contains a supervised record.")
            group = record.get("group_id")
            if group is not None and group_split.setdefault(group, name) != name:
                raise ValueError("Prepared groups overlap across splits.")
        if name != "challenge" and {records[i]["label"] for i in indices} != set(config.labels):
            raise ValueError("Prepared split lacks class coverage.")
    if seen != set(range(len(records))):
        raise ValueError("Prepared splits do not cover every record.")
    return manifest, embeddings, counts
