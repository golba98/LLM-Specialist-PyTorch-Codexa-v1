# LLM-Specialist

Frozen embedding encoder and independently trained classifiers.

Owns the pretrained EmbeddingGemma adapter, frozen embedding cache and linear/MLP classifier. It is not a from-scratch encoder. Preserve the independent specialist environment; do not install its Transformers/tokenizers pins into the generative environment. Outputs default to the caller’s directory or LLM_SPECIALIST_ROOT. Model downloads and classifier training require explicit commands; tests use fake encoders.

## Development

In the sibling workspace, use `../run.py --repo LLM-Specialist test`.
This selects the existing environment and sibling package sources without installing dependencies.
For a separately installed checkout, run `python -m pytest` after provisioning the documented dependencies and exact sibling version 0.1.0. These packages are local and not published to PyPI.

## Entry points

- `python -m llm_specialist.cli.specialist --help`
- `python -m llm_specialist.cli.memory_worker --help`
- `python -m llm_specialist.cli.probe_embedding_states --help`

## Integration and assets

`../compatibility.json` records the complete tested version set.
Checkpoint weights, tokenizers, datasets and generated logs are referenced by path; none are distributed in this package. Preserve tokenizer fingerprints and architecture lineage. Source provenance is in PROVENANCE.md.

## Validation and limitations

See the central VALIDATION.md for commands, results and unverified large-model checks.
The original project is preserved unchanged. No model promotion, training pipeline or remote publishing occurs as part of extraction.

# LLM-Specialist-PyTorch-Codexa-v1

## Canonical workspace integration

This repository remains independently versioned at its existing remote and is pinned as a submodule beneath the Codexa workspace root. Integration decisions live in ../documentation/training/SESSION_DECISIONS.md. Historical assets are external inputs; never commit weights, datasets or recovery snapshots.
