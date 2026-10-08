"""Serve frozen retrieval embeddings in the isolated specialist environment."""

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
import resource
import torch

from llm_specialist.config import EncoderConfig
from llm_specialist.encoder import FrozenEncoder


def resources() -> dict:
    """Measure only this worker's process and optional CUDA allocator."""
    return dict(peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                allocated_bytes=torch.cuda.memory_allocated() if torch.cuda.is_available() else None,
                reserved_bytes=torch.cuda.memory_reserved() if torch.cuda.is_available() else None,
                peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
                peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None)


def main() -> None:
    """Keep the line-delimited protocol stdout free of model-loading output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    with redirect_stdout(sys.stderr):
        encoder = FrozenEncoder(EncoderConfig(device=args.device))
    prompts = {name: encoder.model.prompts.get(name) for name in ("SearchQuery", "Document")}
    if prompts != {"SearchQuery": "task: search result | query: ", "Document": "title: none | text: "}:
        raise ValueError("Retrieval prompt registry mismatch.")
    metadata = encoder.metadata()
    metadata.update(prompts=prompts, prefix=None, chunking="480-utf8-bytes-v1")
    print(json.dumps({"ready": metadata, "resources": resources()}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("shutdown") is True:
                break
            if request["task"] not in prompts:
                raise ValueError("Only retrieval tasks are supported.")
            with redirect_stdout(sys.stderr):
                vectors, counts = encoder.encode(request["texts"], task=request["task"])
            print(json.dumps({"vectors": vectors.tolist(), "counts": counts, "resources": resources()}, allow_nan=False), flush=True)
        except Exception as error:
            print(json.dumps({"error": str(error)}), flush=True)


if __name__ == "__main__":
    main()
