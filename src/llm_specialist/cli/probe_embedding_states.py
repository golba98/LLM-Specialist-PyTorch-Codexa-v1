"""Inspect frozen token-level representations without training any decoder."""

import argparse
import json
from pathlib import Path
import sys

import torch

from llm_specialist.config import EncoderConfig
from llm_specialist.encoder import FrozenEncoder


def main() -> None:
    """Run the explicit command-line operation with validated inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    encoder = FrozenEncoder(EncoderConfig(device=args.device))
    inputs = encoder.model.tokenizer(['task: search result | query: What did we discuss earlier?'], return_tensors='pt')
    inputs = {name: value.to(encoder.device) for name, value in inputs.items()}
    with torch.inference_mode():
        output = encoder.model[0].auto_model(**inputs, output_hidden_states=True)
    report = dict(encoder=encoder.metadata(), last_hidden_state=list(output.last_hidden_state.shape),
                  intermediate_shapes=[list(t.shape) for t in output.hidden_states],
                  trainable_parameters=sum(p.numel() for p in encoder.model.parameters() if p.requires_grad))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
