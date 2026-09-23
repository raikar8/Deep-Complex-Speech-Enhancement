"""Smoke tests for the DCCTN Soft/Hard MoE variants.

Run from the repository root after installing the project dependencies:

    python3 validate_moe.py

Use ``--backward`` to additionally verify that the enhancement loss can
backpropagate through the router. The default test only performs construction
and inference to keep memory usage lower.
"""

import argparse
import torch

from moe_networks import DCCTN_SoftMoE, DCCTN_HardMoE


INPUT_SAMPLES = 64000  # four seconds at 16 kHz, matching Dataprep.py


def validate_model(model_cls, backward=False):
    model = model_cls()
    model.eval()
    if not hasattr(model, "TB") or not hasattr(model.TB, "last_router_probabilities"):
        raise AssertionError("The model does not expose the expected MoE router")

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    audio = torch.randn(1, INPUT_SAMPLES)

    if backward:
        enhanced = model(audio)
        if enhanced.ndim == 1:
            enhanced = enhanced.unsqueeze(0)
        loss = enhanced.square().mean()
        loss.backward()
        if model.TB.router[0].weight.grad is None:
            raise AssertionError("Router did not receive a gradient")
    else:
        with torch.no_grad():
            enhanced = model(audio)

    if enhanced.shape[0] != audio.shape[0]:
        raise AssertionError("Batch mismatch: {} -> {}".format(audio.shape, enhanced.shape))
    if enhanced.ndim not in (1, 2):
        raise AssertionError("Expected waveform output, got shape {}".format(enhanced.shape))

    probabilities = model.TB.last_router_probabilities
    if probabilities is None or probabilities.shape != (audio.shape[0], 2):
        raise AssertionError("Unexpected router shape: {}".format(
            None if probabilities is None else probabilities.shape))
    if not torch.allclose(probabilities.sum(dim=-1), torch.ones(audio.shape[0]), atol=1e-5):
        raise AssertionError("Router probabilities do not sum to one")

    selected = model.TB.last_selected_expert
    if selected is None or selected.shape != (audio.shape[0],):
        raise AssertionError("Unexpected selected-expert shape")

    print("{}: PASS".format(model.name))
    print("  parameters: {:,}".format(parameter_count))
    print("  input:      {}".format(tuple(audio.shape)))
    print("  output:     {}".format(tuple(enhanced.shape)))
    print("  routing:    {} (NB, WB)".format(probabilities[0].tolist()))
    print("  selected:   {}".format(int(selected[0])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backward", action="store_true",
                        help="also run a loss/backward router-gradient check")
    args = parser.parse_args()

    torch.manual_seed(9999)
    validate_model(DCCTN_SoftMoE, backward=args.backward)
    validate_model(DCCTN_HardMoE, backward=args.backward)


if __name__ == "__main__":
    main()
