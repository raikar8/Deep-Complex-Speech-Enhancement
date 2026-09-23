"""MoE bottlenecks for the DCCTN speech-enhancement model.

The repository represents complex activations with ``utils.ComplexTensor``.
Both experts therefore process real and imaginary parts through the existing
cplxmodule layers and return a ComplexTensor with the original shape.
"""

import torch
import torch.nn as nn

from cplxmodule.nn import CplxConv2d, CplxBatchNorm2d
from complexPyTorch.complexFunctions import complex_relu
from modules import ComplexTensor
from Network import DCCTN


class ComplexNBExpert(nn.Module):
    """Local/narrow-band expert.

    The bottleneck of DCCTN has one frequency bin after the encoder. Therefore
    the NB/WB distinction is implemented as different temporal receptive fields
    at this insertion point: NB is local, while WB sees a wider context.
    """

    def __init__(self, channels):
        super().__init__()
        self.conv = CplxConv2d(channels, channels, kernel_size=(1, 3),
                               padding=(0, 1), bias=True)
        self.norm = CplxBatchNorm2d(channels)

    def forward(self, x):
        return complex_relu(self.norm(self.conv(x)))


class ComplexWBExpert(nn.Module):
    """Wide-context expert with a larger temporal receptive field."""

    def __init__(self, channels):
        super().__init__()
        self.conv = CplxConv2d(channels, channels, kernel_size=(1, 9),
                               padding=(0, 4), bias=True)
        self.norm = CplxBatchNorm2d(channels)

    def forward(self, x):
        return complex_relu(self.norm(self.conv(x)))


class ComplexMoE(nn.Module):
    """Two-expert soft or hard complex mixture of experts.

    Routing is sample-level. Soft mode mixes both expert outputs. Hard mode
    uses top-1 routing in the forward pass and a straight-through estimator so
    the router remains trainable with the existing waveform loss.
    """

    def __init__(self, channels, mode="soft", temperature=1.0):
        super().__init__()
        if mode not in ("soft", "hard"):
            raise ValueError("mode must be 'soft' or 'hard'")
        self.mode = mode
        self.temperature = temperature
        hidden = max(channels // 4, 8)
        self.nb_expert = ComplexNBExpert(channels)
        self.wb_expert = ComplexWBExpert(channels)
        self.router = nn.Sequential(
            nn.Linear(2 * channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )
        self.last_router_probabilities = None
        self.last_selected_expert = None

    @staticmethod
    def _combine(nb, wb, weights):
        weights = weights[:, :, None, None]
        return ComplexTensor(
            weights[:, 0] * nb.real + weights[:, 1] * wb.real,
            weights[:, 0] * nb.imag + weights[:, 1] * wb.imag,
        )

    def forward(self, x):
        magnitude = torch.sqrt(x.real.square() + x.imag.square() + 1e-8)
        stats = torch.cat((magnitude.mean(dim=(2, 3)),
                           magnitude.std(dim=(2, 3), unbiased=False)), dim=1)
        probabilities = torch.softmax(self.router(stats) / self.temperature, dim=-1)

        nb = self.nb_expert(x)
        wb = self.wb_expert(x)
        selected = probabilities.argmax(dim=-1)
        if self.mode == "soft":
            weights = probabilities
        else:
            hard = torch.zeros_like(probabilities).scatter_(1, selected[:, None], 1.0)
            weights = hard - probabilities.detach() + probabilities

        self.last_router_probabilities = probabilities.detach()
        self.last_selected_expert = selected.detach()
        return self._combine(nb, wb, weights)


class DCCTN_SoftMoE(DCCTN):
    """DCCTN with a differentiable NB/WB bottleneck mixture."""

    def __init__(self, L=256, N=256, H=128, Mask=[5, 7], B=24, F_dim=129):
        super().__init__(L=L, N=N, H=H, Mask=Mask, B=B, F_dim=F_dim)
        self.TB = ComplexMoE(channels=8 * B, mode="soft")
        self.name = "DCCTN_SoftMoE"


class DCCTN_HardMoE(DCCTN):
    """DCCTN with straight-through top-1 NB/WB routing."""

    def __init__(self, L=256, N=256, H=128, Mask=[5, 7], B=24, F_dim=129):
        super().__init__(L=L, N=N, H=H, Mask=Mask, B=B, F_dim=F_dim)
        self.TB = ComplexMoE(channels=8 * B, mode="hard")
        self.name = "DCCTN_HardMoE"
