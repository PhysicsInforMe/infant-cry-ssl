"""Encoder compatto per il bake-off (vedi README).

CNN in stile BYOL-A: 4 stadi da due convoluzioni 3x3 (la seconda con stride 2),
pooling globale finale, embedding a 256 dimensioni. ~1.2M parametri: dentro il
budget 1-2M del bake-off e compatibile col target on-device.
Lo STESSO encoder viene usato da tutti i candidati A-F: cambia solo la testa.
"""

from __future__ import annotations

import torch
from torch import nn


class _Stadio(nn.Module):
    """Due conv 3x3 (BN+ReLU), la seconda dimezza le dimensioni spaziali."""

    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.blocco = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch_out), nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch_out), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocco(x)


class EncoderCompatto(nn.Module):
    """Encoder convoluzionale compatto: input [B, 1, 300, 64] -> embedding [B, 256]."""

    def __init__(self, dim_embedding: int = 256):
        super().__init__()
        self.dim_embedding = dim_embedding
        self.tronco = nn.Sequential(
            _Stadio(1, 32), _Stadio(32, 64), _Stadio(64, 128), _Stadio(128, 256))
        self.pool = nn.AdaptiveAvgPool2d(1)

    def mappa(self, x: torch.Tensor) -> torch.Tensor:
        """Mappa di feature prima del pooling: [B, 256, ~19, 4] (per i decoder)."""
        return self.tronco(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.tronco(x)).flatten(1)


def conta_parametri(modello: nn.Module) -> int:
    """Numero di parametri addestrabili (per il log del budget)."""
    return sum(p.numel() for p in modello.parameters() if p.requires_grad)
