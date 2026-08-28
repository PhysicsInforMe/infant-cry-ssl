"""Dataset di patch log-mel dalla cache memory-mapped (vedi README).

La cache (script 03) è un .dat float16 per corpus, letto SEMPRE via np.memmap
(vincolo dei 16 GB di RAM). Ogni item è una patch di 3 s (300 frame × 64 mel)
normalizzata per clip con media e deviazione standard salvate nell'indice.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from babycry.mel import apri_cache_memmap


class VoceClip:
    """Riferimento leggero a un clip nella cache (offset, statistiche, metadati)."""

    __slots__ = ("corpus", "clip_id", "offset", "n_frame", "media", "dev_std",
                 "classe", "contributore")

    def __init__(self, corpus: str, clip_id: str, voce: dict):
        self.corpus = corpus
        self.clip_id = clip_id
        self.offset = int(voce["offset_frame"])
        self.n_frame = int(voce["n_frame"])
        self.media = float(voce["media"])
        self.dev_std = max(float(voce["dev_std"]), 1e-3)  # clip quasi muti: no div/0
        self.classe = voce.get("classe", "")
        self.contributore = voce.get("contributore", "")


class CacheMel:
    """Apre le cache mel di più corpora e indicizza i clip disponibili."""

    def __init__(self, radice: Path, corpora: list[str], min_frame: int = 300):
        """Args:
            radice: radice del repo (contiene cache/mel).
            corpora: nomi dei corpora da caricare (es. ["fsd50k", "vocalsound"]).
            min_frame: i clip più corti di una patch vengono scartati.
        """
        self.memmap: dict[str, np.memmap] = {}
        self.clip: list[VoceClip] = []
        for nome in corpora:
            indice = json.loads((radice / "cache" / "mel" / f"{nome}_index.json")
                                .read_text(encoding="utf-8"))
            self.memmap[nome] = apri_cache_memmap(
                radice / "cache" / "mel" / f"{nome}.dat",
                indice["n_frame_totali"], indice["n_mels"])
            for clip_id, voce in indice["clip"].items():
                if voce["n_frame"] >= min_frame:
                    self.clip.append(VoceClip(nome, clip_id, voce))

    def patch(self, voce: VoceClip, inizio: int, n_frame: int) -> np.ndarray:
        """Estrae una patch normalizzata per clip come float32 [n_frame, 64]."""
        blocco = np.asarray(
            self.memmap[voce.corpus][voce.offset + inizio:
                                     voce.offset + inizio + n_frame],
            dtype=np.float32)
        return (blocco - voce.media) / voce.dev_std


class DatasetPatchCasuali(Dataset):
    """Patch casuali per il pretraining SSL: un clip a indice, finestra casuale.

    I clip corti risultano leggermente sovracampionati rispetto ai lunghi:
    accettabile per l'SSL (e comune in letteratura).
    """

    def __init__(self, cache: CacheMel, patch_frame: int = 300, seed: int = 0):
        self.cache = cache
        self.patch_frame = patch_frame
        # Generatore per-worker: il seed di base viene mescolato con l'indice
        self.seed = seed

    def __len__(self) -> int:
        return len(self.cache.clip)

    def __getitem__(self, indice: int) -> torch.Tensor:
        voce = self.cache.clip[indice]
        # Finestra casuale dentro il clip (rng locale: niente stato condiviso)
        rng = np.random.default_rng((self.seed, indice,
                                     np.random.randint(0, 2 ** 31)))
        inizio = int(rng.integers(0, voce.n_frame - self.patch_frame + 1))
        patch = self.cache.patch(voce, inizio, self.patch_frame)
        return torch.from_numpy(patch).unsqueeze(0)  # [1, 300, 64]


def finestre_clip(cache: CacheMel, voce: VoceClip, patch_frame: int = 300,
                  hop: int = 150) -> torch.Tensor:
    """Tutte le finestre scorrevoli di un clip (per le probe, deterministico).

    Returns:
        Tensore [n_finestre, 1, patch_frame, 64]; i clip più corti di una
        patch non arrivano qui (filtrati da CacheMel).
    """
    finestre = []
    for inizio in range(0, max(voce.n_frame - patch_frame, 0) + 1, hop):
        finestre.append(cache.patch(voce, inizio, patch_frame))
    return torch.from_numpy(np.stack(finestre)).unsqueeze(1)
