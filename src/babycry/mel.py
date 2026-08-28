"""Front-end audio log-mel (README).

Parametri di riferimento: 16 kHz mono, 64 bande mel, finestra 25 ms,
hop 10 ms. L'estrazione gira su CPU (il costo è trascurabile e la GPU
resta libera per il training); la cache si salva in float16 e si legge
in training via memory-map. La normalizzazione per clip NON è applicata
qui: media e deviazione standard vengono restituite a parte e salvate
nell'indice, così il training normalizza al load senza ricalcoli.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio


class FrontendLogMel:
    """Estrattore log-mel configurato da configs/audio_frontend.yaml."""

    def __init__(self, cfg: dict):
        """Costruisce l'estrattore dai parametri del front-end.

        Args:
            cfg: dizionario con le chiavi sample_rate, n_mels, finestra_ms,
                hop_ms, f_min, f_max, log_eps (vedi configs/audio_frontend.yaml).
        """
        self.sample_rate = int(cfg["sample_rate"])
        self.n_mels = int(cfg["n_mels"])
        # Conversione ms -> campioni: 25 ms a 16 kHz = 400, 10 ms = 160
        self.n_fft = int(round(self.sample_rate * cfg["finestra_ms"] / 1000))
        self.hop = int(round(self.sample_rate * cfg["hop_ms"] / 1000))
        self.log_eps = float(cfg["log_eps"])

        # Trasformata mel di torchaudio, eseguita su CPU (vedi docstring modulo)
        self._melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.n_fft,
            hop_length=self.hop,
            f_min=float(cfg["f_min"]),
            f_max=float(cfg["f_max"]),
            n_mels=self.n_mels,
            power=2.0,
        )

    def da_file(self, percorso: Path) -> tuple[np.ndarray, float, float]:
        """Calcola il log-mel di un file audio.

        Il file viene letto con soundfile, portato a mono (media dei canali)
        e ricampionato a 16 kHz se necessario.

        Args:
            percorso: file audio (wav/flac/caf/ogg... qualunque formato di libsndfile).

        Returns:
            Tripla (logmel, media, dev_std) dove logmel è un array float16 di
            forma [n_frame, n_mels] e media/dev_std sono le statistiche del
            clip in float32 (per la normalizzazione per clip al load).

        Raises:
            ValueError: se il file è vuoto o più corto di una finestra STFT.
        """
        audio, sr = sf.read(percorso, dtype="float32", always_2d=True)
        # Mono: media dei canali (always_2d garantisce la forma [campioni, canali])
        segnale = torch.from_numpy(audio.mean(axis=1))

        if sr != self.sample_rate:
            segnale = torchaudio.functional.resample(segnale, sr, self.sample_rate)

        if segnale.numel() < self.n_fft:
            raise ValueError(f"clip troppo corto ({segnale.numel()} campioni): {percorso}")

        # Log-mel: log naturale con epsilon per stabilità numerica
        mel = self._melspec(segnale)                    # [n_mels, n_frame]
        logmel = torch.log(mel + self.log_eps).T        # [n_frame, n_mels]

        media = float(logmel.mean())
        dev_std = float(logmel.std())
        return logmel.numpy().astype(np.float16), media, dev_std


def apri_cache_memmap(percorso_dat: Path, n_frame_totali: int, n_mels: int) -> np.memmap:
    """Apre in sola lettura una cache log-mel come memory-map.

    I 16 GB di RAM del PC di sviluppo impongono di NON caricare mai le cache
    intere in memoria: la lettura passa sempre da np.memmap.

    Args:
        percorso_dat: file .dat scritto dallo script 03.
        n_frame_totali: numero totale di frame nel file (dal campo
            n_frame_totali dell'indice JSON).
        n_mels: numero di bande mel (64).

    Returns:
        Array memory-mapped float16 di forma [n_frame_totali, n_mels].
    """
    return np.memmap(percorso_dat, dtype=np.float16, mode="r",
                     shape=(n_frame_totali, n_mels))
