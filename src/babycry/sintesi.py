"""Generazione di varianti sintetiche dei pianti (vedi README).

Due tecniche complementari:
- **vocoder** (`variante_vocoder`): pitch shift della F0 più spostamento delle
  formanti via PSOLA (comando "Change gender" di Praat, attraverso parselmouth).
  Simula un apparato fonatorio di taglia diversa mantenendo IDENTICA la traccia
  temporale (raffiche, pause, rotture): se l'originale emette il primo guaito a
  t=4,36 s, anche la variante lo fa.
- **rumore** (`variante_rumore`): miscela il pianto con rumore domestico a un
  SNR scelto, simulando l'ambiente reale (TV, voci, cane, elettrodomestici).

Avvertenza concettuale: le varianti aggiungono INVARIANZA, non
informazione — ereditano la gestualità del bambino di origine, quindi ai fini
dello split per bambino appartengono SEMPRE al contributore originale.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import parselmouth
import soundfile as sf
import torch
import torchaudio
from parselmouth.praat import call

# Picco massimo dell'audio in uscita (evita il clipping in scrittura PCM16)
_PICCO_MAX = 0.89


def carica_audio_mono(percorso: Path, sr_obiettivo: int) -> np.ndarray:
    """Carica un file audio come mono float64 al sample rate richiesto.

    Args:
        percorso: file audio leggibile da libsndfile.
        sr_obiettivo: sample rate di uscita (16000 nel progetto).

    Returns:
        Array float64 monodimensionale (richiesto da parselmouth).
    """
    audio, sr = sf.read(percorso, dtype="float32", always_2d=True)
    segnale = torch.from_numpy(audio.mean(axis=1))
    if sr != sr_obiettivo:
        segnale = torchaudio.functional.resample(segnale, sr, sr_obiettivo)
    return segnale.numpy().astype(np.float64)


def _normalizza_picco(audio: np.ndarray) -> np.ndarray:
    """Riporta il picco assoluto a _PICCO_MAX se lo supera (contro il clipping)."""
    picco = np.abs(audio).max()
    if picco > _PICCO_MAX and picco > 0:
        audio = audio * (_PICCO_MAX / picco)
    return audio


def variante_vocoder(segnale: np.ndarray, sr: int, semitoni: float, alpha: float,
                     f0_floor: float = 150.0, f0_ceil: float = 1000.0) -> np.ndarray:
    """Genera una variante con F0 spostata e formanti scalate (PSOLA di Praat).

    Args:
        segnale: audio mono float64.
        sr: sample rate.
        semitoni: spostamento della mediana F0 (positivo = più acuto).
        alpha: fattore di scala delle formanti (>1 = tratto vocale più corto,
            cioè bimbo più piccolo; <1 = più lungo).
        f0_floor: limite inferiore dell'analisi del pitch in Hz.
        f0_ceil: limite superiore dell'analisi del pitch in Hz.

    Returns:
        Audio della variante, float64, stessa durata (fattore durata = 1).

    Raises:
        ValueError: se la F0 del clip non è stimabile (clip non tonale).
    """
    suono = parselmouth.Sound(segnale, sampling_frequency=sr)
    pitch = suono.to_pitch(pitch_floor=f0_floor, pitch_ceiling=f0_ceil)
    mediana = call(pitch, "Get quantile", 0, 0, 0.5, "Hertz")
    if not np.isfinite(mediana) or mediana <= 0:
        raise ValueError("F0 non stimabile: variante vocoder impossibile")

    nuova_mediana = mediana * 2.0 ** (semitoni / 12.0)
    # "Change gender": PSOLA con (pitch floor, pitch ceiling, rapporto formanti,
    # nuova mediana F0, fattore di range del pitch, fattore di durata).
    # Fattore di durata 1.0 = la sequenza temporale degli eventi resta identica.
    modificato = call(suono, "Change gender", f0_floor, f0_ceil,
                      float(alpha), float(nuova_mediana), 1.0, 1.0)
    return _normalizza_picco(modificato.values[0].astype(np.float64))


def variante_rumore(pianto: np.ndarray, rumore: np.ndarray, snr_db: float,
                    rng: np.random.Generator) -> np.ndarray:
    """Miscela il pianto con un rumore di fondo al rapporto segnale/rumore dato.

    Il rumore viene tagliato (o ripetuto) alla lunghezza del pianto partendo
    da un offset casuale, poi scalato per ottenere l'SNR richiesto sulla
    potenza media dei due segnali.

    Args:
        pianto: audio mono float64 del pianto.
        rumore: audio mono float64 del rumore (qualunque lunghezza).
        snr_db: rapporto segnale/rumore in dB (basso = rumore forte).
        rng: generatore casuale (per l'offset di taglio del rumore).

    Returns:
        Audio miscelato, float64, stessa durata del pianto.

    Raises:
        ValueError: se il rumore è silenzio (potenza nulla).
    """
    n = len(pianto)
    # Rumore alla lunghezza giusta: ripetuto se corto, tagliato con offset casuale
    if len(rumore) < n:
        rumore = np.tile(rumore, int(np.ceil(n / len(rumore))))
    inizio = int(rng.integers(0, len(rumore) - n + 1))
    spezzone = rumore[inizio:inizio + n]

    potenza_pianto = float(np.mean(pianto ** 2))
    potenza_rumore = float(np.mean(spezzone ** 2))
    if potenza_rumore <= 0:
        raise ValueError("rumore a potenza nulla")

    guadagno = np.sqrt(potenza_pianto / (potenza_rumore * 10.0 ** (snr_db / 10.0)))
    return _normalizza_picco(pianto + guadagno * spezzone)


def scrivi_wav(percorso: Path, audio: np.ndarray, sr: int) -> float:
    """Scrive un wav PCM16 mono e restituisce la durata in secondi."""
    percorso.parent.mkdir(parents=True, exist_ok=True)
    sf.write(percorso, audio.astype(np.float32), sr, subtype="PCM_16")
    return len(audio) / sr
