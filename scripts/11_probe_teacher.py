"""Script 11 — Probe del motivo con un teacher grande (teacher congelato).

Estrae embedding di donateacry con HuBERT-base congelato (95M parametri,
pre-addestrato su LibriSpeech; pesi distribuiti via torchaudio, licenza
MIT del rilascio fairseq) e ripete la probe del motivo con split per bambino
e, per diagnosi, con split leaky per clip.

Scopo diagnostico (deciso il 26/8 dopo l'adattamento senza effetto): se anche
un encoder ~80 volte più grande del nostro resta al caso sullo split per
bambino, il collo di bottiglia sono le etichette/i dati di donateacry, non la
capacità del modello; se sale sensibilmente, il binario B (distillazione dal
teacher) diventa la strada.

Uso:
    python scripts/11_probe_teacher.py
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

RADICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RADICE / "src"))

log = logging.getLogger("11_teacher")


def embedding_teacher(percorsi: list[Path], device: str = "cuda") -> np.ndarray:
    """Embedding HuBERT-base per clip: media temporale dell'ultimo layer.

    I clip donateacry sono brevi (~7 s): si processano interi, uno alla volta.
    """
    bundle = torchaudio.pipelines.HUBERT_BASE
    modello = bundle.get_model().to(device).eval()
    log.info("Teacher HuBERT-base caricato (%.1fM parametri)",
             sum(p.numel() for p in modello.parameters()) / 1e6)
    uscite = []
    with torch.no_grad():
        for percorso in percorsi:
            audio, sr = sf.read(percorso, dtype="float32", always_2d=True)
            onda = torch.from_numpy(audio.mean(axis=1))
            if sr != bundle.sample_rate:
                onda = torchaudio.functional.resample(onda, sr, bundle.sample_rate)
            strati, _ = modello.extract_features(onda.unsqueeze(0).to(device))
            # Media sul tempo dell'ultimo strato: [1, T', 768] -> [768]
            uscite.append(strati[-1].mean(dim=1).squeeze(0).cpu().numpy())
    return np.stack(uscite)


def valuta(X: np.ndarray, y: np.ndarray, cv, gruppi=None) -> tuple[float, float]:
    """AUC macro OVR e balanced accuracy medie sui fold."""
    classi = np.unique(y)
    auc, bal = [], []
    divisioni = cv.split(X, y, gruppi) if gruppi is not None else cv.split(X, y)
    for train, test in divisioni:
        modello = LogisticRegression(max_iter=3000, class_weight="balanced")
        modello.fit(X[train], y[train])
        prob = modello.predict_proba(X[test])
        piene = np.zeros((len(test), len(classi)))
        for j, c in enumerate(modello.classes_):
            piene[:, np.where(classi == c)[0][0]] = prob[:, j]
        try:
            auc.append(roc_auc_score(y[test], piene, multi_class="ovr",
                                     average="macro", labels=classi))
        except ValueError:
            auc.append(np.nan)
        bal.append(balanced_accuracy_score(y[test], classi[piene.argmax(axis=1)]))
    return float(np.nanmean(auc)), float(np.mean(bal))


def main() -> int:
    """Probe teacher su donateacry: split per bambino e split leaky a confronto."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    esclusi = {r["clip_id"] for r in csv.DictReader(
        open(RADICE / "data" / "esclusioni_finetuning.csv", encoding="utf-8"))}
    with open(RADICE / "data" / "licenses_manifest.csv", newline="",
              encoding="utf-8") as f:
        righe = [r for r in csv.DictReader(f)
                 if r["corpus"] == "donateacry" and r["clip_id"] not in esclusi]

    percorsi = [RADICE / r["percorso"] for r in righe]
    y = np.array([r["classe"] for r in righe])
    gruppi = np.array([r["contributore"] for r in righe])
    log.info("Probe teacher su %d clip, %d bambini", len(righe), len(set(gruppi)))

    X = embedding_teacher(percorsi, device)

    auc_b, bal_b = valuta(X, y, StratifiedGroupKFold(5, shuffle=True,
                                                     random_state=0), gruppi)
    auc_l, bal_l = valuta(X, y, StratifiedKFold(5, shuffle=True, random_state=0))
    # Variante a 3 classi alla Ubenwa
    maschera = np.isin(y, ["hungry", "belly_pain", "discomfort"])
    auc3, bal3 = valuta(X[maschera], y[maschera],
                        StratifiedGroupKFold(5, shuffle=True, random_state=0),
                        gruppi[maschera])

    log.info("HuBERT-base — per BAMBINO: AUC %.3f balacc %.3f | leaky per clip: "
             "AUC %.3f balacc %.3f | 3 classi per bambino: AUC %.3f balacc %.3f",
             auc_b, bal_b, auc_l, bal_l, auc3, bal3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
