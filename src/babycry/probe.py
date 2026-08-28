"""Linear probe per l'arbitrato del bake-off (vedi README).

Due probe con encoder CONGELATO (embedding medio delle finestre del clip):
- primaria: motivo del pianto su donateacry, split SEMPRE per bambino
  (StratifiedGroupKFold sul contributore), AUC macro OVR + balanced accuracy
  + ECE. I clip spuri (esclusioni_finetuning.csv) restano fuori.
- secondaria: pianto/non-pianto — positivi dal pool di pianto, negativi
  difficili dal voto umano più negativi facili campionati, gruppi per
  contributore/uploader.

Promemoria: un risultato sopra il 90% su donateacry è un sospetto
di leakage da indagare, non un successo.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from babycry.dataset import CacheMel, VoceClip, finestre_clip
from babycry.encoder import EncoderCompatto

log = logging.getLogger(__name__)


def embedding_clip(encoder: EncoderCompatto, cache: CacheMel, voci: list[VoceClip],
                   device: str = "cuda", batch: int = 128) -> np.ndarray:
    """Embedding medio delle finestre di ogni clip, con encoder congelato.

    Returns:
        Matrice float32 [n_clip, dim_embedding].
    """
    encoder.eval()
    uscite = []
    with torch.no_grad():
        for voce in voci:
            finestre = finestre_clip(cache, voce).to(device)
            pezzi = [encoder(finestre[i:i + batch])
                     for i in range(0, len(finestre), batch)]
            uscite.append(torch.cat(pezzi).mean(dim=0).cpu().numpy())
    return np.stack(uscite)


def ece(probabilita: np.ndarray, veri: np.ndarray, n_bin: int = 10) -> float:
    """Expected Calibration Error sulla probabilità della classe predetta."""
    confidenze = probabilita.max(axis=1)
    predette = probabilita.argmax(axis=1)
    corrette = (predette == veri).astype(float)
    errore, bordi = 0.0, np.linspace(0, 1, n_bin + 1)
    for lo, hi in zip(bordi[:-1], bordi[1:]):
        dentro = (confidenze > lo) & (confidenze <= hi)
        if dentro.any():
            errore += dentro.mean() * abs(corrette[dentro].mean()
                                          - confidenze[dentro].mean())
    return float(errore)


def _probe_cv(X: np.ndarray, y: np.ndarray, gruppi: np.ndarray,
              n_fold: int = 5, seed: int = 0) -> dict:
    """Cross-validation raggruppata per bambino con regressione logistica.

    Returns:
        Dict con auc_macro, balanced_accuracy, ece (media ± dev.std sui fold).
    """
    classi = np.unique(y)
    cv = StratifiedGroupKFold(n_splits=n_fold, shuffle=True, random_state=seed)
    metriche = {"auc_macro": [], "balanced_accuracy": [], "ece": []}
    for train, test in cv.split(X, y, groups=gruppi):
        # class_weight balanced: donateacry è sbilanciatissimo (classi fortemente sbilanciate)
        modello = LogisticRegression(max_iter=2000, class_weight="balanced")
        modello.fit(X[train], y[train])
        prob = modello.predict_proba(X[test])
        # Allinea le probabilità a TUTTE le classi (un fold può non vederne una)
        prob_piene = np.zeros((len(test), len(classi)))
        for j, c in enumerate(modello.classes_):
            prob_piene[:, np.where(classi == c)[0][0]] = prob[:, j]
        try:
            if len(classi) == 2:
                auc = roc_auc_score(y[test], prob_piene[:, 1])
            else:
                auc = roc_auc_score(y[test], prob_piene, multi_class="ovr",
                                    average="macro", labels=classi)
        except ValueError:
            auc = float("nan")   # fold senza una classe nel test: si segnala
        indice_vero = np.searchsorted(classi, y[test])
        metriche["auc_macro"].append(auc)
        metriche["balanced_accuracy"].append(
            balanced_accuracy_score(y[test], classi[prob_piene.argmax(axis=1)]))
        metriche["ece"].append(ece(prob_piene, indice_vero))
    return {nome: (float(np.nanmean(v)), float(np.nanstd(v)))
            for nome, v in metriche.items()}


def probe_motivo(encoder: EncoderCompatto, radice: Path, device: str = "cuda") -> dict:
    """Probe primaria: motivo del pianto su donateacry, split per bambino."""
    esclusi = set()
    percorso_esclusi = radice / "data" / "esclusioni_finetuning.csv"
    if percorso_esclusi.exists():
        with open(percorso_esclusi, newline="", encoding="utf-8") as f:
            esclusi = {r["clip_id"] for r in csv.DictReader(f)}

    cache = CacheMel(radice, ["donateacry"])
    voci = [v for v in cache.clip if v.clip_id not in esclusi]
    X = embedding_clip(encoder, cache, voci, device)
    y = np.array([v.classe for v in voci])
    gruppi = np.array([v.contributore for v in voci])
    log.info("Probe motivo: %d clip, %d bambini, classi %s",
             len(voci), len(set(gruppi)), dict(zip(*np.unique(y, return_counts=True))))
    return _probe_cv(X, y, gruppi)


def probe_pianto(encoder: EncoderCompatto, radice: Path, device: str = "cuda",
                 n_negativi_facili: int = 400, seed: int = 0) -> dict:
    """Probe secondaria: pianto/non-pianto con negativi difficili e facili."""
    rng = np.random.default_rng(seed)
    cache = CacheMel(radice, ["donateacry", "freesound", "fsd50k", "vocalsound"])
    per_id = {v.clip_id: v for v in cache.clip}

    with open(radice / "data" / "pool_pianto.csv", newline="", encoding="utf-8") as f:
        positivi_id = [r["clip_id"] for r in csv.DictReader(f)]
    with open(radice / "data" / "negativi_difficili.csv", newline="",
              encoding="utf-8") as f:
        negativi_id = [r["clip_id"] for r in csv.DictReader(f)]

    positivi = [per_id[c] for c in positivi_id if c in per_id]
    negativi = [per_id[c] for c in negativi_id if c in per_id]
    # Negativi facili: clip a caso di FSD50K/VocalSound fuori dal pool positivo
    vietati = set(positivi_id) | set(negativi_id)
    facili = [v for v in cache.clip if v.corpus in ("fsd50k", "vocalsound")
              and v.clip_id not in vietati]
    negativi += list(rng.choice(facili, size=min(n_negativi_facili, len(facili)),
                                replace=False))

    voci = positivi + negativi
    X = embedding_clip(encoder, cache, voci, device)
    y = np.array([1] * len(positivi) + [0] * len(negativi))
    gruppi = np.array([v.contributore or v.clip_id for v in voci])
    log.info("Probe pianto: %d positivi, %d negativi (%d difficili)",
             len(positivi), len(negativi), len(negativi_id))
    return _probe_cv(X, y, gruppi)
