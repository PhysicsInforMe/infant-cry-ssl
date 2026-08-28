"""Script 10 — Adattamento di dominio SSL sul pool di pianto + ri-probe.

Per un candidato del bake-off: riparte dall'encoder pre-addestrato (script 08)
e continua il SUO stesso pretext task sulle ore di pianto vero
(data/pool_pianto.csv), poi rifà la probe del motivo. Protocollo anti-leakage
non negoziabile: per ogni fold della probe l'adattamento vede
solo i clip donateacry dei bambini di TRAIN di quel fold (i clip
Freesound/FSD50K del pool sono di persone diverse, quindi sicuri ovunque).
Costo: 5 adattamenti per candidato, ma il pool è piccolo e i run sono brevi.

Oltre alle metriche a 5 classi riporta la variante a 3 classi (hungry,
belly_pain, discomfort) per confronto diretto con Gorin et al. (ICASSP 2023).

Uso:
    python scripts/10_adattamento_pianto.py --candidato D
    python scripts/10_adattamento_pianto.py --candidato D --passi 50   # smoke
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

RADICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RADICE / "src"))

from babycry.dataset import CacheMel, DatasetPatchCasuali  # noqa: E402
from babycry.encoder import EncoderCompatto  # noqa: E402
from babycry.pretext import CANDIDATI  # noqa: E402
from babycry.probe import _probe_cv, embedding_clip  # noqa: E402

log = logging.getLogger("10_adattamento")

# Riduzione a 3 classi per il confronto con Ubenwa (tabella 4 del paper)
TRE_CLASSI = {"hungry": "hungry", "belly_pain": "pain", "discomfort": "discomfort"}


def voci_probe(radice: Path) -> tuple:
    """Ricostruisce ESATTAMENTE l'insieme e l'ordine dei clip della probe motivo.

    Stesso codice di probe.probe_motivo: cache donateacry meno le esclusioni,
    nell'ordine dell'indice. Da qui dipendono i fold: non cambiare.
    """
    esclusi = set()
    percorso = radice / "data" / "esclusioni_finetuning.csv"
    if percorso.exists():
        with open(percorso, newline="", encoding="utf-8") as f:
            esclusi = {r["clip_id"] for r in csv.DictReader(f)}
    cache = CacheMel(radice, ["donateacry"])
    voci = [v for v in cache.clip if v.clip_id not in esclusi]
    y = np.array([v.classe for v in voci])
    gruppi = np.array([v.contributore for v in voci])
    return cache, voci, y, gruppi


def adatta_encoder(candidato: str, stato_base: dict, clip_adattamento: list,
                   cache_pool: CacheMel, cfg: dict, device: str) -> EncoderCompatto:
    """Continua il pretext del candidato sul sottoinsieme di pianto indicato.

    Args:
        candidato: lettera del candidato (stesso pretext del pretraining).
        stato_base: state_dict dell'encoder pre-addestrato (punto di partenza).
        clip_adattamento: voci del pool di pianto ammesse per questo fold.
        cache_pool: cache con i memmap dei corpora del pool.
        cfg: sezione `adattamento` della config.
        device: dispositivo cuda.

    Returns:
        Encoder adattato (le teste del pretext si scartano).
    """
    encoder = EncoderCompatto().to(device)
    encoder.load_state_dict(stato_base)
    modello = CANDIDATI[candidato](encoder).to(device)

    # Dataset ristretto ai clip ammessi (riusa i memmap della cache del pool)
    cache_fold = CacheMel.__new__(CacheMel)
    cache_fold.memmap = cache_pool.memmap
    cache_fold.clip = clip_adattamento
    dataset = DatasetPatchCasuali(cache_fold, 300, seed=0)
    loader = DataLoader(dataset, batch_size=int(cfg["batch"]), shuffle=True,
                        num_workers=0, drop_last=True)

    ottimizzatore = torch.optim.AdamW(modello.parameters(), lr=float(cfg["lr"]),
                                      weight_decay=0.05)
    scaler = torch.amp.GradScaler("cuda")
    modello.train()
    passo, obiettivo = 0, int(cfg["passi"])
    while passo < obiettivo:
        for batch in loader:
            if passo >= obiettivo:
                break
            batch = batch.to(device, non_blocking=True)
            ottimizzatore.zero_grad(set_to_none=True)
            with torch.autocast("cuda"):
                loss = modello.passo(batch)
            scaler.scale(loss).backward()
            scaler.step(ottimizzatore)
            scaler.update()
            passo += 1
    return encoder


def main() -> int:
    """Adattamento per-fold + ri-probe del motivo (5 e 3 classi)."""
    parser = argparse.ArgumentParser(description="Adattamento SSL sul pianto")
    parser.add_argument("--candidato", required=True, choices=sorted(CANDIDATI))
    parser.add_argument("--config", type=Path,
                        default=RADICE / "configs" / "bakeoff.yaml")
    parser.add_argument("--passi", type=int, help="override (smoke test)")
    argomenti = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    cfg_tutto = yaml.safe_load(argomenti.config.read_text(encoding="utf-8"))
    cfg = dict(cfg_tutto["adattamento"])
    if argomenti.passi:
        cfg["passi"] = argomenti.passi
    torch.manual_seed(int(cfg_tutto["seed"]))
    device = "cuda"
    if not torch.cuda.is_available():
        log.error("CUDA non disponibile: interrompo (regola di progetto).")
        return 1

    # Encoder di partenza dal bake-off
    percorso_base = (RADICE / cfg_tutto["cartella_checkpoint"]
                     / f"{argomenti.candidato}_encoder.pt")
    stato_base = torch.load(percorso_base, map_location=device,
                            weights_only=True)["encoder"]

    # Pool di pianto: voci nella cache, indicizzate per corpus/contributore
    with open(RADICE / "data" / "pool_pianto.csv", newline="", encoding="utf-8") as f:
        pool_ids = {r["clip_id"] for r in csv.DictReader(f)}
    cache_pool = CacheMel(RADICE, ["donateacry", "freesound", "fsd50k"])
    voci_pool = [v for v in cache_pool.clip if v.clip_id in pool_ids]
    log.info("Pool di pianto utilizzabile (>=3 s): %d clip", len(voci_pool))

    # Probe: stessi clip, stessi fold della probe del bake-off
    cache_probe, voci, y, gruppi = voci_probe(RADICE)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)

    X_per_fold: list[tuple[np.ndarray, np.ndarray]] = []
    embedding_fold: list[np.ndarray] = []
    fold_indici = list(cv.split(np.zeros(len(y)), y, groups=gruppi))
    for numero_fold, (train, test) in enumerate(fold_indici):
        bambini_train = set(gruppi[train])
        # Anti-leakage: dal pool si tolgono i clip donateacry dei bambini di test
        ammessi = [v for v in voci_pool
                   if v.corpus != "donateacry" or v.contributore in bambini_train]
        log.info("Fold %d: %d clip di adattamento (%d bambini di train)",
                 numero_fold, len(ammessi), len(bambini_train))
        encoder = adatta_encoder(argomenti.candidato, stato_base, ammessi,
                                 cache_pool, cfg, device)
        embedding_fold.append(embedding_clip(encoder, cache_probe, voci, device))

    # Probe con embedding PER FOLD: si riusa _probe_cv fold per fold a mano
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from babycry.probe import ece as calcola_ece

    def valuta(etichette: np.ndarray, nome: str) -> dict:
        """Metriche medie sui fold usando l'embedding adattato di ciascun fold."""
        # Le classi si calcolano SOLO sugli indici usati nei fold (la variante
        # a 3 classi lascia etichette vuote fuori dai fold filtrati)
        usati = np.concatenate([np.concatenate([tr, te]) for tr, te in fold_indici])
        classi = np.unique(etichette[usati])
        auc, bal, ecev = [], [], []
        for (train, test), X in zip(fold_indici, embedding_fold):
            modello = LogisticRegression(max_iter=2000, class_weight="balanced")
            modello.fit(X[train], etichette[train])
            prob = modello.predict_proba(X[test])
            piene = np.zeros((len(test), len(classi)))
            for j, c in enumerate(modello.classes_):
                piene[:, np.where(classi == c)[0][0]] = prob[:, j]
            try:
                auc.append(roc_auc_score(etichette[test], piene, multi_class="ovr",
                                         average="macro", labels=classi)
                           if len(classi) > 2 else
                           roc_auc_score(etichette[test], piene[:, 1]))
            except ValueError:
                auc.append(float("nan"))
            bal.append(balanced_accuracy_score(etichette[test],
                                               classi[piene.argmax(axis=1)]))
            ecev.append(calcola_ece(piene, np.searchsorted(classi, etichette[test])))
        log.info("%s — AUC %.4f ±%.4f  bal.acc %.4f  ECE %.4f", nome,
                 np.nanmean(auc), np.nanstd(auc), np.mean(bal), np.mean(ecev))
        return {"auc": np.nanmean(auc), "auc_std": np.nanstd(auc),
                "bal": np.mean(bal), "ece": np.mean(ecev)}

    m5 = valuta(y, "motivo 5 classi (adattato)")
    # Variante a 3 classi per confronto con Ubenwa: si tengono solo i clip
    # delle classi mappate e si rivaluta sugli stessi fold (indici filtrati)
    maschera3 = np.isin(y, list(TRE_CLASSI))
    y3 = np.array([TRE_CLASSI.get(c, "") for c in y])
    fold_indici3 = [(np.array([i for i in tr if maschera3[i]]),
                     np.array([i for i in te if maschera3[i]]))
                    for tr, te in fold_indici]
    fold_indici_originali = fold_indici
    fold_indici = fold_indici3
    m3 = valuta(y3, "motivo 3 classi alla Ubenwa (adattato)")
    fold_indici = fold_indici_originali

    percorso_risultati = RADICE / cfg["risultati"]
    nuovo = not percorso_risultati.exists()
    with open(percorso_risultati, "a", newline="", encoding="utf-8") as f:
        scrittore = csv.DictWriter(f, fieldnames=[
            "quando", "candidato", "passi_adattamento",
            "auc5", "auc5_std", "bal5", "ece5", "auc3", "auc3_std", "bal3"])
        if nuovo:
            scrittore.writeheader()
        scrittore.writerow({
            "quando": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "candidato": argomenti.candidato,
            "passi_adattamento": cfg["passi"],
            "auc5": f"{m5['auc']:.4f}", "auc5_std": f"{m5['auc_std']:.4f}",
            "bal5": f"{m5['bal']:.4f}", "ece5": f"{m5['ece']:.4f}",
            "auc3": f"{m3['auc']:.4f}", "auc3_std": f"{m3['auc_std']:.4f}",
            "bal3": f"{m3['bal']:.4f}"})
    log.info("Risultati accodati a %s", percorso_risultati)
    return 0


if __name__ == "__main__":
    sys.exit(main())
