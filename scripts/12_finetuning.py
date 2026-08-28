"""Script 12 — Fine-tuning end-to-end di chiusura dello studio.

Addestra encoder+testa sul motivo (5 classi donateacry) con cross-validation
per bambino, in quattro bracci di ablation dei sintetici (vedi README):
"nessuno" (solo reali), "rumore", "vocoder", "entrambi". Regole anti-leakage:
- fold identici alle probe (StratifiedGroupKFold sul contributore);
- i sintetici entrano SOLO se il bambino di origine è nel train del fold;
- i sintetici derivati dai clip esclusi (spuri) restano fuori ovunque;
- la valutazione è SOLO su clip reali del fold di test.
Campionamento bilanciato per (classe × reale/sintetico), così le classi
minoritarie e i due tipi di dato pesano uguale.

L'esperimento chiude lo studio e misura il contributo dei sintetici.

Uso:
    python scripts/12_finetuning.py --sintetici entrambi
    python scripts/12_finetuning.py --sintetici nessuno --passi 30   # smoke
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

RADICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RADICE / "src"))

from babycry.dataset import CacheMel, VoceClip, finestre_clip  # noqa: E402
from babycry.encoder import EncoderCompatto  # noqa: E402
from babycry.probe import ece as calcola_ece  # noqa: E402

log = logging.getLogger("12_finetuning")

CLASSI = ["belly_pain", "burping", "discomfort", "hungry", "tired"]
TRE_CLASSI = ["belly_pain", "discomfort", "hungry"]


class DatasetEtichettato(Dataset):
    """Patch casuali con etichetta di classe (per il fine-tuning)."""

    def __init__(self, voci: list[tuple[VoceClip, int]], cache: CacheMel,
                 patch_frame: int = 300):
        self.voci = voci
        self.cache = cache
        self.patch_frame = patch_frame

    def __len__(self) -> int:
        return len(self.voci)

    def __getitem__(self, indice: int):
        voce, etichetta = self.voci[indice]
        inizio = int(np.random.randint(0, voce.n_frame - self.patch_frame + 1))
        patch = self.cache.patch(voce, inizio, self.patch_frame)
        return torch.from_numpy(patch).unsqueeze(0), etichetta


def addestra_fold(voci_train: list[tuple[VoceClip, int]], cache: CacheMel,
                  stato_encoder: dict, cfg: dict, device: str) -> nn.Module:
    """Fine-tuning end-to-end di encoder+testa su un fold."""
    encoder = EncoderCompatto().to(device)
    encoder.load_state_dict(stato_encoder)
    modello = nn.Sequential(encoder, nn.Linear(256, len(CLASSI))).to(device)

    # Pesi di campionamento: gruppi (classe, reale/sintetico) equiprobabili
    gruppi: dict[tuple, int] = {}
    chiavi = [(etichetta, voce.corpus == "sintetici") for voce, etichetta in voci_train]
    for chiave in chiavi:
        gruppi[chiave] = gruppi.get(chiave, 0) + 1
    pesi = [1.0 / gruppi[chiave] for chiave in chiavi]
    campionatore = WeightedRandomSampler(pesi, num_samples=len(pesi))
    loader = DataLoader(DatasetEtichettato(voci_train, cache),
                        batch_size=int(cfg["batch"]), sampler=campionatore,
                        num_workers=0, drop_last=True)

    ottimizzatore = torch.optim.AdamW(modello.parameters(), lr=float(cfg["lr"]),
                                      weight_decay=0.05)
    scaler = torch.amp.GradScaler("cuda")
    modello.train()
    passo, obiettivo = 0, int(cfg["passi"])
    while passo < obiettivo:
        for batch, etichette in loader:
            if passo >= obiettivo:
                break
            batch = batch.to(device, non_blocking=True)
            etichette = etichette.to(device)
            lr = float(cfg["lr"]) * 0.5 * (1 + math.cos(math.pi * passo / obiettivo))
            for g in ottimizzatore.param_groups:
                g["lr"] = lr
            ottimizzatore.zero_grad(set_to_none=True)
            with torch.autocast("cuda"):
                loss = F.cross_entropy(modello(batch), etichette)
            scaler.scale(loss).backward()
            scaler.step(ottimizzatore)
            scaler.update()
            passo += 1
    return modello


def predici_clip(modello: nn.Module, cache: CacheMel, voci: list[VoceClip],
                 device: str) -> np.ndarray:
    """Probabilità per clip: media dei softmax sulle finestre scorrevoli."""
    modello.eval()
    uscite = []
    with torch.no_grad():
        for voce in voci:
            finestre = finestre_clip(cache, voce).to(device)
            prob = torch.softmax(modello(finestre), dim=1).mean(dim=0)
            uscite.append(prob.cpu().numpy())
    return np.stack(uscite)


def main() -> int:
    """Fine-tuning con CV per bambino per un braccio dell'ablation."""
    parser = argparse.ArgumentParser(description="Fine-tuning di chiusura")
    parser.add_argument("--sintetici", required=True,
                        choices=["nessuno", "rumore", "vocoder", "entrambi"])
    parser.add_argument("--candidato", default="E",
                        help="encoder di partenza dal bake-off (default E)")
    parser.add_argument("--passi", type=int, default=1500)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--split", default="bambino", choices=["bambino", "clip"],
                        help="'clip' = split VOLUTAMENTE leaky per clip, come nella "
                             "letteratura che riporta 90%%+: serve a DIMOSTRARE il "
                             "leakage, mai a valutare davvero")
    argomenti = parser.parse_args()
    cfg = {"passi": argomenti.passi, "batch": argomenti.batch, "lr": argomenti.lr}

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(20260826)
    np.random.seed(20260826)
    device = "cuda"
    if not torch.cuda.is_available():
        log.error("CUDA non disponibile: interrompo (regola di progetto).")
        return 1

    stato_encoder = torch.load(
        RADICE / "checkpoints" / "bakeoff" / f"{argomenti.candidato}_encoder.pt",
        map_location=device, weights_only=True)["encoder"]

    # Clip reali (senza spuri) e sintetici (senza derivati degli spuri)
    esclusi = {r["clip_id"] for r in csv.DictReader(
        open(RADICE / "data" / "esclusioni_finetuning.csv", encoding="utf-8"))}
    tecnica_per_id = {r["clip_id"]: r["tecnica"] for r in csv.DictReader(
        open(RADICE / "data" / "sintetici_manifest.csv", encoding="utf-8"))}

    cache = CacheMel(RADICE, ["donateacry", "sintetici"])
    reali = [v for v in cache.clip if v.corpus == "donateacry"
             and v.clip_id not in esclusi]
    tecniche_ammesse = {"nessuno": set(), "rumore": {"rumore"},
                        "vocoder": {"vocoder"},
                        "entrambi": {"rumore", "vocoder"}}[argomenti.sintetici]
    sintetici = [v for v in cache.clip if v.corpus == "sintetici"
                 and v.clip_id.split("__")[0] not in esclusi
                 and tecnica_per_id.get(v.clip_id) in tecniche_ammesse]
    log.info("Braccio '%s': %d clip reali, %d sintetici disponibili",
             argomenti.sintetici, len(reali), len(sintetici))

    y = np.array([v.classe for v in reali])
    gruppi = np.array([v.contributore for v in reali])
    indice_classe = {c: i for i, c in enumerate(CLASSI)}

    if argomenti.split == "clip":
        # Split per clip: lo stesso bambino può stare in train e in test.
        # È il protocollo (sbagliato) della letteratura a 90%+: lo replichiamo
        # per misurare quanto vale il leakage, con accuratezza extra: i
        # sintetici entrano col criterio "bambino in train", che qui replica
        # anche l'errore augment-then-split (varianti del clip di test in train).
        from sklearn.model_selection import StratifiedKFold
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        divisioni = cv.split(np.zeros(len(y)), y)
    else:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
        divisioni = cv.split(np.zeros(len(y)), y, groups=gruppi)

    auc5, bal5, ece5, auc3, acc5 = [], [], [], [], []
    for numero_fold, (train, test) in enumerate(divisioni):
        bambini_train = set(gruppi[train])
        voci_train = [(reali[i], indice_classe[reali[i].classe]) for i in train]
        # Sintetici ammessi: solo se il bambino di origine sta nel train
        # (nello split per clip questo NON protegge: è il punto della simulazione)
        voci_train += [(v, indice_classe[v.classe]) for v in sintetici
                       if v.contributore in bambini_train]
        modello = addestra_fold(voci_train, cache, stato_encoder, cfg, device)

        voci_test = [reali[i] for i in test]
        prob = predici_clip(modello, cache, voci_test, device)
        y_test = y[test]
        try:
            auc5.append(roc_auc_score(y_test, prob, multi_class="ovr",
                                      average="macro", labels=CLASSI))
        except ValueError:
            auc5.append(np.nan)
        predette = np.array([CLASSI[i] for i in prob.argmax(axis=1)])
        bal5.append(balanced_accuracy_score(y_test, predette))
        # Accuracy semplice: e' la metrica riportata dalla letteratura a 90%+,
        # va confrontata con la baseline di maggioranza (84% "hungry")
        acc5.append(float((predette == y_test).mean()))
        ece5.append(calcola_ece(prob, np.searchsorted(CLASSI, y_test)))
        # Variante a 3 classi: si rinormalizzano le probabilità sulle 3
        m3 = np.isin(y_test, TRE_CLASSI)
        idx3 = [CLASSI.index(c) for c in TRE_CLASSI]
        if m3.sum() > 0 and len(np.unique(y_test[m3])) > 1:
            prob3 = prob[np.ix_(m3, idx3)]
            prob3 = prob3 / prob3.sum(axis=1, keepdims=True)
            try:
                auc3.append(roc_auc_score(y_test[m3], prob3, multi_class="ovr",
                                          average="macro", labels=TRE_CLASSI))
            except ValueError:
                auc3.append(np.nan)
        log.info("Fold %d completato (%d train con sintetici, %d test)",
                 numero_fold, len(voci_train), len(voci_test))

    riga = {"quando": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "candidato": argomenti.candidato, "sintetici": argomenti.sintetici,
            "split": argomenti.split, "passi": cfg["passi"],
            "auc5": f"{np.nanmean(auc5):.4f}", "auc5_std": f"{np.nanstd(auc5):.4f}",
            "bal5": f"{np.mean(bal5):.4f}", "acc5": f"{np.mean(acc5):.4f}",
            "ece5": f"{np.mean(ece5):.4f}",
            "auc3": f"{np.nanmean(auc3):.4f}", "auc3_std": f"{np.nanstd(auc3):.4f}"}
    # Le corse leaky vanno in un file separato: non devono mai mescolarsi
    # con i risultati validi
    nome_csv = ("finetuning_results.csv" if argomenti.split == "bambino"
                else "finetuning_leaky.csv")
    percorso = RADICE / "results" / nome_csv
    nuovo = not percorso.exists()
    with open(percorso, "a", newline="", encoding="utf-8") as f:
        scrittore = csv.DictWriter(f, fieldnames=list(riga.keys()))
        if nuovo:
            scrittore.writeheader()
        scrittore.writerow(riga)
    log.info("RISULTATO %s/%s: AUC5 %s ±%s  bal %s  acc %s  AUC3 %s",
             argomenti.sintetici, argomenti.split, riga["auc5"], riga["auc5_std"],
             riga["bal5"], riga["acc5"], riga["auc3"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
