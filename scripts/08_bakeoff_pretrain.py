"""Script 08 — Pretraining di un candidato del bake-off (vedi README).

Addestra l'encoder compatto col pretext task del candidato scelto (A-F),
stesso budget per tutti: la GPU è obbligatoria, AMP sempre attivo, checkpoint
frequenti con resume (siamo su un laptop). Il checkpoint finale contiene solo
l'encoder (le teste dei pretext si buttano via).

Uso:
    python scripts/08_bakeoff_pretrain.py --candidato C
    python scripts/08_bakeoff_pretrain.py --candidato A --passi 100   # smoke test
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

RADICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RADICE / "src"))

from babycry.dataset import CacheMel, DatasetPatchCasuali  # noqa: E402
from babycry.encoder import EncoderCompatto, conta_parametri  # noqa: E402
from babycry.pretext import CANDIDATI  # noqa: E402

log = logging.getLogger("08_bakeoff")


def fattore_lr(passo: int, warmup: int, totale: int) -> float:
    """Schedula del learning rate: rampa lineare poi coseno fino a zero."""
    if passo < warmup:
        return passo / max(warmup, 1)
    avanzamento = (passo - warmup) / max(totale - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * avanzamento))


def main() -> int:
    """Pretraining di un candidato con budget e logistica dalla config."""
    parser = argparse.ArgumentParser(description="Pretraining bake-off")
    parser.add_argument("--candidato", required=True, choices=sorted(CANDIDATI))
    parser.add_argument("--config", type=Path,
                        default=RADICE / "configs" / "bakeoff.yaml")
    parser.add_argument("--passi", type=int, help="override del budget (smoke test)")
    argomenti = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(argomenti.config.read_text(encoding="utf-8"))
    torch.manual_seed(int(cfg["seed"]))

    if not torch.cuda.is_available():
        log.error("CUDA non disponibile: il training gira SEMPRE sulla GPU "
                  "(regola di progetto). Interrompo.")
        return 1
    device = cfg["device"]
    passi_totali = argomenti.passi or int(cfg["passi_totali"])

    # Dati: patch casuali dal pool generico, via memmap
    cache = CacheMel(RADICE, cfg["corpora_pretraining"],
                     min_frame=int(cfg["patch_frame"]))
    dataset = DatasetPatchCasuali(cache, int(cfg["patch_frame"]), int(cfg["seed"]))
    loader = DataLoader(dataset, batch_size=int(cfg["batch"]), shuffle=True,
                        num_workers=0, drop_last=True)
    log.info("Pool di pretraining: %d clip utilizzabili", len(dataset))

    # Modello: encoder condiviso + testa del candidato
    encoder = EncoderCompatto().to(device)
    modello = CANDIDATI[argomenti.candidato](encoder).to(device)
    log.info("Candidato %s: %d parametri encoder, %d totali (teste incluse)",
             argomenti.candidato, conta_parametri(encoder), conta_parametri(modello))

    ottimizzatore = torch.optim.AdamW(modello.parameters(), lr=float(cfg["lr"]),
                                      weight_decay=float(cfg["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda")

    # Resume dal checkpoint se esiste
    cartella_ckpt = RADICE / cfg["cartella_checkpoint"]
    cartella_ckpt.mkdir(parents=True, exist_ok=True)
    percorso_ckpt = cartella_ckpt / f"{argomenti.candidato}_stato.pt"
    passo = 0
    if percorso_ckpt.exists():
        stato = torch.load(percorso_ckpt, map_location=device, weights_only=True)
        modello.load_state_dict(stato["modello"])
        ottimizzatore.load_state_dict(stato["ottimizzatore"])
        scaler.load_state_dict(stato["scaler"])
        passo = int(stato["passo"])
        log.info("Resume dal passo %d", passo)

    modello.train()
    inizio_run, somma_loss, conteggio = time.time(), 0.0, 0
    while passo < passi_totali:
        for batch in loader:
            if passo >= passi_totali:
                break
            batch = batch.to(device, non_blocking=True)
            lr = float(cfg["lr"]) * fattore_lr(passo, int(cfg["warmup_passi"]),
                                               passi_totali)
            for gruppo in ottimizzatore.param_groups:
                gruppo["lr"] = lr
            ottimizzatore.zero_grad(set_to_none=True)
            with torch.autocast("cuda"):
                loss = modello.passo(batch)
            scaler.scale(loss).backward()
            scaler.step(ottimizzatore)
            scaler.update()

            passo += 1
            somma_loss += float(loss)
            conteggio += 1
            if passo % int(cfg["log_ogni"]) == 0:
                velocita = passo / max(time.time() - inizio_run, 1)
                log.info("passo %d/%d  loss %.4f  lr %.2e  (%.1f passi/s)",
                         passo, passi_totali, somma_loss / conteggio, lr, velocita)
                somma_loss, conteggio = 0.0, 0
            if passo % int(cfg["checkpoint_ogni"]) == 0:
                torch.save({"modello": modello.state_dict(),
                            "ottimizzatore": ottimizzatore.state_dict(),
                            "scaler": scaler.state_dict(), "passo": passo},
                           percorso_ckpt)

    # Checkpoint finale: SOLO l'encoder (le teste dei pretext si scartano)
    finale = cartella_ckpt / f"{argomenti.candidato}_encoder.pt"
    torch.save({"encoder": encoder.state_dict(),
                "candidato": argomenti.candidato, "passi": passo}, finale)
    log.info("Fatto: encoder salvato in %s (%.1f minuti)",
             finale, (time.time() - inizio_run) / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
