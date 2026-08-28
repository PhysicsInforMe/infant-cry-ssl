"""Script 09 — Arbitrato del bake-off: linear probe sugli encoder pre-addestrati.

Per ogni candidato con encoder salvato (script 08) esegue le due probe del
README — motivo del pianto su donateacry (split per bambino) e
pianto/non-pianto — e accoda i numeri a results/bakeoff_results.csv.
Include anche la baseline "random": l'encoder NON addestrato, che è il
termine di paragone onesto (se un pretext non batte i pesi casuali, non
ha imparato nulla di utile).

Uso:
    python scripts/09_bakeoff_probe.py                 # tutti i candidati trovati
    python scripts/09_bakeoff_probe.py --candidato C   # uno solo
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml

RADICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RADICE / "src"))

from babycry.encoder import EncoderCompatto  # noqa: E402
from babycry.probe import probe_motivo, probe_pianto  # noqa: E402

log = logging.getLogger("09_probe")


def valuta(nome: str, encoder: EncoderCompatto, device: str, uscita: Path) -> dict:
    """Esegue le due probe su un encoder e accoda la riga dei risultati."""
    log.info("=== Probe del candidato %s ===", nome)
    motivo = probe_motivo(encoder, RADICE, device)
    pianto = probe_pianto(encoder, RADICE, device)

    riga = {"quando": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "candidato": nome}
    for prefisso, metriche in (("motivo", motivo), ("pianto", pianto)):
        for chiave, (media, dev) in metriche.items():
            riga[f"{prefisso}_{chiave}"] = f"{media:.4f}"
            riga[f"{prefisso}_{chiave}_std"] = f"{dev:.4f}"

    nuovo = not uscita.exists()
    with open(uscita, "a", newline="", encoding="utf-8") as f:
        scrittore = csv.DictWriter(f, fieldnames=list(riga.keys()))
        if nuovo:
            scrittore.writeheader()
        scrittore.writerow(riga)

    log.info("%s — motivo: AUC %s  bal.acc %s  ECE %s | pianto: AUC %s",
             nome, riga["motivo_auc_macro"], riga["motivo_balanced_accuracy"],
             riga["motivo_ece"], riga["pianto_auc_macro"])
    return riga


def main() -> int:
    """Valuta gli encoder disponibili (più la baseline random)."""
    parser = argparse.ArgumentParser(description="Probe del bake-off")
    parser.add_argument("--candidato", help="valuta solo questo candidato")
    parser.add_argument("--config", type=Path,
                        default=RADICE / "configs" / "bakeoff.yaml")
    parser.add_argument("--salta-random", action="store_true",
                        help="non rivalutare la baseline random")
    argomenti = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(argomenti.config.read_text(encoding="utf-8"))
    device = cfg["device"] if torch.cuda.is_available() else "cpu"
    torch.manual_seed(int(cfg["seed"]))
    uscita = RADICE / cfg["risultati"]
    uscita.parent.mkdir(exist_ok=True)

    # Baseline: encoder a pesi casuali (stesso seed per riproducibilità)
    if not argomenti.salta_random and not argomenti.candidato:
        valuta("random", EncoderCompatto().to(device), device, uscita)

    cartella = RADICE / cfg["cartella_checkpoint"]
    for percorso in sorted(cartella.glob("*_encoder.pt")):
        nome = percorso.stem.split("_")[0]
        if argomenti.candidato and nome != argomenti.candidato:
            continue
        stato = torch.load(percorso, map_location=device, weights_only=True)
        encoder = EncoderCompatto().to(device)
        encoder.load_state_dict(stato["encoder"])
        valuta(nome, encoder, device, uscita)
    return 0


if __name__ == "__main__":
    sys.exit(main())
