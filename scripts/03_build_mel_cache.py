"""Script 03 — Cache log-mel float16 memory-mapped con indice dei metadati.

Legge data/licenses_manifest.csv (script 02) e per ogni corpus produce:
- cache/mel/<corpus>.dat: log-mel di tutti i clip concatenati sull'asse
  temporale, float16, forma logica [n_frame_totali, 64], letto in training
  via np.memmap (mai caricato intero in RAM: vincolo dei 16 GB);
- cache/mel/<corpus>_index.json: indice dei metadati con, per clip,
  offset e numero di frame nel .dat, classe, contributore e statistiche
  (media/dev_std) per la normalizzazione per clip al load.

Front-end (README): 64 mel, 16 kHz, finestra 25 ms, hop 10 ms.
L'estrazione gira su CPU. Lo script è ripristinabile: i clip già in indice
vengono saltati e il .dat viene esteso in append.

Uso:
    python scripts/03_build_mel_cache.py
    python scripts/03_build_mel_cache.py --solo donateacry
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

# Bootstrap del path: rende importabile src/babycry senza installare il pacchetto
RADICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RADICE / "src"))

from babycry.mel import FrontendLogMel  # noqa: E402

log = logging.getLogger("03_mel")


def carica_indice(percorso: Path) -> dict:
    """Carica l'indice JSON di un corpus, o ne crea uno vuoto se assente.

    Struttura: {"n_frame_totali": int, "n_mels": int, "clip": {clip_id: voce}}
    dove ogni voce ha offset_frame, n_frame, media, dev_std, classe,
    contributore e percorso di origine.
    """
    if percorso.exists():
        return json.loads(percorso.read_text(encoding="utf-8"))
    return {"n_frame_totali": 0, "n_mels": None, "clip": {}}


def salva_indice(percorso: Path, indice: dict) -> None:
    """Salva l'indice JSON in modo atomico (scrittura su file temporaneo + rename).

    L'atomicità protegge la coerenza indice/.dat se lo script viene interrotto
    a metà (laptop: sospensioni e interruzioni sono da mettere in conto).
    """
    temporaneo = percorso.with_suffix(".tmp")
    temporaneo.write_text(json.dumps(indice, indent=1, ensure_ascii=False),
                          encoding="utf-8")
    temporaneo.replace(percorso)


def costruisci_cache_corpus(nome_corpus: str, clip: list[dict], frontend: FrontendLogMel,
                            cartella_cache: Path, salva_ogni: int = 200) -> None:
    """Costruisce (o estende) la cache log-mel di un corpus.

    Args:
        nome_corpus: nome del corpus (determina i nomi dei file di cache).
        clip: righe del manifest relative al corpus.
        frontend: estrattore log-mel configurato.
        cartella_cache: cartella di destinazione (cache/mel).
        salva_ogni: ogni quanti clip salvare l'indice (checkpoint frequenti).
    """
    percorso_dat = cartella_cache / f"{nome_corpus}.dat"
    percorso_indice = cartella_cache / f"{nome_corpus}_index.json"
    indice = carica_indice(percorso_indice)
    indice["n_mels"] = frontend.n_mels

    # Resume: si processano solo i clip non ancora in indice
    da_fare = [c for c in clip if c["clip_id"] not in indice["clip"]]
    log.info("%s: %d clip totali, %d già in cache, %d da processare",
             nome_corpus, len(clip), len(clip) - len(da_fare), len(da_fare))
    if not da_fare:
        return

    errori = 0
    # Append binario: l'offset di ogni nuovo clip è la fine attuale del .dat
    with open(percorso_dat, "ab") as dat:
        for numero, voce in enumerate(tqdm(da_fare, desc=nome_corpus, unit="clip")):
            percorso_audio = RADICE / voce["percorso"]
            try:
                logmel, media, dev_std = frontend.da_file(percorso_audio)
            except Exception as errore:  # file corrotti/troppo corti: si contano e si va avanti
                log.warning("Clip saltato (%s): %s", errore, voce["percorso"])
                errori += 1
                continue

            dat.write(logmel.tobytes())
            indice["clip"][voce["clip_id"]] = {
                "corpus": nome_corpus,
                "percorso": voce["percorso"],
                "offset_frame": indice["n_frame_totali"],
                "n_frame": int(logmel.shape[0]),
                "media": media,
                "dev_std": dev_std,
                "classe": voce.get("classe", ""),
                "contributore": voce.get("contributore", ""),
                "eta": voce.get("eta", ""),
                "genere": voce.get("genere", ""),
            }
            indice["n_frame_totali"] += int(logmel.shape[0])

            # Checkpoint frequenti: flush del .dat e salvataggio dell'indice
            if (numero + 1) % salva_ogni == 0:
                dat.flush()
                salva_indice(percorso_indice, indice)

        dat.flush()
    salva_indice(percorso_indice, indice)
    log.info("%s: cache completata, %d frame totali, %d clip in indice, %d errori",
             nome_corpus, indice["n_frame_totali"], len(indice["clip"]), errori)


def main() -> int:
    """Punto d'ingresso: legge manifest e config del front-end, costruisce le cache."""
    parser = argparse.ArgumentParser(description="Cache log-mel float16 memory-mapped")
    parser.add_argument("--manifest", type=Path,
                        default=RADICE / "data" / "licenses_manifest.csv",
                        help="manifest delle licenze prodotto dallo script 02")
    parser.add_argument("--config", type=Path,
                        default=RADICE / "configs" / "audio_frontend.yaml",
                        help="config del front-end audio")
    parser.add_argument("--solo", help="processa solo il corpus indicato")
    argomenti = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if not argomenti.manifest.exists():
        log.error("Manifest non trovato: %s (eseguire prima lo script 02)",
                  argomenti.manifest)
        return 1

    cfg = yaml.safe_load(argomenti.config.read_text(encoding="utf-8"))
    frontend = FrontendLogMel(cfg)
    cartella_cache = RADICE / cfg["cartella_cache"]
    cartella_cache.mkdir(parents=True, exist_ok=True)

    # Raggruppa le righe del manifest per corpus. Il manifest dei sintetici
    # (script 04) non ha la colonna corpus: quelle righe vanno sotto "sintetici".
    per_corpus: dict[str, list[dict]] = {}
    with open(argomenti.manifest, newline="", encoding="utf-8") as f:
        for riga in csv.DictReader(f):
            per_corpus.setdefault(riga.get("corpus") or "sintetici", []).append(riga)

    for nome_corpus, clip in per_corpus.items():
        if argomenti.solo and nome_corpus != argomenti.solo:
            continue
        costruisci_cache_corpus(nome_corpus, clip, frontend, cartella_cache)

    log.info("Cache in %s — lettura in training con babycry.mel.apri_cache_memmap "
             "(np.memmap, mai tutto in RAM)", cartella_cache)
    return 0


if __name__ == "__main__":
    sys.exit(main())
