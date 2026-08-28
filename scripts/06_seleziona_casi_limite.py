"""Script 06 — Selezione dei casi limite del triage per l'etichettatura manuale.

Dal CSV del triage (script 05) seleziona ~100 clip il cui voto umano ha il
massimo valore informativo, in quattro strati :

- conflitto_gt: FSD50K li dichiara pianto nella ground truth ma PANNs li
  boccia (score combinato basso) — qualcuno sbaglia, decide l'orecchio;
- scoperta: score alto senza etichetta di pianto ufficiale — candidati a
  entrare nel pool di pianto per l'adattamento di dominio, da verificare;
- zona_grigia: score combinato intermedio su FSD50K — servono a calibrare
  la soglia operativa del triage;
- rumore_sospetto / vocalsound_alto: sentinelle di contaminazione (pool di
  rumore dei sintetici e VocalSound).

Output: data/casi_limite.csv, letto dalla scheda "Casi limite" dell'app,
che salva i voti in data/etichette_manuali.csv. I voti umani sovrascrivono
PANNs in ogni uso successivo.

Uso:
    python scripts/06_seleziona_casi_limite.py
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

RADICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RADICE / "src"))

log = logging.getLogger("06_casi_limite")

# Numerosità per strato (totale ~100, il budget d'ascolto concordato).
# Quote ricalibrate sui numeri reali del triage del 25/8: le "scoperte" a
# soglia 0.5 sono risultate zero (FSD50K è etichettato a tappeto), quindi il
# budget va sui conflitti con la ground truth (78 candidati) e sulla zona
# grigia; le sentinelle hanno pochi candidati (4 rumore, 3 VocalSound).
QUOTE = {"conflitto_gt": 40, "scoperta": 10, "zona_grigia": 40,
         "rumore_sospetto": 10, "vocalsound_alto": 10,
         # Strati di validazione (aggiunti in seconda battuta): voto atteso
         # dichiarato nell'app, ma modificabile all'ascolto.
         "validazione_pianto_alto": 12,   # FSD50K con score alto: conferma
         "donateacry_dubbio": 15,         # pianti "veri" che PANNs boccia
         # Secondo giro (26/8): clip Freesound in zona grigia, per decidere
         # a orecchio quali entrano nel pool di pianto dell'adattamento SSL.
         "freesound_grigio": 20}
# Etichette FSD50K che contano come "pianto dichiarato" nella ground truth
GT_PIANTO = ("baby_cry", "crying_and_sobbing", "whimper")
SEED = 20260825


def carica_etichette_fsd50k() -> dict[str, str]:
    """Mappa clip_id -> etichette ground truth di FSD50K (minuscole)."""
    etichette: dict[str, str] = {}
    for nome_csv in ("dev.csv", "eval.csv"):
        trovati = list((RADICE / "data" / "raw" / "fsd50k").glob(f"**/{nome_csv}"))
        if trovati:
            with open(trovati[0], newline="", encoding="utf-8") as f:
                for riga in csv.DictReader(f):
                    etichette[riga["fname"]] = riga["labels"].lower()
    return etichette


def main() -> int:
    """Seleziona i casi limite e scrive data/casi_limite.csv."""
    parser = argparse.ArgumentParser(description="Selezione casi limite triage")
    parser.add_argument("--triage", type=Path, default=RADICE / "data" / "triage_panns.csv")
    parser.add_argument("--manifest", type=Path,
                        default=RADICE / "data" / "licenses_manifest.csv")
    parser.add_argument("--uscita", type=Path, default=RADICE / "data" / "casi_limite.csv")
    argomenti = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    rng = np.random.default_rng(SEED)

    with open(argomenti.manifest, newline="", encoding="utf-8") as f:
        percorsi = {r["clip_id"]: r["percorso"] for r in csv.DictReader(f)}
    with open(argomenti.triage, newline="", encoding="utf-8") as f:
        triage = list(csv.DictReader(f))
    etichette_gt = carica_etichette_fsd50k()

    # Categorie del pool di rumore (le stesse dello script 04)
    cfg_sint = yaml.safe_load((RADICE / "configs" / "sintetici.yaml")
                              .read_text(encoding="utf-8"))
    includi = [s.lower() for s in cfg_sint["rumore_includi"]]
    escludi = [s.lower() for s in cfg_sint["rumore_escludi"]]

    def comb(r: dict) -> float:
        """Score combinato di pianto: massimo tra neonatale, generico e lamento."""
        return max(float(r["p_pianto_neonato"]), float(r["p_pianto_generico"]),
                   float(r["p_lamento"]))

    strati: dict[str, list[dict]] = {s: [] for s in QUOTE}
    for r in triage:
        punteggio = comb(r)
        if r["corpus"] == "fsd50k":
            lab = etichette_gt.get(r["clip_id"], "")
            gt_pianto = any(g in lab for g in GT_PIANTO)
            nel_pool_rumore = (not any(s in lab for s in escludi)
                               and any(s in lab for s in includi))
            if gt_pianto and punteggio < 0.3:
                strati["conflitto_gt"].append(r)
            elif not gt_pianto and punteggio >= 0.5:
                strati["scoperta"].append(r)
            elif not gt_pianto and 0.2 <= punteggio < 0.5:
                strati["zona_grigia"].append(r)
            # Le sentinelle del rumore possono sovrapporsi alla zona grigia:
            # prevale lo strato più specifico
            if nel_pool_rumore and punteggio >= 0.15:
                strati["rumore_sospetto"].append(r)
            if float(r["p_pianto_neonato"]) >= 0.5:
                strati["validazione_pianto_alto"].append(r)
        elif r["corpus"] == "vocalsound" and punteggio >= 0.3:
            strati["vocalsound_alto"].append(r)
        elif r["corpus"] == "donateacry":
            # Tutti i donateacry sono candidati: in selezione si prendono
            # i punteggi PIU' BASSI (i "pianti" che PANNs boccia)
            strati["donateacry_dubbio"].append(r)
        elif r["corpus"] == "freesound" and 0.15 <= punteggio < 0.65:
            strati["freesound_grigio"].append(r)

    # Campionamento per strato: i più ambigui prima dove ha senso, casuale
    # seedato nella zona grigia (che è grande e va coperta uniformemente)
    # Durate dalla cache mel: i clip sotto i 2 secondi sono troppo corti per
    # un giudizio a orecchio affidabile (verifica all'ascolto) e si
    # escludono dagli strati di giudizio (non dalle sentinelle, dove anche
    # un clip corto contaminato va comunque visto).
    import json
    durate: dict[str, float] = {}
    for indice in (RADICE / "cache" / "mel").glob("*_index.json"):
        dati_indice = json.loads(indice.read_text(encoding="utf-8"))
        for cid, v in dati_indice["clip"].items():
            durate[cid] = v["n_frame"] * 0.01
    STRATI_SENZA_FILTRO_DURATA = ("rumore_sospetto", "vocalsound_alto")

    # Modalità append: i casi già selezionati (e magari già votati) restano
    # INTATTI; si aggiungono solo gli strati non ancora presenti nel file.
    selezione: list[dict] = []
    gia_scelti: set[str] = set()
    strati_presenti: set[str] = set()
    if argomenti.uscita.exists():
        with open(argomenti.uscita, newline="", encoding="utf-8") as f:
            for riga_esistente in csv.DictReader(f):
                selezione.append(riga_esistente)
                gia_scelti.add(riga_esistente["clip_id"])
                strati_presenti.add(riga_esistente["strato"])

    for strato, quota in QUOTE.items():
        if strato in strati_presenti:
            log.info("%-16s già nel file (%d casi tenuti), salto", strato,
                     sum(1 for s in selezione if s["strato"] == strato))
            continue
        candidati = [c for c in strati[strato] if c["clip_id"] not in gia_scelti]
        if strato not in STRATI_SENZA_FILTRO_DURATA:
            candidati = [c for c in candidati if durate.get(c["clip_id"], 99.0) >= 2.0]
        if strato in ("zona_grigia", "freesound_grigio"):
            rng.shuffle(candidati)
        elif strato == "donateacry_dubbio":
            candidati.sort(key=comb)                 # gli score più BASSI prima
        else:
            candidati.sort(key=comb, reverse=True)   # gli score più alti prima
        for c in candidati[:quota]:
            gia_scelti.add(c["clip_id"])
            selezione.append({
                "clip_id": c["clip_id"],
                "corpus": c["corpus"],
                "percorso": percorsi.get(c["clip_id"], ""),
                "strato": strato,
                "p_comb": f"{comb(c):.4f}",
                "p_pianto_neonato": c["p_pianto_neonato"],
                "top_etichetta": c["top_etichetta"],
            })
        log.info("%-16s %3d candidati, %2d selezionati",
                 strato, len(strati[strato]), min(quota, len(candidati)))

    with open(argomenti.uscita, "w", newline="", encoding="utf-8") as f:
        scrittore = csv.DictWriter(f, fieldnames=list(selezione[0].keys()))
        scrittore.writeheader()
        scrittore.writerows(selezione)
    log.info("Scritti %d casi limite in %s — etichettali nella scheda "
             "'Casi limite' dell'app", len(selezione), argomenti.uscita)
    return 0


if __name__ == "__main__":
    sys.exit(main())
