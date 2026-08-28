"""Script 04 — Generazione delle varianti sintetiche di donateacry.

Per ogni clip reale genera varianti con due tecniche (vedi README):
- vocoder: F0 e formanti spostate (bambino di taglia diversa), timing identico;
- rumore: pianto originale + rumore domestico dal pool FSD50K a SNR variabile.

Obiettivi dalla config (configs/sintetici.yaml): ore per classe uguali per
tutte le classi (equi-bilanciamento) e quota vocoder/rumore. Output:
- data/sintetici/<classe>/*.wav
- data/sintetici_manifest.csv con origine, tecnica, parametri e, ereditati
  dal clip di origine, classe/contributore/eta/genere (REGOLA: il sintetico
  appartiene al bambino di origine ai fini dello split).

Lo script è ripristinabile: il manifest si scrive riga per riga e alla
ripartenza le ore già generate vengono scontate dal target. La generazione
serve al GATE D'ASCOLTO nell'app di esplorazione: il fine-tuning con questi
dati parte solo dopo l'approvazione a orecchio.

Uso:
    python scripts/04_genera_sintetici.py
    python scripts/04_genera_sintetici.py --solo-classe belly_pain
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

# Bootstrap del path: rende importabile src/babycry senza installare il pacchetto
RADICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RADICE / "src"))

from babycry.sintesi import (  # noqa: E402
    carica_audio_mono,
    scrivi_wav,
    variante_rumore,
    variante_vocoder,
)

log = logging.getLogger("04_sintetici")

COLONNE_MANIFEST = ["clip_id", "origine_clip_id", "tecnica", "classe",
                    "contributore", "eta", "genere", "percorso", "durata_s",
                    "parametri"]


def carica_pool_rumore(cfg: dict, manifest: list[dict]) -> list[Path]:
    """Costruisce il pool di rumore domestico dalle etichette ufficiali di FSD50K.

    Usa i CSV di ground truth (dev.csv, eval.csv) per selezionare le categorie
    di rumore della config (match per sottostringa) e interseca con i clip già
    ammessi dal filtro licenze — così il pool eredita la pulizia delle licenze.

    Args:
        cfg: config della generazione (liste rumore_includi / rumore_escludi).
        manifest: righe del manifest licenze.

    Returns:
        Lista dei percorsi dei clip di rumore utilizzabili.
    """
    includi = [s.lower() for s in cfg["rumore_includi"]]
    escludi = [s.lower() for s in cfg["rumore_escludi"]]

    # Mappa clip_id -> etichette dalle ground truth ufficiali
    etichette: dict[str, str] = {}
    for nome_csv in ("dev.csv", "eval.csv"):
        trovati = list((RADICE / "data" / "raw" / "fsd50k").glob(f"**/{nome_csv}"))
        if not trovati:
            log.warning("Ground truth FSD50K non trovata: %s", nome_csv)
            continue
        with open(trovati[0], newline="", encoding="utf-8") as f:
            for riga in csv.DictReader(f):
                etichette[riga["fname"]] = riga["labels"].lower()

    pool = []
    for riga in manifest:
        if riga["corpus"] != "fsd50k":
            continue
        lab = etichette.get(riga["clip_id"], "")
        if any(s in lab for s in escludi):
            continue
        if any(s in lab for s in includi):
            pool.append(RADICE / riga["percorso"])
    log.info("Pool di rumore: %d clip FSD50K (licenze già filtrate)", len(pool))
    return pool


def ore_gia_generate(percorso_manifest: Path) -> dict[tuple[str, str], float]:
    """Legge i secondi già generati per (classe, tecnica) dal manifest (resume)."""
    contatori: dict[tuple[str, str], float] = {}
    if not percorso_manifest.exists():
        return contatori
    with open(percorso_manifest, newline="", encoding="utf-8") as f:
        for riga in csv.DictReader(f):
            chiave = (riga["classe"], riga["tecnica"])
            contatori[chiave] = contatori.get(chiave, 0.0) + float(riga["durata_s"])
    return contatori


def genera_per_classe(classe: str, clip_reali: list[dict], tecnica: str,
                      secondi_da_fare: float, cfg: dict, pool_rumore: list[Path],
                      rng: np.random.Generator, scrittore: csv.DictWriter,
                      file_manifest) -> tuple[float, int]:
    """Genera varianti di una tecnica per una classe fino al target in secondi.

    Le varianti sono distribuite equamente sui clip reali della classe; ogni
    clip viene caricato una sola volta e le sue varianti generate in blocco.

    Args:
        classe: nome della classe donateacry.
        clip_reali: righe del manifest licenze della classe.
        tecnica: "vocoder" oppure "rumore".
        secondi_da_fare: secondi ancora mancanti al target per questa coppia.
        cfg: config della generazione.
        pool_rumore: percorsi dei clip di rumore (per la tecnica rumore).
        rng: generatore casuale seedato.
        scrittore: writer CSV del manifest sintetici (append riga per riga).
        file_manifest: file handle del manifest, per il flush a ogni riga.

    Returns:
        Coppia (secondi generati, errori).
    """
    sr = int(cfg["sample_rate"])
    cartella = RADICE / cfg["cartella_uscita"] / classe

    # Quante varianti per clip: target diviso per la durata complessiva reale
    import soundfile as sf
    durate = {c["clip_id"]: sf.info(RADICE / c["percorso"]).duration for c in clip_reali}
    durata_totale = sum(durate.values())
    per_clip = int(np.ceil(secondi_da_fare / max(durata_totale, 1e-9)))

    generati, errori = 0.0, 0
    barra = tqdm(total=int(secondi_da_fare), desc=f"{classe}/{tecnica}", unit="s")
    for voce in clip_reali:
        if generati >= secondi_da_fare:
            break
        audio = carica_audio_mono(RADICE / voce["percorso"], sr)
        for i in range(per_clip):
            if generati >= secondi_da_fare:
                break
            try:
                if tecnica == "vocoder":
                    # Modulo dello shift nei limiti config, segno casuale
                    semitoni = float(rng.uniform(cfg["pitch_semitoni_min"],
                                                 cfg["pitch_semitoni_max"]))
                    semitoni *= -1 if rng.random() < 0.5 else 1
                    alpha = float(rng.uniform(cfg["formanti_alpha_min"],
                                              cfg["formanti_alpha_max"]))
                    sintetico = variante_vocoder(audio, sr, semitoni, alpha,
                                                 cfg["f0_floor"], cfg["f0_ceil"])
                    parametri = f"pitch {semitoni:+.1f} st, formanti x{alpha:.2f}"
                else:
                    percorso_rumore = pool_rumore[int(rng.integers(len(pool_rumore)))]
                    rumore = carica_audio_mono(percorso_rumore, sr)
                    snr = float(rng.uniform(cfg["snr_db_min"], cfg["snr_db_max"]))
                    sintetico = variante_rumore(audio, rumore, snr, rng)
                    parametri = f"SNR {snr:.0f} dB, rumore {percorso_rumore.stem}"
            except ValueError as errore:
                # F0 non stimabile o rumore muto: si conta e si va avanti
                log.debug("Variante saltata (%s): %s", errore, voce["clip_id"])
                errori += 1
                continue

            nuovo_id = f"{voce['clip_id']}__{tecnica}_{i:03d}"
            percorso_wav = cartella / f"{nuovo_id}.wav"
            durata = scrivi_wav(percorso_wav, sintetico, sr)
            scrittore.writerow({
                "clip_id": nuovo_id,
                "origine_clip_id": voce["clip_id"],
                "tecnica": tecnica,
                "classe": classe,
                # Ereditati: il sintetico appartiene al bambino di origine
                "contributore": voce["contributore"],
                "eta": voce["eta"],
                "genere": voce["genere"],
                "percorso": percorso_wav.relative_to(RADICE).as_posix(),
                "durata_s": f"{durata:.3f}",
                "parametri": parametri,
            })
            file_manifest.flush()
            generati += durata
            barra.update(int(durata))
    barra.close()
    return generati, errori


def main() -> int:
    """Punto d'ingresso: legge le config, genera le varianti fino ai target."""
    parser = argparse.ArgumentParser(description="Generazione varianti sintetiche")
    parser.add_argument("--config", type=Path,
                        default=RADICE / "configs" / "sintetici.yaml")
    parser.add_argument("--manifest-licenze", type=Path,
                        default=RADICE / "data" / "licenses_manifest.csv")
    parser.add_argument("--solo-classe", help="genera solo per la classe indicata")
    argomenti = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    cfg = yaml.safe_load(argomenti.config.read_text(encoding="utf-8"))
    rng = np.random.default_rng(int(cfg["seed"]))

    with open(argomenti.manifest_licenze, newline="", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))

    # Clip reali di donateacry raggruppati per classe
    per_classe: dict[str, list[dict]] = {}
    for riga in manifest:
        if riga["corpus"] == "donateacry":
            per_classe.setdefault(riga["classe"], []).append(riga)
    if not per_classe:
        log.error("Nessun clip donateacry nel manifest: eseguire prima lo script 02")
        return 1

    pool_rumore = carica_pool_rumore(cfg, manifest)
    if not pool_rumore:
        log.error("Pool di rumore vuoto: FSD50K estratto? Ground truth presente?")
        return 1

    percorso_manifest = RADICE / cfg["manifest_uscita"]
    gia_fatto = ore_gia_generate(percorso_manifest)
    nuovo_manifest = not percorso_manifest.exists()

    target_sec = float(cfg["ore_target_per_classe"]) * 3600.0
    quota_voc = float(cfg["quota_vocoder"])

    with open(percorso_manifest, "a", newline="", encoding="utf-8") as f:
        scrittore = csv.DictWriter(f, fieldnames=COLONNE_MANIFEST)
        if nuovo_manifest:
            scrittore.writeheader()

        for classe, clip_reali in sorted(per_classe.items()):
            if argomenti.solo_classe and classe != argomenti.solo_classe:
                continue
            for tecnica, quota in (("vocoder", quota_voc), ("rumore", 1 - quota_voc)):
                mancanti = target_sec * quota - gia_fatto.get((classe, tecnica), 0.0)
                if mancanti <= 0:
                    log.info("%s/%s: target già raggiunto, salto", classe, tecnica)
                    continue
                log.info("%s/%s: da generare %.1f minuti", classe, tecnica, mancanti / 60)
                fatti, errori = genera_per_classe(classe, clip_reali, tecnica,
                                                  mancanti, cfg, pool_rumore, rng,
                                                  scrittore, f)
                log.info("%s/%s: generati %.1f minuti (%d varianti fallite)",
                         classe, tecnica, fatti / 60, errori)

    # Riepilogo finale dal manifest
    totali = ore_gia_generate(percorso_manifest)
    ore_totali = sum(totali.values()) / 3600
    log.info("=== Riepilogo sintetici: %.1f ore totali ===", ore_totali)
    for (classe, tecnica), secondi in sorted(totali.items()):
        log.info("  %-12s %-8s %6.2f ore", classe, tecnica, secondi / 3600)
    return 0


if __name__ == "__main__":
    sys.exit(main())
