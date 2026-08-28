"""Script 07 — Consolidamento di triage ed etichette umane in decisioni operative.

Applica le conseguenze dei voti umani (scheda "Casi limite" dell'app)
combinandoli con gli score PANNs. I voti umani SOVRASCRIVONO sempre il triage.
Produce quattro artefatti:

1. data/pool_pianto.csv — i clip del pool di pianto per l'adattamento di
   dominio SSL (donateacry depurato + Freesound/FSD50K approvati), con la
   fonte di ogni decisione (voto umano, soglia triage, corpus etichettato).
2. data/esclusioni_finetuning.csv — i clip donateacry giudicati spuri
   dall'orecchio umano: fuori sia dal fine-tuning sia dal pool SSL.
3. data/negativi_difficili.csv — pianti di adulti, versi di animali e
   clip con ground truth FSD50K smentita: i negativi per il detector.
4. results/calibrazione_triage.json — la soglia dello score combinato che
   meglio separa i voti umani pianto/non-pianto (per gli usi automatici).

Semantica dei voti : "pianto_neonato" = positivo; "pianto" nello
strato freesound_grigio e in donateacry equivale a pianto_neonato (la consegna
dell'app chiedeva di votare Pianto solo per neonato/bimbo); "pianto_adulto" e
"verso_animale" sono NEGATIVI per il detector; "non_sicuro" resta fuori da tutto.

Uso:
    python scripts/07_consolida_etichette.py
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np

RADICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RADICE / "src"))

log = logging.getLogger("07_consolida")

# Sopra questa soglia un clip non votato entra nel pool di pianto anche senza
# orecchio umano (ben sopra la zona grigia 0.15-0.65 giudicata a mano)
SOGLIA_AUTO_POOL = 0.65
GT_PIANTO = ("baby_cry", "crying_and_sobbing", "whimper")


def comb(riga: dict) -> float:
    """Score combinato di pianto: massimo tra neonatale, generico e lamento."""
    return max(float(riga["p_pianto_neonato"]), float(riga["p_pianto_generico"]),
               float(riga["p_lamento"]))


def carica_etichette_fsd50k() -> dict[str, str]:
    """Mappa clip_id -> etichette ground truth FSD50K (minuscole)."""
    etichette: dict[str, str] = {}
    for nome_csv in ("dev.csv", "eval.csv"):
        trovati = list((RADICE / "data" / "raw" / "fsd50k").glob(f"**/{nome_csv}"))
        if trovati:
            with open(trovati[0], newline="", encoding="utf-8") as f:
                for riga in csv.DictReader(f):
                    etichette[riga["fname"]] = riga["labels"].lower()
    return etichette


def main() -> int:
    """Consolida voti e triage nei quattro artefatti di decisione."""
    parser = argparse.ArgumentParser(description="Consolidamento etichette")
    argomenti = parser.parse_args()
    del argomenti  # nessun parametro: percorsi fissi di progetto

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    with open(RADICE / "data" / "licenses_manifest.csv", newline="",
              encoding="utf-8") as f:
        manifest = {r["clip_id"]: r for r in csv.DictReader(f)}
    with open(RADICE / "data" / "triage_panns.csv", newline="",
              encoding="utf-8") as f:
        triage = {r["clip_id"]: r for r in csv.DictReader(f)}
    with open(RADICE / "data" / "etichette_manuali.csv", newline="",
              encoding="utf-8") as f:
        voti = {r["clip_id"]: r["voto"] for r in csv.DictReader(f)}
    etichette_gt = carica_etichette_fsd50k()

    # Durate dalla cache mel (n_frame * hop 10 ms)
    durate: dict[str, float] = {}
    for indice in (RADICE / "cache" / "mel").glob("*_index.json"):
        dati = json.loads(indice.read_text(encoding="utf-8"))
        for cid, v in dati["clip"].items():
            durate[cid] = v["n_frame"] * 0.01

    # Normalizzazione dei voti alla semantica del detector
    POSITIVI = {"pianto_neonato", "pianto"}      # "pianto" = neonato (vedi docstring)
    NEGATIVI = {"non_pianto", "pianto_adulto", "verso_animale"}

    # --- 1. Pool di pianto per l'adattamento SSL -----------------------------
    pool: list[dict] = []
    def aggiungi_al_pool(cid: str, fonte: str) -> None:
        """Aggiunge un clip al pool con i metadati dal manifest."""
        m = manifest[cid]
        pool.append({"clip_id": cid, "corpus": m["corpus"], "percorso": m["percorso"],
                     "contributore": m["contributore"],
                     "durata_s": f"{durate.get(cid, 0):.2f}", "fonte_decisione": fonte})

    esclusi_donateacry: list[dict] = []
    for cid, m in manifest.items():
        voto = voti.get(cid)
        if m["corpus"] == "donateacry":
            if voto in NEGATIVI or voto == "non_sicuro":
                esclusi_donateacry.append({"clip_id": cid, "classe": m["classe"],
                                           "voto": voto})
            else:
                aggiungi_al_pool(cid, "voto_umano" if voto in POSITIVI
                                 else "corpus_etichettato")
        elif m["corpus"] == "freesound":
            if voto in POSITIVI:
                aggiungi_al_pool(cid, "voto_umano")
            elif voto is None and cid in triage and comb(triage[cid]) >= SOGLIA_AUTO_POOL:
                aggiungi_al_pool(cid, "soglia_triage")
            elif voto is None and cid in triage and comb(triage[cid]) >= 0.15:
                # Inferenza statistica (26/8): il campione
                # casuale dello strato grigio Freesound ha dato 20 pianti su 20,
                # quindi i non ascoltati dello stesso strato entrano nel pool
                # con flag dedicato (purezza attesa >90%, scelta reversibile).
                aggiungi_al_pool(cid, "inferenza_statistica")
        elif m["corpus"] == "fsd50k":
            gt_pianto = any(g in etichette_gt.get(cid, "") for g in GT_PIANTO)
            if voto in POSITIVI:
                aggiungi_al_pool(cid, "voto_umano")
            elif (voto is None and gt_pianto and cid in triage
                  and comb(triage[cid]) >= SOGLIA_AUTO_POOL):
                aggiungi_al_pool(cid, "gt_e_soglia_triage")

    # --- 2. Esclusioni dal fine-tuning (donateacry spuri) --------------------
    # --- 3. Negativi difficili per il detector -------------------------------
    negativi: list[dict] = []
    for cid, voto in voti.items():
        if cid not in manifest:
            continue
        m = manifest[cid]
        if voto == "pianto_adulto":
            categoria = "pianto_adulto"
        elif voto == "verso_animale":
            categoria = "verso_animale"
        elif (voto == "non_pianto" and m["corpus"] == "fsd50k"
              and any(g in etichette_gt.get(cid, "") for g in GT_PIANTO)):
            categoria = "gt_pianto_smentita"   # FSD50K lo dichiara pianto, l'orecchio no
        else:
            continue
        negativi.append({"clip_id": cid, "corpus": m["corpus"],
                         "percorso": m["percorso"], "categoria": categoria})

    # --- 4. Calibrazione della soglia sul giudizio umano ---------------------
    coppie = []
    for cid, voto in voti.items():
        if cid not in triage:
            continue
        if voto in POSITIVI:
            coppie.append((comb(triage[cid]), 1))
        elif voto in NEGATIVI:
            coppie.append((comb(triage[cid]), 0))
    punteggi = np.array([p for p, _ in coppie])
    veri = np.array([v for _, v in coppie])
    migliore = {"soglia": None, "balanced_accuracy": 0.0}
    for soglia in np.arange(0.05, 0.66, 0.01):
        previsti = (punteggi >= soglia).astype(int)
        sens = (previsti[veri == 1] == 1).mean() if (veri == 1).any() else 0.0
        spec = (previsti[veri == 0] == 0).mean() if (veri == 0).any() else 0.0
        ba = (sens + spec) / 2
        if ba > migliore["balanced_accuracy"]:
            migliore = {"soglia": round(float(soglia), 2),
                        "balanced_accuracy": round(float(ba), 3),
                        "sensibilita": round(float(sens), 3),
                        "specificita": round(float(spec), 3)}
    migliore["n_voti_usati"] = len(coppie)
    migliore["nota"] = ("Soglia sullo score combinato PANNs che meglio replica i "
                       "voti umani. ATTENZIONE: stimata su casi limite, quindi "
                       "pessimistica rispetto ai casi facili.")

    # --- scrittura degli artefatti -------------------------------------------
    def scrivi_csv(percorso: Path, righe: list[dict]) -> None:
        """Scrive un CSV con l'header dedotto dalla prima riga."""
        with open(percorso, "w", newline="", encoding="utf-8") as f:
            scrittore = csv.DictWriter(f, fieldnames=list(righe[0].keys()))
            scrittore.writeheader()
            scrittore.writerows(righe)

    scrivi_csv(RADICE / "data" / "pool_pianto.csv", pool)
    scrivi_csv(RADICE / "data" / "esclusioni_finetuning.csv", esclusi_donateacry)
    scrivi_csv(RADICE / "data" / "negativi_difficili.csv", negativi)
    (RADICE / "results").mkdir(exist_ok=True)
    (RADICE / "results" / "calibrazione_triage.json").write_text(
        json.dumps(migliore, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- riepilogo -----------------------------------------------------------
    ore_pool = sum(float(r["durata_s"]) for r in pool) / 3600
    per_corpus: dict[str, float] = {}
    for r in pool:
        per_corpus[r["corpus"]] = per_corpus.get(r["corpus"], 0) + float(r["durata_s"])
    log.info("Pool di pianto SSL: %d clip, %.2f ore totali", len(pool), ore_pool)
    for corpus, secondi in sorted(per_corpus.items()):
        log.info("  %-12s %6.2f ore", corpus, secondi / 3600)
    log.info("Esclusi dal fine-tuning (donateacry spuri): %d", len(esclusi_donateacry))
    log.info("Negativi difficili: %d (%s)", len(negativi),
             ", ".join(sorted({n['categoria'] for n in negativi})))
    log.info("Soglia calibrata: %s", migliore)
    return 0


if __name__ == "__main__":
    sys.exit(main())
