"""Script 01 — Download dei corpora (FSD50K, VocalSound, donateacry).

Scarica i corpora definiti in configs/corpora.yaml con verifica checksum e
resume dei download interrotti. Freesound è previsto (slot per la API key)
ma disattivato finché la chiave non viene inserita nella config.

Uso:
    python scripts/01_download_corpora.py                 # scarica tutto
    python scripts/01_download_corpora.py --solo fsd50k   # un solo corpus
    python scripts/01_download_corpora.py --extract       # scarica ed estrae

Note:
- I checksum md5 di FSD50K arrivano dall'API di Zenodo (fonte ufficiale).
- Per VocalSound e donateacry la fonte non pubblica checksum: al primo
  download l'md5 viene calcolato e registrato in data/raw/checksums_registrati.json,
  e nei run successivi fa da riferimento.
- FSD50K è uno zip multi-volume: l'estrazione richiede 7-Zip nel PATH;
  in mancanza lo script stampa le istruzioni manuali.
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import yaml

# Bootstrap del path: rende importabile src/babycry senza installare il pacchetto
RADICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RADICE / "src"))

from babycry.download import (  # noqa: E402  (import dopo il bootstrap del path)
    leggi_checksum_registrato,
    lista_file_zenodo,
    md5_di_file,
    registra_checksum,
    scarica_con_resume,
)

log = logging.getLogger("01_download")


def scarica_zenodo(nome_corpus: str, cfg: dict, cartella: Path) -> bool:
    """Scarica tutti i file di un record Zenodo con i checksum ufficiali.

    Args:
        nome_corpus: nome del corpus (per log e cartella di destinazione).
        cfg: sezione del corpus in configs/corpora.yaml (chiave zenodo_record).
        cartella: cartella base dei dati grezzi (data/raw).

    Returns:
        True se tutti i file sono stati scaricati e verificati.
    """
    destinazione = cartella / nome_corpus
    esclusi = set(cfg.get("files_esclusi") or [])

    log.info("Interrogo l'API Zenodo per il record %s", cfg["zenodo_record"])
    file_record = lista_file_zenodo(int(cfg["zenodo_record"]))
    log.info("Il record contiene %d file", len(file_record))

    tutto_ok = True
    for voce in file_record:
        if voce["nome"] in esclusi:
            log.info("Escluso da config: %s", voce["nome"])
            continue
        ok = scarica_con_resume(voce["url"], destinazione / voce["nome"],
                                md5_atteso=voce["md5"])
        tutto_ok = tutto_ok and ok
    return tutto_ok


def scarica_http(nome_corpus: str, cfg: dict, cartella: Path, registro: Path) -> bool:
    """Scarica i file HTTP di un corpus, registrando l'md5 se la fonte non lo dà.

    Args:
        nome_corpus: nome del corpus.
        cfg: sezione del corpus in configs/corpora.yaml (lista `files`).
        cartella: cartella base dei dati grezzi (data/raw).
        registro: file JSON dove registrare gli md5 calcolati localmente.

    Returns:
        True se tutti i file sono disponibili e verificati (dove possibile).
    """
    destinazione = cartella / nome_corpus
    tutto_ok = True
    for voce in cfg["files"]:
        # Priorità: md5 dalla config; in mancanza, quello registrato al primo download
        md5 = voce.get("md5") or leggi_checksum_registrato(registro, voce["nome"])
        ok = scarica_con_resume(voce["url"], destinazione / voce["nome"], md5_atteso=md5)
        if ok and md5 is None:
            # Prima volta senza checksum: lo calcoliamo e lo registriamo per il futuro
            calcolato = md5_di_file(destinazione / voce["nome"], mostra_progresso=True)
            registra_checksum(registro, voce["nome"], calcolato)
            log.info("md5 registrato per %s: %s", voce["nome"], calcolato)
        tutto_ok = tutto_ok and ok
    return tutto_ok


def scarica_freesound(cfg: dict, cartella: Path, registro: Path) -> bool:
    """Scarica da Freesound i clip delle query con licenza CC0/CC BY.

    Per ogni query interroga /apiv2/search/text/ con filtro licenza e durata,
    scarica il preview HQ mp3 di ogni risultato (l'originale richiederebbe
    OAuth2 interattivo; il preview a 44.1 kHz basta per il log-mel a 16 kHz)
    e registra i metadati (autore, licenza, durata) per lo script 02.
    Deduplica sia tra query sia contro i clip FSD50K già in manifest
    (gli id sono gli stessi: FSD50K è un sottoinsieme di Freesound).

    Args:
        cfg: sezione freesound di configs/corpora.yaml.
        cartella: cartella base dei dati grezzi (data/raw).
        registro: registro JSON degli md5 calcolati localmente.

    Returns:
        True se il corpus è disponibile (o legittimamente saltato).
    """
    import time

    import requests

    if not cfg.get("enabled"):
        log.info("Freesound disattivato in config: salto.")
        return True
    percorso_chiave = RADICE / cfg.get("api_key_file", "")
    if not percorso_chiave.exists():
        log.warning("API key Freesound assente (%s): salto.", percorso_chiave)
        return True
    chiave = percorso_chiave.read_text(encoding="utf-8").strip()

    destinazione = cartella / "freesound"
    destinazione.mkdir(parents=True, exist_ok=True)
    cartella_audio = destinazione / "audio"
    percorso_metadati = destinazione / "freesound_metadati.csv"
    colonne = ["clip_id", "nome", "autore", "licenza", "durata_s", "query"]

    # Resume: metadati già raccolti nei run precedenti
    gia_noti: dict[str, dict] = {}
    if percorso_metadati.exists():
        with open(percorso_metadati, newline="", encoding="utf-8") as f:
            gia_noti = {r["clip_id"]: r for r in csv.DictReader(f)}

    # Dedup contro FSD50K: gli id Freesound coincidono con i clip_id del manifest
    ids_fsd50k: set[str] = set()
    manifest = RADICE / "data" / "licenses_manifest.csv"
    if manifest.exists():
        with open(manifest, newline="", encoding="utf-8") as f:
            ids_fsd50k = {r["clip_id"] for r in csv.DictReader(f)
                          if r["corpus"] == "fsd50k"}

    filtro = (f'license:("Creative Commons 0" OR "Attribution") '
              f'duration:[{cfg["durata_min_s"]} TO {cfg["durata_max_s"]}]')
    sessione = requests.Session()
    nuovi, scartati_fsd50k, errori = 0, 0, 0

    with open(percorso_metadati, "a", newline="", encoding="utf-8") as f:
        scrittore = csv.DictWriter(f, fieldnames=colonne)
        if not gia_noti:
            scrittore.writeheader()

        for query in cfg["query"]:
            log.info("Freesound, query: \"%s\"", query)
            url = "https://freesound.org/apiv2/search/text/"
            parametri = {"query": query, "token": chiave, "page_size": 150,
                         "filter": filtro, "sort": "downloads_desc",
                         "fields": "id,name,username,license,duration,previews"}
            raccolti = 0
            while url and raccolti < int(cfg["max_risultati_per_query"]):
                try:
                    risposta = sessione.get(url, params=parametri, timeout=60)
                    risposta.raise_for_status()
                except requests.RequestException as errore_rete:
                    log.warning("Ricerca interrotta (%s), riprovo al prossimo run",
                                errore_rete)
                    break
                dati = risposta.json()
                for suono in dati.get("results", []):
                    clip_id = str(suono["id"])
                    raccolti += 1
                    if clip_id in ids_fsd50k:
                        scartati_fsd50k += 1
                        continue
                    if clip_id in gia_noti:
                        continue
                    percorso_mp3 = cartella_audio / f"{clip_id}.mp3"
                    ok = scarica_con_resume(suono["previews"]["preview-hq-mp3"],
                                            percorso_mp3, md5_atteso=None,
                                            tentativi=3)
                    if not ok:
                        errori += 1
                        continue
                    riga = {"clip_id": clip_id, "nome": suono["name"],
                            "autore": suono["username"],
                            "licenza": suono["license"],
                            "durata_s": f"{float(suono['duration']):.2f}",
                            "query": query}
                    scrittore.writerow(riga)
                    f.flush()
                    gia_noti[clip_id] = riga
                    nuovi += 1
                    time.sleep(0.2)   # cortesia verso i rate limit dell'API
                # Paginazione: l'URL "next" contiene già query e filtro
                url = dati.get("next")
                parametri = {"token": chiave}

    log.info("Freesound: %d clip nuovi scaricati, %d totali noti, "
             "%d duplicati di FSD50K saltati, %d errori",
             nuovi, len(gia_noti), scartati_fsd50k, errori)
    return True


def estrai_archivi(cartella: Path) -> None:
    """Estrae gli archivi scaricati: zip (anche multi-volume) e tar.

    - Zip singoli (donateacry): zipfile della standard library.
    - Zip multi-volume (FSD50K dev/eval audio, parti .z01...): serve 7-Zip;
      se non è nel PATH vengono stampate le istruzioni manuali.
    - Tar (VocalSound dal mirror Zenodo in formato WebDataset): tarfile della
      standard library, estratti in una sottocartella `estratti/` accanto
      agli archivi.

    L'estrazione è idempotente: un archivio già estratto (marcatore
    `.estratto.ok` presente) viene saltato.
    """
    for archivio in sorted(cartella.rglob("*.zip")):
        # Le parti .z01/.z02/... non si estraggono da sole: si parte dal .zip finale
        multi_volume = archivio.with_suffix(".z01").exists()
        marcatore = archivio.with_suffix(".estratto.ok")
        if marcatore.exists():
            log.info("Già estratto, salto: %s", archivio.name)
            continue

        if multi_volume:
            # Zip diviso in volumi: solo 7-Zip li ricompone in estrazione.
            # Si cerca prima nel PATH, poi la copia portable in tools/7zip
            # (scaricata durante il setup, nessuna installazione di sistema).
            eseguibile = shutil.which("7z")
            portable = RADICE / "tools" / "7zip" / "7z.exe"
            if eseguibile is None and portable.exists():
                eseguibile = str(portable)
            if eseguibile is None:
                log.error(
                    "%s è un archivio multi-volume e 7-Zip non è nel PATH.\n"
                    "  Opzione 1: installare 7-Zip (winget install 7zip.7zip) e rilanciare.\n"
                    "  Opzione 2 (Git Bash): zip -s 0 %s --out unito.zip && unzip unito.zip",
                    archivio.name, archivio.name)
                continue
            log.info("Estraggo (7-Zip, multi-volume): %s", archivio.name)
            esito = subprocess.run([eseguibile, "x", "-y", str(archivio),
                                    f"-o{archivio.parent}"], capture_output=True, text=True)
            if esito.returncode != 0:
                log.error("7-Zip fallito su %s:\n%s", archivio.name, esito.stderr)
                continue
        else:
            log.info("Estraggo: %s", archivio.name)
            try:
                with zipfile.ZipFile(archivio) as zf:
                    zf.extractall(archivio.parent)
            except zipfile.BadZipFile:
                log.error("Archivio corrotto (riscaricare?): %s", archivio.name)
                continue

        marcatore.touch()

    # Archivi tar (WebDataset del mirror Zenodo di VocalSound): estrazione
    # nella sottocartella `estratti/`, che lo script 02 scandisce via rglob
    for archivio in sorted(cartella.rglob("*.tar")):
        marcatore = archivio.with_suffix(".estratto.ok")
        if marcatore.exists():
            log.info("Già estratto, salto: %s", archivio.name)
            continue
        log.info("Estraggo (tar): %s", archivio.name)
        try:
            with tarfile.open(archivio) as tf:
                # filter="data" blocca percorsi assoluti/traversal nel tar
                tf.extractall(archivio.parent / "estratti", filter="data")
        except tarfile.TarError:
            log.error("Tar corrotto (riscaricare?): %s", archivio.name)
            continue
        marcatore.touch()


def main() -> int:
    """Punto d'ingresso: legge la config, scarica i corpora abilitati, estrae se richiesto."""
    parser = argparse.ArgumentParser(description="Download dei corpora")
    parser.add_argument("--config", type=Path, default=RADICE / "configs" / "corpora.yaml",
                        help="percorso della config dei corpora")
    parser.add_argument("--solo", choices=["fsd50k", "vocalsound", "donateacry", "freesound"],
                        help="scarica solo il corpus indicato")
    parser.add_argument("--extract", action="store_true",
                        help="estrae gli archivi dopo il download")
    argomenti = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    cfg = yaml.safe_load(argomenti.config.read_text(encoding="utf-8"))
    cartella_dati = RADICE / cfg["cartella_dati"]
    cartella_dati.mkdir(parents=True, exist_ok=True)
    registro_checksum = cartella_dati / "checksums_registrati.json"

    esiti: dict[str, bool] = {}
    for nome, corpus in cfg["corpora"].items():
        if argomenti.solo and nome != argomenti.solo:
            continue
        if not corpus.get("enabled") and nome != "freesound":
            log.info("Corpus disabilitato in config: %s", nome)
            continue

        log.info("=== Corpus: %s ===", nome)
        if corpus["tipo"] == "zenodo":
            esiti[nome] = scarica_zenodo(nome, corpus, cartella_dati)
        elif corpus["tipo"] == "http":
            esiti[nome] = scarica_http(nome, corpus, cartella_dati, registro_checksum)
        elif corpus["tipo"] == "freesound_api":
            esiti[nome] = scarica_freesound(corpus, cartella_dati, registro_checksum)
        else:
            log.error("Tipo di corpus sconosciuto: %s", corpus["tipo"])
            esiti[nome] = False

    if argomenti.extract:
        estrai_archivi(cartella_dati)

    # Riepilogo finale ed exit code onesto (utile per concatenare gli script)
    log.info("=== Riepilogo download ===")
    for nome, ok in esiti.items():
        log.info("  %-12s %s", nome, "OK" if ok else "FALLITO")
    return 0 if all(esiti.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
