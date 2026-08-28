"""Utility di download per i corpora: HTTP con resume, verifica md5, API Zenodo.

Il download è ripristinabile: i byte parziali finiscono in un file `.part`
e alla ripartenza si riprende dal punto di interruzione con una richiesta
HTTP Range. A verifica md5 superata il `.part` viene rinominato nel nome
definitivo e viene scritto un marcatore `.md5.ok` per non riverificare
file enormi a ogni esecuzione.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import requests
from tqdm import tqdm

log = logging.getLogger(__name__)

# Dimensione dei blocchi di lettura/scrittura (1 MiB: buon compromesso I/O)
_CHUNK = 1024 * 1024


def md5_di_file(percorso: Path, mostra_progresso: bool = False) -> str:
    """Calcola l'md5 di un file leggendolo a blocchi (mai tutto in RAM).

    Args:
        percorso: file di cui calcolare il checksum.
        mostra_progresso: se True mostra una barra tqdm (utile sui file grandi).

    Returns:
        Digest md5 esadecimale in minuscolo.
    """
    h = hashlib.md5()
    dimensione = percorso.stat().st_size
    barra = tqdm(total=dimensione, unit="B", unit_scale=True,
                 desc=f"md5 {percorso.name}", disable=not mostra_progresso)
    with open(percorso, "rb") as f:
        while blocco := f.read(_CHUNK):
            h.update(blocco)
            barra.update(len(blocco))
    barra.close()
    return h.hexdigest()


def _marcatore_ok(destinazione: Path) -> Path:
    """Percorso del file marcatore che attesta la verifica md5 già superata."""
    return destinazione.with_suffix(destinazione.suffix + ".md5.ok")


def scarica_con_resume(url: str, destinazione: Path, md5_atteso: str | None = None,
                       tentativi: int = 5, timeout: int = 60) -> bool:
    """Scarica `url` in `destinazione` con resume e verifica md5 opzionale.

    Comportamento:
    - se il file definitivo esiste ed è già stato verificato (marcatore .md5.ok
      o md5 non richiesto), non fa nulla;
    - se esiste un `.part`, riprende dal byte successivo con header Range;
    - a fine download verifica l'md5 (se fornito) e rinomina il `.part`.

    Args:
        url: URL sorgente.
        destinazione: percorso finale del file.
        md5_atteso: digest md5 atteso; None = nessuna verifica possibile
            (il chiamante può registrare il digest calcolato per il futuro).
        tentativi: numero massimo di tentativi in caso di errori di rete.
        timeout: timeout in secondi delle richieste HTTP.

    Returns:
        True se il file è disponibile e (dove possibile) verificato.
    """
    destinazione.parent.mkdir(parents=True, exist_ok=True)
    parziale = destinazione.with_suffix(destinazione.suffix + ".part")

    # File già presente: evita di riscaricare e, se serve, riverifica una volta sola.
    if destinazione.exists():
        if md5_atteso is None or _marcatore_ok(destinazione).exists():
            log.info("Già presente, salto: %s", destinazione.name)
            return True
        log.info("Verifica md5 del file esistente: %s", destinazione.name)
        if md5_di_file(destinazione, mostra_progresso=True) == md5_atteso.lower():
            _marcatore_ok(destinazione).touch()
            return True
        log.warning("md5 errato per %s: riscarico da zero", destinazione.name)
        destinazione.unlink()

    sessione = requests.Session()
    for tentativo in range(1, tentativi + 1):
        try:
            posizione = parziale.stat().st_size if parziale.exists() else 0
            intestazioni = {"Range": f"bytes={posizione}-"} if posizione else {}
            risposta = sessione.get(url, headers=intestazioni, stream=True,
                                    timeout=timeout, allow_redirects=True)

            if posizione and risposta.status_code == 200:
                # Il server ha ignorato il Range: si riparte da zero.
                log.warning("Il server non supporta il resume, riparto da zero: %s", url)
                posizione = 0
            elif posizione and risposta.status_code == 206:
                log.info("Riprendo %s dal byte %d", destinazione.name, posizione)
            risposta.raise_for_status()

            # Dimensione totale per la barra (Content-Range se in resume)
            totale = None
            if "Content-Range" in risposta.headers:
                totale = int(risposta.headers["Content-Range"].split("/")[-1])
            elif "Content-Length" in risposta.headers:
                totale = posizione + int(risposta.headers["Content-Length"])

            modalita = "ab" if posizione else "wb"
            with open(parziale, modalita) as f, tqdm(
                total=totale, initial=posizione, unit="B", unit_scale=True,
                desc=destinazione.name,
            ) as barra:
                for blocco in risposta.iter_content(chunk_size=_CHUNK):
                    f.write(blocco)
                    barra.update(len(blocco))

            # Verifica del checksum prima di dichiarare il file valido
            if md5_atteso is not None:
                log.info("Verifica md5 di %s", destinazione.name)
                calcolato = md5_di_file(parziale, mostra_progresso=True)
                if calcolato != md5_atteso.lower():
                    log.error("md5 errato per %s (atteso %s, calcolato %s): "
                              "elimino il parziale e ritento",
                              destinazione.name, md5_atteso, calcolato)
                    parziale.unlink()
                    continue

            parziale.replace(destinazione)
            if md5_atteso is not None:
                _marcatore_ok(destinazione).touch()
            log.info("Completato: %s", destinazione.name)
            return True

        except (requests.RequestException, OSError) as errore:
            attesa = min(2 ** tentativo, 60)
            log.warning("Tentativo %d/%d fallito per %s (%s), riprovo tra %d s",
                        tentativo, tentativi, destinazione.name, errore, attesa)
            time.sleep(attesa)

    log.error("Download fallito dopo %d tentativi: %s", tentativi, url)
    return False


def lista_file_zenodo(record_id: int, timeout: int = 60) -> list[dict]:
    """Interroga l'API di Zenodo e restituisce i file di un record con i checksum.

    I checksum md5 ufficiali arrivano così direttamente dalla fonte, senza
    doverli mantenere a mano nella config.

    Args:
        record_id: id numerico del record Zenodo (es. 4060432 per FSD50K).

    Returns:
        Lista di dict con chiavi: nome, url, md5, dimensione.
    """
    url_api = f"https://zenodo.org/api/records/{record_id}"
    risposta = requests.get(url_api, timeout=timeout)
    risposta.raise_for_status()
    dati = risposta.json()

    file_record = []
    for voce in dati.get("files", []):
        # Il campo checksum ha la forma "md5:<digest>"
        checksum = voce.get("checksum", "")
        md5 = checksum.split(":", 1)[1] if checksum.startswith("md5:") else None
        file_record.append({
            "nome": voce["key"],
            "url": voce["links"]["self"],
            "md5": md5,
            "dimensione": voce.get("size"),
        })
    return file_record


def registra_checksum(registro: Path, nome_file: str, md5: str) -> None:
    """Registra un md5 calcolato localmente nel registro JSON dei checksum.

    Serve per le fonti che non pubblicano checksum (Dropbox, archivi GitHub):
    al primo download il digest viene salvato e nei download futuri fa da
    riferimento di riproducibilità.
    """
    registro.parent.mkdir(parents=True, exist_ok=True)
    contenuto: dict[str, str] = {}
    if registro.exists():
        # utf-8-sig: tollera l'eventuale BOM (es. file toccati da PowerShell 5.1)
        contenuto = json.loads(registro.read_text(encoding="utf-8-sig"))
    contenuto[nome_file] = md5
    registro.write_text(json.dumps(contenuto, indent=2, ensure_ascii=False),
                        encoding="utf-8")


def leggi_checksum_registrato(registro: Path, nome_file: str) -> str | None:
    """Legge dal registro JSON l'md5 registrato per `nome_file` (None se assente)."""
    if not registro.exists():
        return None
    # utf-8-sig: tollera l'eventuale BOM (es. file toccati da PowerShell 5.1)
    contenuto = json.loads(registro.read_text(encoding="utf-8-sig"))
    return contenuto.get(nome_file)
