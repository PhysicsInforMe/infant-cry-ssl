"""Script 02 — Filtro licenze clip per clip e manifest delle attribuzioni.

Attraversa i corpora scaricati (script 01, con --extract) e produce
data/licenses_manifest.csv con una riga per ogni clip AMMESSA: origine,
autore, licenza e link. I clip con licenza non ammessa o non verificabile
vengono scartati e conteggiati nel log (mai inclusi "per buona volontà").

Regole per corpus (README):
- FSD50K: licenza per clip nei metadati ufficiali; ammesse CC0 e CC BY,
  escluse CC BY-NC e qualunque licenza non riconosciuta.
- VocalSound: CC BY-SA 4.0 sull'intero corpus, attribuzione a livello corpus.
- donateacry: ODbL/DbCL sull'intero corpus; il contributore (bambino) si
  ricava dal prefisso UUID del filename e finisce nel manifest, perché
  governa lo split per bambino in valutazione.
- Freesound: previsto, si attiverà insieme allo script 01.

Uso:
    python scripts/02_filter_licenses.py
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

# Bootstrap del path: rende importabile src/babycry senza installare il pacchetto
RADICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RADICE / "src"))

from babycry.licenses import licenza_ammessa  # noqa: E402

log = logging.getLogger("02_licenze")

# Colonne del manifest delle licenze (una riga per clip ammessa).
# eta e genere sono valorizzate solo dove i metadati esistono (donateacry);
# l'eta resta nel codice originale del corpus: 04 = 0-4 settimane,
# 48 = 4-8 settimane, 26 = 2-6 mesi, 72 = 7 mesi-2 anni, 22 = oltre 2 anni.
COLONNE_MANIFEST = ["corpus", "clip_id", "percorso", "classe", "contributore",
                    "eta", "genere", "autore", "licenza", "link_origine"]

# Classi della versione pulita di donateacry (cartelle del corpus)
CLASSI_DONATEACRY = ["belly_pain", "burping", "discomfort", "hungry", "tired"]


def filtra_fsd50k(cartella: Path) -> tuple[list[dict], dict[str, int]]:
    """Filtra FSD50K per licenza usando i metadati ufficiali per clip.

    I file dev_clips_info_FSD50K.json / eval_clips_info_FSD50K.json (dentro
    FSD50K.metadata) riportano per ogni clip uploader e URL della licenza.

    Args:
        cartella: cartella del corpus estratto (data/raw/fsd50k).

    Returns:
        Coppia (righe ammesse per il manifest, contatori degli scarti).
    """
    righe: list[dict] = []
    scarti = {"licenza_non_ammessa": 0, "licenza_non_verificabile": 0,
              "audio_assente": 0}

    # Cartella metadati: il nome esatto dipende dall'estrazione, quindi la cerchiamo
    cartelle_metadati = list(cartella.glob("**/dev_clips_info_FSD50K.json"))
    if not cartelle_metadati:
        log.warning("Metadati FSD50K non trovati in %s: corpus saltato "
                    "(eseguire prima 01 con --extract)", cartella)
        return righe, scarti
    cartella_metadati = cartelle_metadati[0].parent

    # Coppie (file di metadati, cartella audio corrispondente)
    partizioni = [
        ("dev_clips_info_FSD50K.json", "FSD50K.dev_audio"),
        ("eval_clips_info_FSD50K.json", "FSD50K.eval_audio"),
    ]
    for nome_json, nome_audio in partizioni:
        percorso_json = cartella_metadati / nome_json
        if not percorso_json.exists():
            log.warning("Metadati mancanti: %s", percorso_json)
            continue
        info_clip = json.loads(percorso_json.read_text(encoding="utf-8"))
        cartella_audio = cartella / nome_audio

        for clip_id, info in info_clip.items():
            ammessa, normalizzata = licenza_ammessa(info.get("license"))
            if normalizzata is None:
                scarti["licenza_non_verificabile"] += 1
                continue
            if not ammessa:
                scarti["licenza_non_ammessa"] += 1
                continue
            percorso_wav = cartella_audio / f"{clip_id}.wav"
            if not percorso_wav.exists():
                scarti["audio_assente"] += 1
                continue
            righe.append({
                "corpus": "fsd50k",
                "clip_id": clip_id,
                "percorso": percorso_wav.relative_to(RADICE).as_posix(),
                "classe": "",              # nessuna etichetta di motivo qui
                "contributore": "",        # non applicabile
                "eta": "",                 # non applicabile (non sono pianti)
                "genere": "",
                "autore": info.get("uploader", ""),
                "licenza": normalizzata,
                # Link canonico della pagina Freesound del clip
                "link_origine": f"https://freesound.org/s/{clip_id}/",
            })
    return righe, scarti


def filtra_vocalsound(cartella: Path) -> tuple[list[dict], dict[str, int]]:
    """Include VocalSound (CC BY-SA 4.0 sull'intero corpus).

    L'autore per clip è lo speaker anonimizzato (prefisso del filename,
    es. f0214 / m0359); l'attribuzione vera è a livello di corpus.

    Args:
        cartella: cartella del corpus estratto (data/raw/vocalsound).

    Returns:
        Coppia (righe ammesse, contatori degli scarti).
    """
    righe: list[dict] = []
    scarti = {"audio_assente": 0}

    wav_trovati = sorted(cartella.rglob("*.wav"))
    if not wav_trovati:
        log.warning("Nessun wav VocalSound in %s: corpus saltato "
                    "(eseguire prima 01 con --extract)", cartella)
        return righe, scarti

    for wav in wav_trovati:
        # Speaker anonimizzato = primo token del filename (es. "f0214_0_cough.wav")
        speaker = wav.stem.split("_")[0]
        righe.append({
            "corpus": "vocalsound",
            "clip_id": wav.stem,
            "percorso": wav.relative_to(RADICE).as_posix(),
            "classe": "",
            "contributore": speaker,
            "eta": "",                 # eta' del neonato: non applicabile qui
            "genere": "",
            "autore": "VocalSound (Y. Gong et al., MIT)",
            "licenza": "CC BY-SA",
            "link_origine": "https://github.com/YuanGongND/vocalsound",
        })
    return righe, scarti


def filtra_donateacry(cartella: Path) -> tuple[list[dict], dict[str, int]]:
    """Include donateacry pulito (ODbL/DbCL) coi metadati dal filename.

    Il filename codifica tutti i metadati nel formato
    ``UUID-timestamp-versioneapp-genere-eta-motivo.wav``:
    - il prefisso UUID (36 caratteri) identifica il contributore, cioè il
      bambino: è la chiave dello split per bambino in valutazione;
    - genere: m/f; eta: fascia in codice (vedi COLONNE_MANIFEST).

    Args:
        cartella: cartella del corpus estratto (data/raw/donateacry).

    Returns:
        Coppia (righe ammesse, contatori degli scarti).
    """
    righe: list[dict] = []
    scarti = {"filename_malformato": 0, "fuori_target_eta": 0}

    # Versione pulita ed etichettata del corpus (cartelle per classe)
    cartelle_pulite = list(cartella.glob("**/donateacry_corpus_cleaned_and_updated_data"))
    if not cartelle_pulite:
        log.warning("Corpus donateacry pulito non trovato in %s: saltato "
                    "(eseguire prima 01 con --extract)", cartella)
        return righe, scarti

    for classe in CLASSI_DONATEACRY:
        for wav in sorted(cartelle_pulite[0].glob(f"{classe}/*.wav")):
            # UUID v4 = 36 caratteri: se il prefisso non ha quella forma il
            # contributore non è ricavabile e il clip è inutilizzabile per lo split
            prefisso = wav.stem[:36]
            if len(prefisso) < 36 or prefisso.count("-") != 4:
                scarti["filename_malformato"] += 1
                continue
            # Metadati dopo l'UUID: [5]=timestamp, [6]=versione app,
            # [7]=genere (m/f), [8]=fascia d'eta, [9]=codice motivo.
            # Se il formato non torna, i campi restano vuoti ma il clip
            # si tiene (l'eta e' un extra, non un requisito di ammissione).
            parti = wav.stem.split("-")
            genere = parti[7] if len(parti) == 10 and parti[7] in ("m", "f") else ""
            eta = parti[8] if len(parti) == 10 else ""
            # Fascia "22" = oltre 2 anni: fuori dal target dello studio (0-24 mesi),
            # esclusa da training e valutazione (decisione del 25/8)
            if eta == "22":
                scarti["fuori_target_eta"] += 1
                continue
            righe.append({
                "corpus": "donateacry",
                "clip_id": wav.stem,
                "percorso": wav.relative_to(RADICE).as_posix(),
                "classe": classe,
                "contributore": prefisso,
                "eta": eta,
                "genere": genere,
                "autore": "donateacry-corpus (G. Veres)",
                "licenza": "ODbL/DbCL",
                "link_origine": "https://github.com/gveres/donateacry-corpus",
            })
    return righe, scarti


def filtra_freesound(cartella: Path) -> tuple[list[dict], dict[str, int]]:
    """Filtra i clip Freesound (query dirette) per licenza, dai metadati API.

    I metadati arrivano dallo script 01 (freesound_metadati.csv): la licenza
    è l'URL Creative Commons dichiarato dall'API, normalizzato e filtrato
    come per FSD50K. Il contributore è l'username dell'uploader: per i pianti
    è il miglior proxy disponibile della famiglia/bambino, e governa lo split.

    Args:
        cartella: cartella del corpus (data/raw/freesound).

    Returns:
        Coppia (righe ammesse, contatori degli scarti).
    """
    righe: list[dict] = []
    scarti = {"licenza_non_ammessa": 0, "licenza_non_verificabile": 0,
              "audio_assente": 0}

    percorso_metadati = cartella / "freesound_metadati.csv"
    if not percorso_metadati.exists():
        log.warning("Metadati Freesound non trovati in %s: corpus saltato "
                    "(eseguire prima 01 con la API key)", cartella)
        return righe, scarti

    with open(percorso_metadati, newline="", encoding="utf-8") as f:
        for riga in csv.DictReader(f):
            ammessa, normalizzata = licenza_ammessa(riga["licenza"])
            if normalizzata is None:
                scarti["licenza_non_verificabile"] += 1
                continue
            if not ammessa or normalizzata not in ("CC0", "CC BY"):
                scarti["licenza_non_ammessa"] += 1
                continue
            percorso_mp3 = cartella / "audio" / f"{riga['clip_id']}.mp3"
            if not percorso_mp3.exists():
                scarti["audio_assente"] += 1
                continue
            righe.append({
                "corpus": "freesound",
                "clip_id": riga["clip_id"],
                "percorso": percorso_mp3.relative_to(RADICE).as_posix(),
                "classe": "",              # il motivo non e' conoscibile qui
                "contributore": riga["autore"],   # uploader: proxy dello split
                "eta": "",
                "genere": "",
                "autore": riga["autore"],
                "licenza": normalizzata,
                "link_origine": f"https://freesound.org/s/{riga['clip_id']}/",
            })
    return righe, scarti


def main() -> int:
    """Punto d'ingresso: filtra i corpora e scrive data/licenses_manifest.csv."""
    parser = argparse.ArgumentParser(description="Filtro licenze e manifest attribuzioni")
    parser.add_argument("--dati", type=Path, default=RADICE / "data" / "raw",
                        help="cartella dei corpora scaricati")
    parser.add_argument("--manifest", type=Path,
                        default=RADICE / "data" / "licenses_manifest.csv",
                        help="percorso del manifest CSV in uscita")
    argomenti = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    filtri = [
        ("fsd50k", filtra_fsd50k),
        ("vocalsound", filtra_vocalsound),
        ("donateacry", filtra_donateacry),
        ("freesound", filtra_freesound),
    ]

    tutte_le_righe: list[dict] = []
    for nome, funzione in filtri:
        log.info("=== Corpus: %s ===", nome)
        righe, scarti = funzione(argomenti.dati / nome)
        tutte_le_righe.extend(righe)
        log.info("%s: %d clip ammesse", nome, len(righe))
        for motivo, conteggio in scarti.items():
            if conteggio:
                log.info("%s: %d clip scartate (%s)", nome, conteggio, motivo)

    # Scrittura del manifest (una riga per clip ammessa, encoding UTF-8)
    argomenti.manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(argomenti.manifest, "w", newline="", encoding="utf-8") as f:
        scrittore = csv.DictWriter(f, fieldnames=COLONNE_MANIFEST)
        scrittore.writeheader()
        scrittore.writerows(tutte_le_righe)

    log.info("Manifest scritto: %s (%d clip totali ammesse)",
             argomenti.manifest, len(tutte_le_righe))
    if not tutte_le_righe:
        log.error("Nessuna clip ammessa: i corpora sono stati scaricati ed estratti?")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
