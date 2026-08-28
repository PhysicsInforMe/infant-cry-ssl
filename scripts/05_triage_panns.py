"""Script 05 — Triage pianto/non-pianto con PANNs (tagger AudioSet).

Passa il tagger Cnn14 su tutti i clip del manifest licenze e scrive
data/triage_panns.csv con, per ogni clip, le probabilità delle classi
AudioSet di interesse (pianto neonatale, pianto generico, lamento, parlato)
più l'etichetta AudioSet dominante. Il triage serve a quattro cose:

1. trovare il pianto vero dentro FSD50K (si aggiunge a donateacry per
   l'adattamento di dominio SSL);
2. dare le etichette alla probe secondaria pianto/non-pianto del bake-off;
3. verificare che il pool di rumore dei sintetici non contenga pianti;
4. fare da base al futuro detector di stadio 1.

Licenze verificate (25/8/2026): codice PANNs MIT, checkpoint CC BY 4.0
(Zenodo 3987831). Uso di sviluppo. Il motivo del pianto NON viene mai
etichettato qui: solo presenza/assenza (vincolo di progetto).

Uso:
    python scripts/05_triage_panns.py
    python scripts/05_triage_panns.py --solo donateacry
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
import soundfile as sf
import yaml
from tqdm import tqdm

# Bootstrap del path: rende importabile src/babycry senza installare il pacchetto
RADICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RADICE / "src"))

from babycry.download import lista_file_zenodo, scarica_con_resume  # noqa: E402

log = logging.getLogger("05_triage")


def scarica_checkpoint(cfg: dict) -> Path:
    """Scarica il checkpoint Cnn14 da Zenodo con l'md5 ufficiale (idempotente)."""
    destinazione = RADICE / cfg["cartella_checkpoint"] / cfg["checkpoint_nome"]
    if destinazione.exists():
        return destinazione
    log.info("Scarico il checkpoint %s da Zenodo", cfg["checkpoint_nome"])
    file_record = lista_file_zenodo(int(cfg["zenodo_record"]))
    voce = next(f for f in file_record if f["nome"] == cfg["checkpoint_nome"])
    if not scarica_con_resume(voce["url"], destinazione, md5_atteso=voce["md5"]):
        raise RuntimeError("download del checkpoint fallito")
    return destinazione


def carica_audio_32k(percorso: Path, sr_obiettivo: int, durata_max_s: float) -> np.ndarray:
    """Carica un clip mono a 32 kHz (il sample rate di addestramento di Cnn14).

    Args:
        percorso: file audio.
        sr_obiettivo: 32000.
        durata_max_s: troncamento dei clip più lunghi (risparmio inutile oltre).

    Returns:
        Array float32 monodimensionale.
    """
    audio, sr = sf.read(percorso, dtype="float32", always_2d=True,
                        frames=-1)
    segnale = torch.from_numpy(audio.mean(axis=1))
    if sr != sr_obiettivo:
        segnale = torchaudio.functional.resample(segnale, sr, sr_obiettivo)
    campioni_max = int(durata_max_s * sr_obiettivo)
    segnale = segnale[:campioni_max]
    # Cnn14 richiede almeno ~0.5 s di input: i clip ultracorti (FSD50K arriva
    # a 0.3 s) si portano a 1 s con padding di zeri in coda
    if segnale.numel() < sr_obiettivo:
        segnale = torch.nn.functional.pad(segnale, (0, sr_obiettivo - segnale.numel()))
    return segnale.numpy()


def main() -> int:
    """Punto d'ingresso: checkpoint, inferenza su tutto il manifest, CSV."""
    parser = argparse.ArgumentParser(description="Triage pianto/non-pianto con PANNs")
    parser.add_argument("--config", type=Path, default=RADICE / "configs" / "triage.yaml")
    parser.add_argument("--manifest", type=Path,
                        default=RADICE / "data" / "licenses_manifest.csv")
    parser.add_argument("--solo", help="processa solo il corpus indicato")
    argomenti = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(argomenti.config.read_text(encoding="utf-8"))

    percorso_ckpt = scarica_checkpoint(cfg)

    # Import qui perché panns_inference legge file al primo import
    from panns_inference import AudioTagging, labels

    # Indici delle classi AudioSet di interesse (match esatto sull'etichetta)
    indici = {}
    for colonna, etichetta in cfg["classi_interesse"].items():
        indici[colonna] = labels.index(etichetta)

    dispositivo = cfg["device"] if torch.cuda.is_available() else "cpu"
    if dispositivo != "cuda":
        log.warning("CUDA non disponibile: triage su CPU (lento)")
    tagger = AudioTagging(checkpoint_path=str(percorso_ckpt), device=dispositivo)

    with open(argomenti.manifest, newline="", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))
    if argomenti.solo:
        manifest = [r for r in manifest if r["corpus"] == argomenti.solo]

    # Resume: clip già triagiati nel CSV esistente
    uscita = RADICE / cfg["uscita"]
    gia_fatti: set[str] = set()
    if uscita.exists():
        with open(uscita, newline="", encoding="utf-8") as f:
            gia_fatti = {r["clip_id"] for r in csv.DictReader(f)}
    da_fare = [r for r in manifest if r["clip_id"] not in gia_fatti]
    log.info("%d clip totali, %d già triagiati, %d da fare",
             len(manifest), len(manifest) - len(da_fare), len(da_fare))

    colonne = ["clip_id", "corpus", *cfg["classi_interesse"].keys(),
               "top_etichetta", "top_p"]
    nuovo = not uscita.exists()
    errori = 0
    with open(uscita, "a", newline="", encoding="utf-8") as f:
        scrittore = csv.DictWriter(f, fieldnames=colonne)
        if nuovo:
            scrittore.writeheader()
        for numero, voce in enumerate(tqdm(da_fare, desc="triage", unit="clip")):
            try:
                audio = carica_audio_32k(RADICE / voce["percorso"],
                                         int(cfg["sample_rate"]),
                                         float(cfg["durata_max_s"]))
                if len(audio) < 1000:
                    raise ValueError("clip troppo corto")
                # Inferenza: clipwise_output ha forma (1, 527)
                clipwise, _ = tagger.inference(audio[None, :])
                probabilita = clipwise[0]
            except Exception as errore:
                log.warning("Clip saltato (%s): %s", errore, voce["percorso"])
                errori += 1
                continue

            riga = {"clip_id": voce["clip_id"], "corpus": voce["corpus"],
                    "top_etichetta": labels[int(np.argmax(probabilita))],
                    "top_p": f"{float(probabilita.max()):.4f}"}
            for colonna, indice in indici.items():
                riga[colonna] = f"{float(probabilita[indice]):.4f}"
            scrittore.writerow(riga)
            if (numero + 1) % 200 == 0:
                f.flush()

    # Riepilogo per corpus con la soglia della config
    soglia = float(cfg["soglia_pianto"])
    conteggi: dict[str, list[int]] = {}
    with open(uscita, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            tot, pianti = conteggi.get(r["corpus"], [0, 0])
            e_pianto = float(r["p_pianto_neonato"]) >= soglia
            conteggi[r["corpus"]] = [tot + 1, pianti + int(e_pianto)]
    log.info("=== Riepilogo triage (soglia pianto neonatale %.2f) ===", soglia)
    for corpus, (tot, pianti) in sorted(conteggi.items()):
        log.info("  %-12s %6d clip, %5d con pianto neonatale (%.1f%%)",
                 corpus, tot, pianti, 100 * pianti / max(tot, 1))
    log.info("Clip saltati per errore: %d", errori)
    return 0


if __name__ == "__main__":
    sys.exit(main())
