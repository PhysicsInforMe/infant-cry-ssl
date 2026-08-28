"""App locale di esplorazione dei dati (Streamlit) — tool di supporto, non uno
stadio della pipeline (per questo non è numerata).

Serve a ispezionare i corpora scaricati PRIMA di scrivere codice di modello:
per ogni clip mostra le grandezze che contano per il task e per l'affinamento
del pretext task (README):

- spettrogramma log-mel, lo stesso front-end della cache (64 mel, 16 kHz);
- contorno della F0 (la banda 250-700 Hz del pianto neonatale è evidenziata);
- inviluppo d'ampiezza e spettro di modulazione (0.5-16 Hz): il "ritmo" del
  pianto, cioè l'informazione che un filter-ID statico non cattura;
- metriche rapide: durata, quota di energia in banda 250-700 Hz, centroide
  spettrale, frequenza di modulazione dominante.

Più una scheda di statistiche del dataset (numeriche per corpus/classe/
contributore, durate dalla cache mel se disponibile).

Uso (dal venv, dopo gli script 01-02):
    streamlit run scripts/esplora_dati_app.py
"""

from __future__ import annotations

import csv
import json
import random
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st
import torch
import torchaudio
import yaml

# Bootstrap del path: rende importabile src/babycry senza installare il pacchetto
RADICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RADICE / "src"))

from babycry.mel import FrontendLogMel  # noqa: E402

# Gate d'ascolto dei sintetici : True = marca con 🎧 le varianti
# prioritarie da ascoltare. Spento il 25/8/2026 dopo l'approvazione all'ascolto
# (entrambe le tecniche giudicate buone; confronto rimandato all'ablation).
MOSTRA_CONSIGLI_ASCOLTO = False

# Banda della F0 del pianto neonatale (vedi README), evidenziata nei grafici
F0_PIANTO_HZ = (250.0, 700.0)
# Banda delle modulazioni lente dell'inviluppo (vedi README)
MODULAZIONI_HZ = (0.5, 16.0)


# ----------------------------------------------------------------------------
# Caricamento dati (manifest o, in mancanza, scansione di data/raw)
# ----------------------------------------------------------------------------

@st.cache_data
def carica_clip_disponibili() -> list[dict]:
    """Elenca i clip disponibili: dal manifest (script 02) o scandendo data/raw.

    Returns:
        Lista di dict con corpus, clip_id, percorso, classe, contributore.
        Il fallback di scansione serve a esplorare l'audio anche prima del
        filtro licenze (ma il manifest resta la fonte di verità per il training).
    """
    manifest = RADICE / "data" / "licenses_manifest.csv"
    if manifest.exists():
        with open(manifest, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    # Fallback: scansione dei wav in data/raw (tetto di sicurezza a 5000 file)
    righe = []
    for wav in (RADICE / "data" / "raw").rglob("*.wav"):
        righe.append({
            "corpus": wav.relative_to(RADICE / "data" / "raw").parts[0],
            "clip_id": wav.stem,
            "percorso": wav.relative_to(RADICE).as_posix(),
            "classe": "",
            "contributore": "",
        })
        if len(righe) >= 5000:
            break
    return righe


@st.cache_data
def carica_sintetici() -> dict[str, list[dict]]:
    """Carica il manifest delle varianti sintetiche (script 04), se esiste.

    Returns:
        Dizionario origine_clip_id -> lista delle varianti (righe del manifest
        sintetici). Vuoto se lo script 04 non è ancora stato eseguito.
    """
    percorso = RADICE / "data" / "sintetici_manifest.csv"
    if not percorso.exists():
        return {}
    gruppi: dict[str, list[dict]] = {}
    with open(percorso, newline="", encoding="utf-8") as f:
        for riga in csv.DictReader(f):
            gruppi.setdefault(riga["origine_clip_id"], []).append(riga)
    return gruppi


@st.cache_data
def ascolti_consigliati() -> set[str]:
    """Seleziona le varianti prioritarie per il gate d'ascolto (marcate 🎧).

    Criterio: per ogni classe e tecnica, i casi dove un eventuale artefatto
    si sente prima — per il vocoder le varianti con i parametri più estremi
    (pitch massimo in modulo, formanti ai due estremi), per il rumore le tre
    con SNR più basso su rumori diversi. Se gli estremi passano l'ascolto,
    il resto della generazione passa a maggior ragione (~30 varianti totali).

    Returns:
        Insieme dei clip_id delle varianti consigliate.
    """
    per_gruppo: dict[tuple[str, str], list[dict]] = {}
    for varianti in carica_sintetici().values():
        for v in varianti:
            per_gruppo.setdefault((v["classe"], v["tecnica"]), []).append(v)

    def parametri_vocoder(v: dict) -> tuple[float, float]:
        """Estrae (pitch in semitoni, fattore formanti) dalla stringa parametri."""
        m = re.search(r"pitch ([+-]?[\d.]+) st, formanti x([\d.]+)", v["parametri"])
        return (float(m.group(1)), float(m.group(2))) if m else (0.0, 1.0)

    def snr_variante(v: dict) -> int:
        """Estrae l'SNR in dB dalla stringa parametri (99 se assente)."""
        m = re.search(r"SNR (\d+)", v["parametri"])
        return int(m.group(1)) if m else 99

    scelte: set[str] = set()
    for (_, tecnica), gruppo in per_gruppo.items():
        if tecnica == "vocoder":
            # I tre estremi: pitch massimo in modulo, formanti max e min
            scelte.add(max(gruppo, key=lambda v: abs(parametri_vocoder(v)[0]))["clip_id"])
            scelte.add(max(gruppo, key=lambda v: parametri_vocoder(v)[1])["clip_id"])
            scelte.add(min(gruppo, key=lambda v: parametri_vocoder(v)[1])["clip_id"])
        else:
            # I tre SNR più bassi, evitando di ripetere lo stesso rumore
            rumori_visti: set[str] = set()
            for v in sorted(gruppo, key=snr_variante):
                stem = v["parametri"].split("rumore ")[-1]
                if stem in rumori_visti:
                    continue
                rumori_visti.add(stem)
                scelte.add(v["clip_id"])
                if len(rumori_visti) >= 3:
                    break
    return scelte


@st.cache_data
def carica_triage() -> dict[str, dict]:
    """Carica gli score del triage PANNs (script 05), se il CSV esiste.

    Nota: il file cresce mentre lo script 05 gira; la cache di Streamlit lo
    legge una volta per sessione (ricaricare l'app per aggiornare).

    Returns:
        Dizionario clip_id -> riga del triage (score delle classi AudioSet).
    """
    percorso = RADICE / "data" / "triage_panns.csv"
    if not percorso.exists():
        return {}
    with open(percorso, newline="", encoding="utf-8") as f:
        return {r["clip_id"]: r for r in csv.DictReader(f)}


@st.cache_resource
def carica_frontend() -> FrontendLogMel:
    """Costruisce il front-end log-mel dalla stessa config della cache (coerenza!)."""
    cfg = yaml.safe_load((RADICE / "configs" / "audio_frontend.yaml")
                         .read_text(encoding="utf-8"))
    return FrontendLogMel(cfg)


@st.cache_data(max_entries=32)
def analizza_clip(percorso_relativo: str) -> dict:
    """Carica un clip e calcola tutte le grandezze mostrate dall'app.

    Args:
        percorso_relativo: percorso del file audio relativo alla radice del repo.

    Returns:
        Dict con: segnale mono a 16 kHz, log-mel, contorno F0 (NaN sui frame
        non voiced), inviluppo per frame, spettro di modulazione e metriche
        scalari (durata, quota di energia in banda F0, centroide, ecc.).
    """
    frontend = carica_frontend()
    sr_obiettivo = frontend.sample_rate

    audio, sr = sf.read(RADICE / percorso_relativo, dtype="float32", always_2d=True)
    segnale = torch.from_numpy(audio.mean(axis=1))
    if sr != sr_obiettivo:
        segnale = torchaudio.functional.resample(segnale, sr, sr_obiettivo)

    # Spettrogramma log-mel con lo stesso identico front-end della cache
    logmel, _, _ = frontend.da_file(RADICE / percorso_relativo)

    # Spettrogramma lineare di servizio per le metriche in Hz
    spec = torchaudio.transforms.Spectrogram(
        n_fft=frontend.n_fft, hop_length=frontend.hop, power=2.0)(segnale)
    freq_bin = np.linspace(0, sr_obiettivo / 2, spec.shape[0])

    # Quota di energia nella banda della F0 del pianto (indizio "quanto pianto c'è")
    in_banda = (freq_bin >= F0_PIANTO_HZ[0]) & (freq_bin <= F0_PIANTO_HZ[1])
    energia_totale = float(spec.sum()) + 1e-12
    quota_banda_f0 = float(spec[in_banda].sum()) / energia_totale

    # Centroide spettrale medio (baricentro dello spettro, in Hz)
    profilo = spec.mean(dim=1).numpy()
    centroide_hz = float((freq_bin * profilo).sum() / (profilo.sum() + 1e-12))

    # Inviluppo d'ampiezza per frame (radice dell'energia, campionato a 100 Hz
    # dall'hop di 10 ms): è la base del "ritmo" del pianto
    inviluppo = np.sqrt(spec.sum(dim=0).numpy() + 1e-12)

    # Spettro di modulazione: FFT dell'inviluppo senza media, finestra di Hann.
    # Le modulazioni lente (0.5-16 Hz) distinguono pianto ritmico da lamento continuo.
    frame_rate = sr_obiettivo / frontend.hop  # 100 Hz
    env = (inviluppo - inviluppo.mean()) * np.hanning(len(inviluppo))
    spettro_mod = np.abs(np.fft.rfft(env))
    freq_mod = np.fft.rfftfreq(len(env), d=1.0 / frame_rate)

    # Frequenza di modulazione dominante dentro la banda 0.5-16 Hz
    maschera_mod = (freq_mod >= MODULAZIONI_HZ[0]) & (freq_mod <= MODULAZIONI_HZ[1])
    mod_dominante_hz = (float(freq_mod[maschera_mod][spettro_mod[maschera_mod].argmax()])
                        if maschera_mod.any() else float("nan"))

    # Contorno F0 (autocorrelazione di torchaudio); i frame quasi silenziosi
    # vengono mascherati a NaN perché lì la stima è rumore
    try:
        f0 = torchaudio.functional.detect_pitch_frequency(
            segnale.unsqueeze(0), sample_rate=sr_obiettivo,
            freq_low=100, freq_high=1000).squeeze(0).numpy().astype(float)
        # Energia media per frame F0 (il detector usa frame da ~10 ms)
        n_f0 = len(f0)
        energia_f0 = np.interp(np.linspace(0, 1, n_f0),
                               np.linspace(0, 1, len(inviluppo)), inviluppo)
        f0[energia_f0 < 0.1 * energia_f0.max()] = np.nan
    except Exception:
        f0 = np.full(8, np.nan)  # clip troppo corto per il detector: nessun contorno

    return {
        "segnale": segnale.numpy(),
        "sr": sr_obiettivo,
        "sr_originale": sr,
        "logmel": logmel.astype(np.float32),
        "hop_s": frontend.hop / sr_obiettivo,
        "f0": f0,
        "inviluppo": inviluppo,
        "freq_mod": freq_mod,
        "spettro_mod": spettro_mod,
        "durata_s": len(segnale) / sr_obiettivo,
        "quota_banda_f0": quota_banda_f0,
        "centroide_hz": centroide_hz,
        "mod_dominante_hz": mod_dominante_hz,
        "rms_db": float(20 * np.log10(np.sqrt(np.mean(segnale.numpy() ** 2)) + 1e-12)),
    }


# ----------------------------------------------------------------------------
# Scheda 1: esplorazione del singolo clip
# ----------------------------------------------------------------------------

def scheda_clip(clip: list[dict]) -> None:
    """Scheda interattiva: scelta del clip, ascolto e grafici delle feature."""
    st.sidebar.header("Selezione clip")

    corpora = sorted({c["corpus"] for c in clip})
    corpus = st.sidebar.selectbox("Corpus", corpora)
    del_corpus = [c for c in clip if c["corpus"] == corpus]

    # Filtro per classe (utile per donateacry)
    classi = sorted({c["classe"] for c in del_corpus if c["classe"]})
    if classi:
        classe = st.sidebar.selectbox("Classe", ["(tutte)"] + classi)
        if classe != "(tutte)":
            del_corpus = [c for c in del_corpus if c["classe"] == classe]

    # Filtro testuale sul nome + pulsante clip casuale
    filtro = st.sidebar.text_input("Filtro sul nome (opzionale)")
    if filtro:
        del_corpus = [c for c in del_corpus if filtro.lower() in c["clip_id"].lower()]
    if st.sidebar.button("🎲 Clip casuale"):
        st.session_state["clip_scelto"] = random.choice(del_corpus)["clip_id"]

    # Gate d'ascolto: 🎧 marca i clip che contengono varianti consigliate
    consigli = ascolti_consigliati() if MOSTRA_CONSIGLI_ASCOLTO else set()
    clip_con_consigli = {origine for origine, vs in carica_sintetici().items()
                         if any(v["clip_id"] in consigli for v in vs)}
    if consigli and st.sidebar.checkbox("🎧 Solo clip con ascolti consigliati"):
        del_corpus = [c for c in del_corpus if c["clip_id"] in clip_con_consigli]

    nomi = [c["clip_id"] for c in del_corpus[:2000]]  # tetto per non appesantire la UI
    if not nomi:
        st.warning("Nessun clip col filtro attuale.")
        return
    predefinito = st.session_state.get("clip_scelto")
    indice = nomi.index(predefinito) if predefinito in nomi else 0
    scelto = st.selectbox(f"Clip ({len(del_corpus)} disponibili, primi 2000 in lista)",
                          nomi, index=indice,
                          format_func=lambda n: f"🎧 {n}" if n in clip_con_consigli else n)
    voce = next(c for c in del_corpus if c["clip_id"] == scelto)

    percorso = RADICE / voce["percorso"]
    if not percorso.exists():
        st.error(f"File non trovato: {voce['percorso']}")
        return

    # Varianti sintetiche del clip (script 04): selettore e ascolto A/B.
    # I grafici sotto si riferiscono a ciò che è selezionato qui (originale
    # o variante), così si possono confrontare anche visivamente.
    varianti = carica_sintetici().get(voce["clip_id"], [])
    percorso_analisi = voce["percorso"]
    variante_scelta = None
    if varianti:
        st.markdown("##### 🧪 Varianti sintetiche di questo clip")
        # Le varianti consigliate per l'ascolto (🎧) vengono in testa alla lista
        consigli = ascolti_consigliati() if MOSTRA_CONSIGLI_ASCOLTO else set()
        varianti = sorted(varianti, key=lambda v: v["clip_id"] not in consigli)
        etichette = ["(originale)"] + [
            ("🎧 " if v["clip_id"] in consigli else "") +
            f"{v['tecnica']} · {v['parametri']}"
            for v in varianti]
        scelta = st.selectbox(f"Confronta ({len(varianti)} varianti generate)",
                              etichette)
        if scelta != "(originale)":
            variante_scelta = varianti[etichette.index(scelta) - 1]
            percorso_analisi = variante_scelta["percorso"]

    dati = analizza_clip(percorso_analisi)

    # Player audio: affiancati per l'ascolto A/B quando c'è una variante scelta
    if variante_scelta is not None:
        col_orig, col_var = st.columns(2)
        col_orig.markdown("**Originale**")
        col_orig.audio(str(percorso))
        col_var.markdown(f"**Variante** ({variante_scelta['tecnica']})")
        col_var.audio(str(RADICE / variante_scelta["percorso"]))
        st.info("I grafici e le metriche qui sotto si riferiscono alla "
                "**variante** selezionata. Scegli \"(originale)\" per tornare "
                "al clip reale.")
    else:
        st.audio(str(percorso))
    col = st.columns(5)
    col[0].metric("Durata", f"{dati['durata_s']:.2f} s")
    col[1].metric("Energia in 250-700 Hz", f"{100 * dati['quota_banda_f0']:.0f} %",
                  help="Quota dell'energia nella banda tipica della F0 del pianto")
    col[2].metric("Centroide spettrale", f"{dati['centroide_hz']:.0f} Hz")
    col[3].metric("Modulazione dominante", f"{dati['mod_dominante_hz']:.1f} Hz",
                  help="Picco dello spettro di modulazione in 0.5-16 Hz: il ritmo del pianto")
    col[4].metric("Livello RMS", f"{dati['rms_db']:.1f} dB")
    if voce.get("classe") or voce.get("contributore"):
        # Decodifica delle fasce d'eta di donateacry (codici del corpus)
        fasce_eta = {"04": "0-4 settimane", "48": "4-8 settimane",
                     "26": "2-6 mesi", "72": "7 mesi-2 anni", "22": "oltre 2 anni"}
        eta = fasce_eta.get(voce.get("eta", ""), voce.get("eta") or "—")
        st.caption(f"classe: **{voce.get('classe') or '—'}** · "
                   f"età: **{eta}** · genere: **{voce.get('genere') or '—'}** · "
                   f"contributore: `{voce.get('contributore') or '—'}` · "
                   f"sr originale: {dati['sr_originale']} Hz")

    # Score del triage PANNs sul clip reale (se lo script 05 è stato eseguito).
    # Sono sigmoidi AudioSet NON calibrate: contano per il ranking, non come
    # probabilità assolute (i pianti veri stanno spesso tra 0.1 e 0.6).
    triage = carica_triage().get(voce["clip_id"])
    if triage:
        st.caption(f"Triage PANNs — pianto neonatale: **{float(triage['p_pianto_neonato']):.2f}** · "
                   f"pianto generico: {float(triage['p_pianto_generico']):.2f} · "
                   f"lamento: {float(triage['p_lamento']):.2f} · "
                   f"parlato: {float(triage['p_parlato']):.2f} · "
                   f"etichetta dominante AudioSet: *{triage['top_etichetta']}*")

    # --- Grafici: titoli neutri (il clip può essere qualunque suono) e sotto
    # --- ogni figura un pannello con le regole di lettura, per studiare ------
    tempo = np.arange(len(dati["segnale"])) / dati["sr"]

    # 1) Forma d'onda
    fig, asse = plt.subplots(figsize=(11, 2.2))
    asse.plot(tempo, dati["segnale"], linewidth=0.4, color="tab:blue")
    asse.set_title("Forma d'onda")
    asse.set_xlabel("tempo [s]")
    asse.set_xlim(0, tempo[-1])
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    with st.expander("📖 Come leggere la forma d'onda"):
        st.markdown("""
- **Struttura a raffiche** (burst di 0.5–2 s separati da pause): tipica del pianto —
  la pausa è l'inspirazione. Raffiche **regolari e ripetute** → compatibile col pianto
  ritmico (fame). **Attacco improvviso + urlo lungo + pausa lunga** (apnea) → schema
  classico del dolore.
- **Ampiezza quasi costante senza pause**: suono stazionario (onde del mare, ventola,
  pioggia, brusio). Il pianto vero torna quasi a zero tra una raffica e l'altra.
- **Picchi tagliati piatti** al massimo dell'ampiezza = clipping del microfono:
  clip di qualità dubbia, da trattare con sospetto.
""")

    # 2) Log-mel (identico a quello che vedrà il modello)
    fig, asse = plt.subplots(figsize=(11, 3.6))
    estensione = [0, dati["logmel"].shape[0] * dati["hop_s"], 0, dati["logmel"].shape[1]]
    asse.imshow(dati["logmel"].T, origin="lower", aspect="auto",
                extent=estensione, cmap="magma")
    asse.set_title("Spettrogramma log-mel 64 bande (il front-end del modello)")
    asse.set_ylabel("banda mel")
    asse.set_xlabel("tempo [s]")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    with st.expander("📖 Come leggere il log-mel"):
        st.markdown("""
- Asse y = bande mel (in basso i gravi), colore chiaro = più energia. È **esattamente
  l'input del modello**: quello che non si vede qui, il modello non lo vede.
- **Righe orizzontali parallele e brillanti** = armoniche di un suono tonale (voce).
  Nel pianto neonatale partono dalla F0 alta (250–700 Hz) e sono ben spaziate; nel
  **parlato adulto** sono più fitte e più in basso (F0 100–250 Hz).
- Le righe che **salgono e scendono insieme** disegnano la melodia: contorno
  **discendente** → lamento/stanchezza; **ascendente** → disagio; **piatto e alto
  a lungo** → dolore.
- Righe che **si spezzano in zone caotiche** (rumore largo, sdoppiamenti) = rotture
  della voce e subarmoniche: marcatori acustici del pianto intenso/dolore.
- **Energia diffusa senza righe** = rumore (mare, pioggia, traffico): nessuna
  struttura armonica, il colore sfuma senza organizzarsi.
""")

    # 3) Contorno F0 con banda del pianto evidenziata
    fig, asse = plt.subplots(figsize=(11, 2.6))
    tempo_f0 = np.linspace(0, dati["durata_s"], len(dati["f0"]))
    asse.axhspan(*F0_PIANTO_HZ, color="tab:green", alpha=0.15,
                 label="banda F0 pianto neonatale (250-700 Hz)")
    asse.plot(tempo_f0, dati["f0"], ".", markersize=3, color="tab:red")
    asse.set_ylim(0, 1000)
    asse.set_title("Contorno della frequenza fondamentale (frame silenziosi mascherati)")
    asse.set_ylabel("Hz")
    asse.set_xlabel("tempo [s]")
    asse.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    with st.expander("📖 Come leggere il contorno F0"):
        st.markdown("""
- **Punti dentro la banda verde** (250–700 Hz) e stabili per tratti lunghi →
  compatibile con pianto neonatale. **Punti stabili sotto i 250 Hz** → voce adulta.
  **Pochi punti o punti sparsi a caso** → suono non tonale (rumore): la F0 di un
  rumore non esiste e il detector produce spazzatura.
- La **forma** del contorno è informazione di classe: **discendente** (parte alto e
  scivola giù) è il profilo del lamento da stanchezza; **ascendente** compare nel
  disagio; **alto, teso e prolungato** (anche oltre la banda) nel dolore.
- **Salti improvvisi al doppio o alla metà** del valore = errori d'ottava del
  detector, non fisiologia: guarda la tendenza, non il singolo punto.
""")

    # 4) Inviluppo d'ampiezza (il ritmo, nel tempo)
    fig, asse = plt.subplots(figsize=(11, 2.4))
    tempo_env = np.arange(len(dati["inviluppo"])) * dati["hop_s"]
    asse.plot(tempo_env, dati["inviluppo"], color="tab:purple", linewidth=0.8)
    asse.set_title("Inviluppo d'ampiezza (campionato a 100 Hz)")
    asse.set_xlabel("tempo [s]")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    with st.expander("📖 Come leggere l'inviluppo"):
        st.markdown("""
- È l'energia del suono nel tempo, senza il dettaglio spettrale: la vista giusta
  per il **ritmo**.
- **Colline regolari quasi periodiche** che toccano quasi lo zero tra l'una e
  l'altra → raffiche con respiri: la firma del pianto ritmico (fame). La distanza
  tra le colline è il periodo del ritmo (2 colline al secondo → modulazione a 2 Hz).
- **Plateau lungo senza avvallamenti** → emissione continua: urlo prolungato
  (dolore) oppure suono stazionario non vocale — distingui i due casi col log-mel
  (armoniche presenti o no).
- **Inviluppo che non torna mai vicino a zero** → c'è rumore di fondo costante
  sotto il segnale (stanza rumorosa, TV): il clip è "sporco".
""")

    # 5) Spettro di modulazione (asse in Hz di modulazione, non in Hz audio)
    fig, asse = plt.subplots(figsize=(11, 2.6))
    maschera = dati["freq_mod"] <= 25
    asse.plot(dati["freq_mod"][maschera], dati["spettro_mod"][maschera],
              color="tab:purple")
    asse.axvspan(*MODULAZIONI_HZ, color="tab:green", alpha=0.15,
                 label="modulazioni lente (0.5-16 Hz)")
    asse.set_title("Spettro di modulazione dell'inviluppo")
    asse.set_xlabel("frequenza di modulazione [Hz]")
    asse.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    with st.expander("📖 Come leggere lo spettro di modulazione"):
        st.markdown("""
- È la FFT **dell'inviluppo qui sopra**, non dell'audio: dice *a che velocità
  l'energia va su e giù*. Un picco a 2 Hz = il suono pulsa 2 volte al secondo.
- **Picco netto tra 1 e 4 Hz** → ritmo regolare di raffiche: firma del pianto
  ritmico (fame). **Spettro piatto/basso in banda** → emissione continua senza
  ritmo: lamento della stanchezza, urlo tenuto, o rumore stazionario.
- **Picco tra 3 e 8 Hz** con armoniche in un suono a F0 bassa → probabile parlato
  adulto (è il ritmo sillabico).
- **Energia diffusa solo sotto 1 Hz** senza picchi → fluttuazione lenta e
  irregolare: onde del mare, traffico, vento.
- Perché conta per noi: questa firma **vive nella dinamica temporale** ed è
  invisibile a un pretext task risolvibile dallo spettro medio (vedi README) —
  se i pianti la mostrano chiara e i rumori no, il pretext deve forzare la
  sensibilità all'inviluppo.
""")

    # Tabella di studio: firme acustiche attese per classe
    with st.expander("🎓 Firme acustiche attese per classe (riferimento di studio)"):
        st.markdown("""
| Classe | Forma d'onda / inviluppo | Contorno F0 | Spettro di modulazione |
|---|---|---|---|
| **Fame** | raffiche brevi e regolari con pause di respiro | in banda, relativamente stabile per raffica | picco netto 1–4 Hz |
| **Sonno/stanchezza** | lamento prolungato, poche pause | **melodia discendente**, spesso parte alto | piatto, poca energia in banda |
| **Dolore** | attacco improvviso, urlo lungo, poi pausa lunga (apnea) | alto e teso, anche sopra 700 Hz, con **rotture della voce** | debole durante l'urlo, energia a bassissime frequenze |
| **Disagio/gas** | pianto teso e intermittente, pause irregolari | **melodia ascendente** | picchi presenti ma meno regolari della fame |

Avvertenze: sono **tendenze dalla letteratura**, non regole rigide — la variabilità
tra bambini è enorme (è il motivo per cui serve un modello, e per cui lo split di
valutazione è per bambino). Su FSD50K e VocalSound queste firme di classe non si
applicano: lì il confronto utile è pianto vs non-pianto (armoniche alte + ritmo
lento contro tutto il resto).
""")


# ----------------------------------------------------------------------------
# Scheda 2: statistiche del dataset
# ----------------------------------------------------------------------------

def scheda_statistiche(clip: list[dict]) -> None:
    """Statistiche aggregate: numerosità per corpus/classe/contributore e durate."""
    st.subheader("Numerosità per corpus")
    per_corpus = {}
    for c in clip:
        per_corpus[c["corpus"]] = per_corpus.get(c["corpus"], 0) + 1
    st.bar_chart(per_corpus)

    # Dettaglio donateacry: classi e contributori (le chiavi dello split!)
    donateacry = [c for c in clip if c["corpus"] == "donateacry"]
    if donateacry:
        st.subheader("donateacry: classi e contributori")
        per_classe = {}
        for c in donateacry:
            per_classe[c["classe"]] = per_classe.get(c["classe"], 0) + 1
        contributori = {c["contributore"] for c in donateacry if c["contributore"]}
        col1, col2 = st.columns([2, 1])
        col1.bar_chart(per_classe)
        col2.metric("Contributori unici (bambini)", len(contributori))
        col2.caption("Lo split di valutazione è SEMPRE per contributore, mai per clip. "
                     "Nota lo sbilanciamento verso 'hungry': il ribilanciamento è già "
                     "previsto.")

    # Durate dalla cache mel, se già costruita (n_frame * 10 ms: gratis e preciso)
    st.subheader("Durate dei clip (dalla cache log-mel)")
    indici = sorted((RADICE / "cache" / "mel").glob("*_index.json"))
    if not indici:
        st.info("Cache non ancora costruita (script 03): le durate compariranno qui.")
        return
    for percorso_indice in indici:
        indice = json.loads(percorso_indice.read_text(encoding="utf-8"))
        durate = np.array([v["n_frame"] for v in indice["clip"].values()]) * 0.01
        if durate.size == 0:
            continue
        ore = durate.sum() / 3600
        fig, asse = plt.subplots(figsize=(9, 2.2))
        asse.hist(durate, bins=60, color="tab:blue")
        asse.set_title(f"{percorso_indice.stem.replace('_index', '')} — "
                       f"{len(durate)} clip, {ore:.1f} ore totali, "
                       f"mediana {np.median(durate):.1f} s")
        asse.set_xlabel("durata [s]")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ----------------------------------------------------------------------------
# Scheda 3: etichettatura manuale dei casi limite del triage
# ----------------------------------------------------------------------------

def scheda_casi_limite() -> None:
    """Etichettatura umana dei casi limite del triage .

    Presenta un caso alla volta (audio + score PANNs + motivo della selezione)
    e salva il voto in data/etichette_manuali.csv. I voti umani sovrascrivono
    PANNs in ogni uso successivo (pool di pianto, probe, soglia operativa).
    """
    percorso_casi = RADICE / "data" / "casi_limite.csv"
    percorso_voti = RADICE / "data" / "etichette_manuali.csv"
    if not percorso_casi.exists():
        st.info("Nessun caso limite ancora selezionato: serve prima il triage "
                "completo (script 05) e la selezione (script 06).")
        return

    with open(percorso_casi, newline="", encoding="utf-8") as f:
        casi = list(csv.DictReader(f))
    voti: dict[str, str] = {}
    if percorso_voti.exists():
        with open(percorso_voti, newline="", encoding="utf-8") as f:
            voti = {r["clip_id"]: r["voto"] for r in csv.DictReader(f)}

    da_fare = [c for c in casi if c["clip_id"] not in voti]
    st.progress(len(voti) / max(len(casi), 1),
                text=f"Etichettati {len(voti)} su {len(casi)} casi limite")

    def riscrivi_voto(clip_id: str, nuovo_voto: str) -> None:
        """Riscrive il CSV dei voti cambiando il voto del clip indicato."""
        with open(percorso_voti, newline="", encoding="utf-8") as f:
            righe_voti = list(csv.DictReader(f))
        for r in righe_voti:
            if r["clip_id"] == clip_id:
                r["voto"] = nuovo_voto
        with open(percorso_voti, "w", newline="", encoding="utf-8") as f:
            scrittore = csv.DictWriter(f, fieldnames=["clip_id", "corpus",
                                                       "strato", "voto"])
            scrittore.writeheader()
            scrittore.writerows(righe_voti)

    if not da_fare:
        st.success("Tutti i casi limite sono etichettati — grazie! I voti sono "
                   "in data/etichette_manuali.csv e verranno usati al posto "
                   "degli score PANNs.")
        conteggio: dict[str, int] = {}
        for v in voti.values():
            conteggio[v] = conteggio.get(v, 0) + 1
        st.write({v: n for v, n in sorted(conteggio.items())})

        # --- Revisione dei voti "pianto" fuori donateacry -------------------
        # Semantica : "pianto" per il detector = pianto di NEONATO
        # o bambino piccolo. Il pianto di un adulto è un negativo difficile
        # (mamma che piange, TV) e va etichettato a parte, come i versi di
        # animali. Questa passata separa le quattro categorie.
        per_id = {c["clip_id"]: c for c in casi}
        da_rivedere = [cid for cid, v in voti.items()
                       if v == "pianto" and per_id.get(cid, {}).get("corpus")
                       not in ("donateacry", None)]
        if da_rivedere:
            st.markdown("---")
            st.markdown(f"##### 🔎 Revisione: {len(da_rivedere)} voti \"pianto\" "
                        f"da precisare")
            st.caption("Per il detector conta solo il pianto di **neonato/bambino "
                       "piccolo**: il pianto di un adulto o il guaito di un cane "
                       "sono *negativi difficili* (il detector NON deve scattare). "
                       "Riascolta e scegli la categoria precisa.")
            caso_rev = per_id[da_rivedere[0]]
            st.caption(f"corpus: {caso_rev['corpus']} · strato: {caso_rev['strato']} · "
                       f"score pianto: {float(caso_rev['p_comb']):.2f}")
            st.audio(str(RADICE / caso_rev["percorso"]))
            r1, r2, r3, r4 = st.columns(4)
            if r1.button("👶 Neonato/bimbo", use_container_width=True):
                riscrivi_voto(caso_rev["clip_id"], "pianto_neonato")
                st.rerun()
            if r2.button("🧑 Adulto", use_container_width=True):
                riscrivi_voto(caso_rev["clip_id"], "pianto_adulto")
                st.rerun()
            if r3.button("🐕 Animale", use_container_width=True):
                riscrivi_voto(caso_rev["clip_id"], "verso_animale")
                st.rerun()
            if r4.button("🚫 Altro", use_container_width=True):
                riscrivi_voto(caso_rev["clip_id"], "non_pianto")
                st.rerun()
        else:
            st.info("Revisione completata: tutte le etichette \"pianto\" sono "
                    "precise (neonato, adulto, animale o altro).")
        return

    # Un caso alla volta, dal primo non ancora votato
    caso = da_fare[0]
    spiegazioni = {
        "conflitto_gt": "FSD50K lo dichiara pianto, PANNs lo boccia: chi ha ragione?",
        "scoperta": "PANNs sente pianto ma l'etichetta ufficiale non c'è: è pianto vero?",
        "zona_grigia": "Score intermedio: il tuo voto calibra la soglia del triage.",
        "rumore_sospetto": "È nel pool di rumore dei sintetici: se c'è pianto va tolto!",
        "vocalsound_alto": "VocalSound non dovrebbe contenere pianti: questo score è alto.",
        "validazione_pianto_alto": "Tutti d'accordo che sia pianto: conferma (o smentisci).",
        "donateacry_dubbio": "Per donateacry è un pianto, ma PANNs dubita: è davvero "
                             "un pianto o un borbottio/rumore?",
        "freesound_grigio": "Candidato Freesound per il pool di pianto SSL: è pianto "
                            "vero di neonato/bimbo? (vota 'Pianto' solo in quel caso; "
                            "adulti, imitazioni ed effetti sonori sono 'Non pianto')",
    }
    # Voto atteso per gli strati di validazione: dichiarato ma modificabile
    voti_attesi = {"validazione_pianto_alto": "👶 Pianto",
                   "donateacry_dubbio": "👶 Pianto"}
    st.markdown(f"#### Caso {len(voti) + 1} di {len(casi)} — strato: `{caso['strato']}`")
    st.caption(spiegazioni.get(caso["strato"], ""))
    if caso["strato"] in voti_attesi:
        st.caption(f"Voto atteso: **{voti_attesi[caso['strato']]}** — premi quello "
                   f"per validare, oppure correggi se l'orecchio dice altro.")
    st.caption(f"corpus: {caso['corpus']} · score pianto combinato: "
               f"**{float(caso['p_comb']):.2f}** · etichetta AudioSet dominante: "
               f"*{caso['top_etichetta']}*")
    st.audio(str(RADICE / caso["percorso"]))

    def salva_voto(voto: str) -> None:
        """Appende il voto al CSV delle etichette manuali (crea l'header se nuovo)."""
        nuovo = not percorso_voti.exists()
        with open(percorso_voti, "a", newline="", encoding="utf-8") as f:
            scrittore = csv.DictWriter(f, fieldnames=["clip_id", "corpus", "strato",
                                                       "voto"])
            if nuovo:
                scrittore.writeheader()
            scrittore.writerow({"clip_id": caso["clip_id"], "corpus": caso["corpus"],
                                "strato": caso["strato"], "voto": voto})

    col1, col2, col3 = st.columns(3)
    if col1.button("👶 Pianto", use_container_width=True):
        salva_voto("pianto")
        st.rerun()
    if col2.button("🚫 Non pianto", use_container_width=True):
        salva_voto("non_pianto")
        st.rerun()
    if col3.button("🤷 Non sicuro", use_container_width=True):
        salva_voto("non_sicuro")
        st.rerun()

    # Correzione di un voto già dato: riscrive il CSV con il nuovo valore.
    # Limitata agli strati di validazione (voto atteso precompilato): sono
    # quelli dove ha senso ripensarci; per gli altri il primo ascolto fa fede.
    per_id = {c["clip_id"]: c for c in casi}
    strati_correggibili = ("validazione_pianto_alto", "donateacry_dubbio")
    correggibili = sorted(cid for cid in voti
                          if per_id.get(cid, {}).get("strato") in strati_correggibili)
    if correggibili:
        with st.expander(f"✏️ Correggi un voto di validazione ({len(correggibili)} votati)"):
            da_correggere = st.selectbox(
                "Clip votato", correggibili,
                format_func=lambda cid: f"{cid} — voto attuale: {voti[cid]}")
            caso_corr = per_id.get(da_correggere)
            if caso_corr:
                st.audio(str(RADICE / caso_corr["percorso"]))
            nuovo_voto = st.radio("Nuovo voto", ["pianto", "non_pianto", "non_sicuro"],
                                  horizontal=True, key="correzione_voto")
            if st.button("Salva la correzione"):
                with open(percorso_voti, newline="", encoding="utf-8") as f:
                    righe_voti = list(csv.DictReader(f))
                for r in righe_voti:
                    if r["clip_id"] == da_correggere:
                        r["voto"] = nuovo_voto
                with open(percorso_voti, "w", newline="", encoding="utf-8") as f:
                    scrittore = csv.DictWriter(f, fieldnames=["clip_id", "corpus",
                                                               "strato", "voto"])
                    scrittore.writeheader()
                    scrittore.writerows(righe_voti)
                st.rerun()


# ----------------------------------------------------------------------------
# Ingresso dell'app
# ----------------------------------------------------------------------------

def main() -> None:
    """Layout dell'app: due schede, esplorazione clip e statistiche dataset."""
    st.set_page_config(page_title="BabyCry — esplora dati", layout="wide")
    st.title("BabyCry — esplorazione dei dati")

    clip = carica_clip_disponibili()
    if not clip:
        st.error("Nessun dato trovato. Eseguire prima gli script 01 (download, "
                 "con --extract) e 02 (filtro licenze): l'app legge "
                 "data/licenses_manifest.csv, con fallback sulla scansione di data/raw.")
        return
    manifest_presente = (RADICE / "data" / "licenses_manifest.csv").exists()
    if not manifest_presente:
        st.warning("Manifest licenze assente: sto scandendo data/raw direttamente. "
                   "Per il training vale solo ciò che passa dallo script 02.")

    scheda1, scheda2, scheda3 = st.tabs(["🔍 Esplora clip", "📊 Statistiche dataset",
                                         "🏷️ Casi limite"])
    with scheda1:
        scheda_clip(clip)
    with scheda2:
        scheda_statistiche(clip)
    with scheda3:
        scheda_casi_limite()


main()
