# Self-Supervised Pretext Tasks for Infant Cry Analysis

A single-GPU, fully reproducible study of self-supervised learning (SSL) for
infant cry acoustics, built exclusively on license-verified public data. The
project independently confirms key findings of Gorin et al. (Ubenwa Health,
ICASSP 2023) with a ~70× smaller encoder, and contributes a negative result
the field has been missing: **under honest subject-wise evaluation, the only
public cry-reason dataset (donateacry) does not support cry-reason
classification — and the 90%+ accuracies reported on it in the literature are
reproducible artifacts of clip-level data leakage.**

*Code and comments are in Italian; this README is the English entry point.*

---

## TL;DR — four findings

1. **Reconstructive pretext tasks win for cry detection.** In a controlled
   6-task bake-off (same 1.17M-parameter encoder, same step budget, same
   evaluation), masked-spectrogram modeling reaches **0.988 AUC** on
   cry/non-cry with a *linear probe* and subject-wise splits — without ever
   seeing a single cry during pretraining. Ranking: masked-spec 0.988 >
   denoising 0.982 > SimSiam 0.972 ≈ SimSiam+temporal 0.967 ≫ filter-ID on
   real audio 0.880 ≫ filter-ID on synthetic signals 0.793.
2. **Cry *reason* is not recoverable from donateacry under subject-wise
   splits.** Every encoder — including a frozen **HuBERT-base (94M params)**
   used as a capacity control — sits at chance (AUC 0.38–0.54, 5 classes).
   Domain adaptation on 1.8 h of real cries does not move it. End-to-end
   fine-tuning does not move it (best arm: 0.49). The bottleneck is the
   labels, not the representation.
3. **The leakage that explains the literature, quantified.** Switching from
   subject-wise to clip-wise (leaky) splits inflates the same frozen
   embeddings from ~0.40 to only ~0.60 AUC; published 90%+ results
   additionally train end-to-end on leaked features. Reproducing the
   leakage mechanism in-house puts a number on how much of the published
   performance is subject identity rather than cry semantics.
4. **Augmentation does not substitute subjects.** A 20× augmentation of the
   labeled set (PSOLA-vocoder speaker perturbation + noise mixing, 21 h of
   synthetic variants, class-balanced) leaves cross-subject AUC unchanged —
   direct evidence that the effective sample size for this task is the
   number of *infants*, not the number of clips or hours.

---

## Motivation

The reference point is Gorin et al., *Self-supervised learning for infant
cry analysis* (ICASSP 2023 workshops, [arXiv:2305.01578](https://arxiv.org/abs/2305.01578)):
SSL pretraining + cry-domain adaptation reaches **74.5 AUC** on 3-class cry
triggers, using a private clinical corpus (~1,150 recordings, ~900 newborns,
staff-assigned labels) and an 80M-parameter CNN14. Two questions this study
asks independently, with much smaller means:

- Do their conclusions (SSL > supervised transfer; domain labels matter
  more than architecture) hold with a compact encoder and public data only?
- Which *pretext task* actually earns its keep at this scale? The original
  paper commits to SimCLR; here six alternatives compete under a fixed
  budget, including two hypotheses about frequency-band sensitivity that
  deserve a fair trial rather than an opinion.

## Data (all licenses verified per clip)

| Corpus | Clips kept | Hours | License filter |
|---|---|---|---|
| FSD50K | 43,379 | 90.1 | CC0 + CC BY (7,818 CC-BY-NC excluded) |
| VocalSound | 20,985 | 24.4 | CC BY-SA 4.0 |
| donateacry (cleaned) | 447 → 433 | 0.9 | ODbL/DbCL; 14 spurious clips removed by ear |
| Freesound direct queries | 204 | 2.1 | CC0 + CC BY, deduplicated against FSD50K |
| Synthetic variants | 10,932 | 21.0 | derived from donateacry (see Augmentation) |

The pipeline (scripts `01`–`02`) enforces licensing *in code*: every admitted
clip lands in `data/licenses_manifest.csv` with source, author, license and
link; anything unverifiable is dropped and counted. Audio is cached as
memory-mapped float16 log-mel (64 bands, 16 kHz, 25 ms / 10 ms) by script
`03`.

**Human-verified labels.** A PANNs Cnn14 tagger (CC BY 4.0 checkpoint)
triages the pool; 127 borderline cases selected by stratified value-of-
information (conflicts with FSD50K ground truth, gray zone for threshold
calibration, contamination sentinels) were labeled by ear through a small
Streamlit tool, with a second pass separating infant cries from **adult
cries** and animal sounds. Notable finding: about half of the FSD50K clips
tagged "Baby cry" that survived triage are actually *adults* crying — they
became hard negatives. 11 of the 15 lowest-scoring donateacry "cries" turned
out not to be cries at all.

## Method

**Encoder.** A 4-stage CNN (two 3×3 convs per stage, BN, stride-2), global
average pooling, 256-d embedding — 1.17M parameters, chosen deliberately at
edge-deployable scale. Same encoder for every candidate; only the (discarded)
head changes.

**The six pretext tasks** (script `08`, `src/babycry/pretext.py`):

| | Task | Rationale / hypothesis under test |
|---|---|---|
| A | Filter-ID on synthetic harmonic signals | Can band-filter classification on *synthetic* stationary signals teach frequency awareness? (Hypothesis: no — shortcut learning on band energies, no natural statistics.) |
| B | Filter-ID on real audio + continuous cutoff regression | Same idea, but with real-audio statistics and a harder regression target. |
| C | SimSiam, with band-filters *as augmentation* plus noise mixing, masking, pitch/time shifts | The "filters belong in the augmentation, not in the label" counter-hypothesis. |
| D | Masked spectrogram modeling (trained from scratch; no NC-licensed weights) | Reconstruction in input space. |
| E | C + auxiliary time-direction classification | Force sensitivity to temporal dynamics (attack/decay asymmetry). |
| F | Denoising (reconstruct clean log-mel from a mix with another sample) | Noise robustness as a *learned* property, no runtime denoiser. |

The filter-ID line (A/B) is the project's own hypothesis about teaching
frequency-band structure explicitly; the bake-off was designed so that its
failure modes predicted on paper (shortcut solutions from mean spectra,
static tasks ignoring temporal dynamics, poor synthetic-to-real transfer)
would be measured rather than argued. All three predictions materialized —
and the same instinct survives in C, where the filters work well as
augmentation.

**Evaluation protocol (the part most of the literature gets wrong).**
Every split is by *subject* (donateacry contributor prefix, Freesound
uploader), never by clip — including the SSL stages: domain adaptation is
re-run **per fold** on the training subjects only, because representation-
level contamination is leakage just like label leakage. Metrics: macro
one-vs-rest AUC, balanced accuracy, ECE, via StratifiedGroupKFold (5 folds).
A clip-wise (deliberately leaky) variant is computed once, as a diagnostic.

**Augmentation** (script `04`): each donateacry clip is expanded with
(a) PSOLA vocoder variants — moderate F0 shifts (±0.5–3 st) plus formant
scaling ×0.88–1.12, simulating different vocal-tract sizes while preserving
the temporal gesture exactly; and (b) mixes with domestic noise drawn from
the license-clean FSD50K pool at 3–20 dB SNR. Synthetic variants inherit the
source infant's identity for splitting purposes (they are *not* new
subjects), and evaluation only ever uses real audio.

## Results

**Cry/non-cry linear probe** (541 verified positives, 428 negatives incl.
30 hard ones; subject-wise 5-fold):

| Pretext | AUC |
|---|---|
| D — masked spectrogram | **0.988 ± 0.004** |
| F — denoising | 0.982 ± 0.006 |
| C — SimSiam + filter augmentation | 0.972 ± 0.015 |
| E — C + temporal aux | 0.967 ± 0.020 |
| B — filter-ID (real) | 0.880 ± 0.024 |
| A — filter-ID (synthetic) | 0.793 ± 0.040 |

**Cry-reason probes on donateacry** (5 classes, subject-wise):

| Setting | AUC macro |
|---|---|
| Best pretext, linear probe | 0.539 ± 0.034 (E; all others 0.38–0.47) |
| After SSL adaptation on 1.84 h of cries (per-fold) | 0.33–0.45 |
| Frozen **HuBERT-base** (capacity control) | 0.42 |
| Same embeddings, **clip-wise leaky split** | 0.58–0.61 (HuBERT 0.61) |
| End-to-end fine-tuning, best of 4 augmentation arms | 0.49 ± 0.11 |
| Reference: Gorin et al., clinical labels, 3 classes | 74.5 |

The fine-tuning ablation (real-only / +noise / +vocoder / +both:
0.47 / 0.46 / 0.43 / 0.49) shows the augmentation arms statistically
indistinguishable — finding 4 above.

## What is confirmed, and what is new

**Independently confirmed** (public data, 1/70 the parameters, consumer
laptop GPU): SSL pretraining produces strong transferable cry
representations; the decisive ingredient for *reason* classification is
label quality on in-domain data, not architecture or compute — Gorin et
al.'s clinical-label pipeline remains the only demonstrated route to
above-chance reason classification.

**New here:** (1) the controlled six-task pretext comparison at fixed
budget, with the reconstructive family clearly ahead at this scale; (2) the
donateacry negative result with a model-capacity control, plus the
quantified leaky-vs-grouped gap that explains the inflated literature;
(3) the per-fold adaptation protocol for leakage-free SSL evaluation;
(4) the subject-count vs clip-count evidence from the vocoder/noise
augmentation ablation; (5) an ear-verified hard-negative set (adult cries
mislabeled as infant cries in FSD50K ground truth).

## Reproducing

Windows / PowerShell, Python 3.12, single NVIDIA GPU (developed on an RTX
4060 Laptop, 8 GB). Total compute: a few evenings.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python scripts/01_download_corpora.py --extract   # ~25 GB, checksums + resume
python scripts/02_filter_licenses.py              # license manifest
python scripts/03_build_mel_cache.py              # memory-mapped log-mel cache
python scripts/04_genera_sintetici.py             # synthetic variants (optional)
python scripts/05_triage_panns.py                 # PANNs triage
python scripts/06_seleziona_casi_limite.py        # borderline-case selection
# ... human labeling in the Streamlit tool:
#   streamlit run scripts/esplora_dati_app.py
python scripts/07_consolida_etichette.py          # consolidated label artifacts
python scripts/08_bakeoff_pretrain.py --candidato A..F
python scripts/09_bakeoff_probe.py                # bake-off arbitration
python scripts/10_adattamento_pianto.py --candidato D|F|E|C
python scripts/11_probe_teacher.py                # HuBERT capacity control
python scripts/12_finetuning.py --sintetici nessuno|rumore|vocoder|entrambi
```

Seeds are fixed in `configs/`; all reported numbers are in `results/*.csv`.
Audio, caches and checkpoints are not redistributed (see licenses below) —
the pipeline rebuilds everything from the original sources.

## Dataset licenses and attribution

This repository redistributes **no audio**. Sources, each filtered per clip:
[FSD50K](https://zenodo.org/record/4060432) (CC0/CC BY per clip; ground
truth CC BY 4.0), [VocalSound](https://github.com/YuanGongND/vocalsound)
(CC BY-SA 4.0), [donateacry-corpus](https://github.com/gveres/donateacry-corpus)
(ODbL/DbCL), [Freesound](https://freesound.org) direct queries (CC0/CC BY;
per-clip attribution in `data/licenses_manifest.csv`), and the
[PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn) Cnn14
checkpoint (CC BY 4.0) for triage only. Development-only tooling includes
praat-parselmouth (GPL-3), used to generate training data and not part of
any model.

## Limitations and future work

- The reason-classification negative result is a statement about
  **donateacry's labels** (in-the-moment parent guesses, 84% "hungry",
  minority classes of 8–25 clips), not about the task: clinical-grade or
  retrospective outcome-based labels remain the open requirement.
- Mean-pooled clip embeddings discard temporal structure; the only
  above-chance pretext was the one with a temporal auxiliary task. A
  latent-space predictive objective over time (JEPA-style) is the natural
  next candidate for the bake-off harness once better-labeled data exists.
- Cry detection is reported as ranking quality (AUC); picking an operating
  point (false alarms per night at fixed recall) and a systematic multi-SNR
  evaluation are mechanical next steps the harness already supports.

## References

- Gorin et al., *Self-supervised learning for infant cry analysis*, ICASSP
  2023 SASB workshop — [arXiv:2305.01578](https://arxiv.org/abs/2305.01578)
- Fonseca et al., *FSD50K: an open dataset of human-labeled sound events* —
  [arXiv:2010.00475](https://arxiv.org/abs/2010.00475)
- Gong et al., *VocalSound: a dataset for improved human vocal sounds
  recognition* — [arXiv:2205.03433](https://arxiv.org/abs/2205.03433)
- Kong et al., *PANNs: large-scale pretrained audio neural networks* —
  [arXiv:1912.10211](https://arxiv.org/abs/1912.10211)
- Chen et al., *SimSiam: exploring simple Siamese representation learning* —
  [arXiv:2011.10566](https://arxiv.org/abs/2011.10566)
