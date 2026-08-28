"""I sei candidati del bake-off dei pretext task (vedi README).

Tutti condividono lo stesso EncoderCompatto e lo stesso budget di passi:
cambia solo la testa (che si butta via dopo il pretraining) e il segnale di
addestramento. Le augmentation operano sul log-mel normalizzato per clip
[B, 1, 300 frame, 64 mel]: offset additivi nel dominio log (gain, attenuazioni
di banda) e miscele via logaddexp — non è fisica esatta dopo la normalizzazione,
ma è la pratica standard (à la SpecAugment) e conta la coerenza tra candidati.

Candidati:
- A FiltroIDSintetico: classificare quale dei 4 filtri è applicato a segnali
  sintetici armonici (la proposta di partenza, baseline).
- B FiltroIDReale: filtri su audio reale + regressione continua dei tagli.
- C Contrastive (SimSiam) con filtri/rumore/maschere come augmentation.
- D MaskedSpec: ricostruzione delle zone mascherate dello spettrogramma.
- E il candidato C più task ausiliaria sul verso del tempo.
- F Denoising: ricostruire lo spettrogramma pulito da quello sporcato.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from babycry.encoder import EncoderCompatto

# Geometria delle patch e della banda mel (coerente con configs/audio_frontend.yaml)
N_FRAME, N_MEL = 300, 64
F_MIN, F_MAX = 50.0, 8000.0


def _hz_a_bin(hz: float) -> int:
    """Converte una frequenza in Hz nell'indice della banda mel più vicina."""
    def mel(f: float) -> float:
        return 2595.0 * math.log10(1.0 + f / 700.0)
    frazione = (mel(hz) - mel(F_MIN)) / (mel(F_MAX) - mel(F_MIN))
    return int(round(frazione * (N_MEL - 1)))


# ---------------------------------------------------------------------------
# Filtri di banda nel dominio log-mel (usati da A, B e C)
# ---------------------------------------------------------------------------

def applica_filtro(x: torch.Tensor, tipo: torch.Tensor, taglio_a: torch.Tensor,
                   taglio_b: torch.Tensor, attenuazione: torch.Tensor) -> torch.Tensor:
    """Applica per-campione uno dei 4 filtri come attenuazione di banda.

    Tipi: 0 passa-basso (attenua sopra taglio_a), 1 passa-alto (attenua sotto
    taglio_a), 2 passa-banda (attenua fuori da [a, b]), 3 elimina-banda
    (attenua dentro [a, b]). L'attenuazione è una sottrazione nel dominio log.

    Args:
        x: batch [B, 1, T, M].
        tipo: interi [B] in {0..3}.
        taglio_a, taglio_b: indici di banda [B] (b >= a).
        attenuazione: intensità [B] (unità di log-mel normalizzato).
    """
    B = x.shape[0]
    bande = torch.arange(N_MEL, device=x.device).view(1, N_MEL).expand(B, N_MEL)
    a = taglio_a.view(B, 1)
    b = taglio_b.view(B, 1)
    dentro = (bande >= a) & (bande <= b)          # banda [a, b]
    sopra = bande > a
    maschere = torch.stack([
        sopra,                                     # 0: LP attenua sopra a
        bande < a,                                 # 1: HP attenua sotto a
        ~dentro,                                   # 2: BP attenua fuori [a,b]
        dentro,                                    # 3: BS attenua dentro [a,b]
    ], dim=0)                                      # [4, B, M]
    scelta = maschere[tipo, torch.arange(B, device=x.device)]  # [B, M]
    return x - attenuazione.view(B, 1, 1, 1) * scelta.view(B, 1, 1, N_MEL).float()


def filtro_casuale(B: int, device, rng: torch.Generator | None = None
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Campiona tipo e tagli casuali per un batch di filtri."""
    tipo = torch.randint(0, 4, (B,), device=device, generator=rng)
    t1 = torch.randint(8, N_MEL - 8, (B,), device=device, generator=rng)
    ampiezza = torch.randint(6, 24, (B,), device=device, generator=rng)
    t2 = torch.clamp(t1 + ampiezza, max=N_MEL - 1)
    attenuazione = 3.0 + 3.0 * torch.rand(B, device=device, generator=rng)
    return tipo, t1, t2, attenuazione


# ---------------------------------------------------------------------------
# Augmentation per i candidati contrastive (C, E)
# ---------------------------------------------------------------------------

def augmenta(x: torch.Tensor) -> torch.Tensor:
    """Vista augmentata: shift, gain, filtro, rumore (altro campione), maschere.

    Il pitch shift è approssimato con uno scorrimento di ±2 bande mel
    (≈1-2 semitoni nella zona della F0: il limite del README).
    """
    B = x.shape[0]
    y = x
    # Scorrimento in frequenza (pitch grossolano) e nel tempo
    y = torch.roll(y, shifts=int(torch.randint(-2, 3, (1,)).item()), dims=3)
    y = torch.roll(y, shifts=int(torch.randint(-50, 51, (1,)).item()), dims=2)
    # Gain (offset log) per campione
    y = y + (0.6 * torch.randn(B, 1, 1, 1, device=x.device))
    # Filtro di banda casuale su metà dei campioni
    tipo, t1, t2, att = filtro_casuale(B, x.device)
    filtrato = applica_filtro(y, tipo, t1, t2, att)
    usa_filtro = (torch.rand(B, 1, 1, 1, device=x.device) < 0.5).float()
    y = usa_filtro * filtrato + (1 - usa_filtro) * y
    # Rumore: miscela (logaddexp) con un altro campione del batch, attenuato
    perm = torch.randperm(B, device=x.device)
    offset = -3.0 + 2.0 * torch.rand(B, 1, 1, 1, device=x.device)
    y = torch.logaddexp(y, x[perm] + offset)
    # Maschera temporale moderata (fino a 40 frame)
    inizio = int(torch.randint(0, N_FRAME - 40, (1,)).item())
    durata = int(torch.randint(10, 41, (1,)).item())
    y[:, :, inizio:inizio + durata, :] = 0.0
    return y


# ---------------------------------------------------------------------------
# Decoder leggero per i candidati ricostruttivi (D, F) — si butta via dopo
# ---------------------------------------------------------------------------

class _Decoder(nn.Module):
    """Dalla mappa [B, 256, 19, 4] allo spettrogramma [B, 1, 300, 64]."""

    def __init__(self):
        super().__init__()
        self.rete = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
        )

    def forward(self, mappa: torch.Tensor) -> torch.Tensor:
        y = self.rete(mappa)                      # [B, 1, 304, 64]
        return y[:, :, :N_FRAME, :]


# ---------------------------------------------------------------------------
# Le teste MLP del contrastive (SimSiam)
# ---------------------------------------------------------------------------

class _Proiettore(nn.Module):
    """Proiettore SimSiam: 256 -> 256 -> 256 con BatchNorm."""

    def __init__(self, dim: int = 256):
        super().__init__()
        self.rete = nn.Sequential(
            nn.Linear(dim, dim, bias=False), nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True), nn.Linear(dim, dim, bias=False),
            nn.BatchNorm1d(dim, affine=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rete(x)


class _Predittore(nn.Module):
    """Predittore SimSiam: 256 -> 64 -> 256 (il collo stretto è essenziale)."""

    def __init__(self, dim: int = 256, collo: int = 64):
        super().__init__()
        self.rete = nn.Sequential(
            nn.Linear(dim, collo, bias=False), nn.BatchNorm1d(collo),
            nn.ReLU(inplace=True), nn.Linear(collo, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rete(x)


def _neg_cos(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Coseno negativo con stop-gradient sul secondo argomento (SimSiam)."""
    return -F.cosine_similarity(p, z.detach(), dim=1).mean()


# ---------------------------------------------------------------------------
# I candidati
# ---------------------------------------------------------------------------

class CandidatoBase(nn.Module):
    """Interfaccia comune: encoder condiviso + testa; passo(batch) -> loss."""

    def __init__(self, encoder: EncoderCompatto):
        super().__init__()
        self.encoder = encoder

    def passo(self, batch: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class CandidatoA(CandidatoBase):
    """A: filter-ID su segnali sintetici armonici (la proposta letterale)."""

    def __init__(self, encoder: EncoderCompatto):
        super().__init__(encoder)
        self.testa = nn.Linear(encoder.dim_embedding, 4)

    def _sintetizza(self, B: int, device) -> torch.Tensor:
        """Genera spettri armonici stazionari con AM lenta, già "normalizzati".

        F0 uniforme in 250-700 Hz (la banda del pianto), armoniche con
        decadimento, pavimento di rumore e modulazione d'ampiezza a 1-8 Hz.
        """
        bande = torch.arange(N_MEL, device=device).float()
        f0 = 250.0 + 450.0 * torch.rand(B, device=device)
        spettro = torch.full((B, N_MEL), -2.0, device=device)  # pavimento
        for k in range(1, 12):
            hz = f0 * k
            valide = hz < F_MAX
            if not valide.any():
                break
            centri = torch.tensor([_hz_a_bin(float(h)) for h in hz.tolist()],
                                  device=device).float()
            bump = torch.exp(-0.5 * ((bande.view(1, -1) - centri.view(-1, 1))
                                     / 1.2) ** 2)
            spettro = spettro + valide.view(-1, 1) * bump * (3.0 / math.sqrt(k))
        # AM lenta: 1-8 Hz sul tempo (frame rate 100 Hz)
        t = torch.arange(N_FRAME, device=device).float() / 100.0
        f_am = 1.0 + 7.0 * torch.rand(B, 1, device=device)
        am = 0.5 * torch.sin(2 * math.pi * f_am * t.view(1, -1))
        x = spettro.view(B, 1, 1, N_MEL) + am.view(B, 1, N_FRAME, 1)
        x = x + 0.3 * torch.randn_like(x)          # rumore di osservazione
        return (x - x.mean(dim=(1, 2, 3), keepdim=True)) / \
               (x.std(dim=(1, 2, 3), keepdim=True) + 1e-4)

    def passo(self, batch: torch.Tensor) -> torch.Tensor:
        B, device = batch.shape[0], batch.device   # del batch reale usa solo forma/device
        x = self._sintetizza(B, device)
        tipo, t1, t2, att = filtro_casuale(B, device)
        x = applica_filtro(x, tipo, t1, t2, att)
        logit = self.testa(self.encoder(x))
        return F.cross_entropy(logit, tipo)


class CandidatoB(CandidatoBase):
    """B: filter-ID rinforzato su audio reale + regressione continua dei tagli."""

    def __init__(self, encoder: EncoderCompatto):
        super().__init__(encoder)
        self.testa_tipo = nn.Linear(encoder.dim_embedding, 4)
        self.testa_tagli = nn.Linear(encoder.dim_embedding, 2)

    def passo(self, batch: torch.Tensor) -> torch.Tensor:
        tipo, t1, t2, att = filtro_casuale(batch.shape[0], batch.device)
        x = applica_filtro(batch, tipo, t1, t2, att)
        emb = self.encoder(x)
        bersaglio = torch.stack([t1.float(), t2.float()], dim=1) / (N_MEL - 1)
        return (F.cross_entropy(self.testa_tipo(emb), tipo)
                + F.mse_loss(torch.sigmoid(self.testa_tagli(emb)), bersaglio))


class CandidatoC(CandidatoBase):
    """C: contrastive SimSiam con filtri e rumore come augmentation (favorito)."""

    def __init__(self, encoder: EncoderCompatto):
        super().__init__(encoder)
        self.proiettore = _Proiettore(encoder.dim_embedding)
        self.predittore = _Predittore(encoder.dim_embedding)

    def _simsiam(self, batch: torch.Tensor) -> torch.Tensor:
        z1 = self.proiettore(self.encoder(augmenta(batch)))
        z2 = self.proiettore(self.encoder(augmenta(batch)))
        return 0.5 * (_neg_cos(self.predittore(z1), z2)
                      + _neg_cos(self.predittore(z2), z1))

    def passo(self, batch: torch.Tensor) -> torch.Tensor:
        return self._simsiam(batch)


class CandidatoD(CandidatoBase):
    """D: masked spectrogram modeling, riaddestrato in proprio (niente pesi NC)."""

    def __init__(self, encoder: EncoderCompatto):
        super().__init__(encoder)
        self.decoder = _Decoder()

    @staticmethod
    def _maschera(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Maschera blocchi tempo-frequenza (~30% dell'area) azzerandoli."""
        B = x.shape[0]
        maschera = torch.zeros_like(x, dtype=torch.bool)
        for _ in range(4):
            t0 = torch.randint(0, N_FRAME - 40, (B,))
            m0 = torch.randint(0, N_MEL - 16, (B,))
            for i in range(B):
                maschera[i, :, t0[i]:t0[i] + 40, m0[i]:m0[i] + 16] = True
        return x.masked_fill(maschera, 0.0), maschera

    def passo(self, batch: torch.Tensor) -> torch.Tensor:
        mascherato, maschera = self._maschera(batch)
        ricostruito = self.decoder(self.encoder.mappa(mascherato))
        return F.mse_loss(ricostruito[maschera], batch[maschera])


class CandidatoE(CandidatoC):
    """E: il candidato C più task ausiliaria sul verso del tempo (vedi README)."""

    def __init__(self, encoder: EncoderCompatto):
        super().__init__(encoder)
        self.testa_tempo = nn.Linear(encoder.dim_embedding, 1)

    def passo(self, batch: torch.Tensor) -> torch.Tensor:
        B = batch.shape[0]
        # Task ausiliaria: metà batch invertito nel tempo, la testa lo riconosce.
        # Forza la sensibilità alla direzione temporale (attacchi vs code).
        invertito = torch.rand(B, device=batch.device) < 0.5
        x = torch.where(invertito.view(B, 1, 1, 1), batch.flip(dims=(2,)), batch)
        logit = self.testa_tempo(self.encoder(x)).squeeze(1)
        ausiliaria = F.binary_cross_entropy_with_logits(logit, invertito.float())
        return self._simsiam(batch) + 0.2 * ausiliaria


class CandidatoF(CandidatoBase):
    """F: denoising — ricostruire lo spettrogramma pulito da quello sporcato.

    La corruzione usa un ALTRO campione del batch come rumore (il pool di
    pretraining è fatto di suoni domestici: la miscela è realistica) più un
    jitter di gain. Aggiunto nella 0.8 come risposta appresa al rumore.
    """

    def __init__(self, encoder: EncoderCompatto):
        super().__init__(encoder)
        self.decoder = _Decoder()

    def passo(self, batch: torch.Tensor) -> torch.Tensor:
        B = batch.shape[0]
        perm = torch.randperm(B, device=batch.device)
        offset = -2.0 + 2.5 * torch.rand(B, 1, 1, 1, device=batch.device)
        sporco = torch.logaddexp(batch, batch[perm] + offset)
        sporco = sporco + 0.4 * torch.randn(B, 1, 1, 1, device=batch.device)
        ricostruito = self.decoder(self.encoder.mappa(sporco))
        return F.mse_loss(ricostruito, batch)


CANDIDATI = {"A": CandidatoA, "B": CandidatoB, "C": CandidatoC,
             "D": CandidatoD, "E": CandidatoE, "F": CandidatoF}
