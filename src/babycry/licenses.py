"""Normalizzazione e filtro delle licenze Creative Commons per clip.

Regola di progetto : sono ammesse solo licenze permissive verificate — CC0 e CC BY (FSD50K, Freesound),
CC BY-SA (VocalSound), ODbL/DbCL (donateacry). Un clip senza licenza
verificabile viene ESCLUSO dal codice, non dalla buona volontà: qualunque
valore non riconosciuto da queste funzioni finisce tra gli scartati.
"""

from __future__ import annotations

# Licenze ammesse nella pipeline, in forma normalizzata
LICENZE_AMMESSE = frozenset({
    "CC0",           # dominio pubblico, nessun obbligo
    "CC BY",         # attribuzione obbligatoria (finisce nel manifest)
    "CC BY-SA",      # attribuzione + share-alike (VocalSound)
    "ODbL/DbCL",     # donateacry: database ODbL, contenuti DbCL, con attribuzione
})

# Mappa da sottostringhe degli URL Creative Commons alla forma normalizzata.
# L'ordine conta: i pattern più specifici (by-nc, by-sa) vanno controllati
# prima di quello generico "licenses/by/".
_PATTERN_URL = [
    ("publicdomain/zero", "CC0"),
    ("licenses/by-nc-sa", "CC BY-NC-SA"),
    ("licenses/by-nc-nd", "CC BY-NC-ND"),
    ("licenses/by-nc", "CC BY-NC"),
    ("licenses/by-nd", "CC BY-ND"),
    ("licenses/by-sa", "CC BY-SA"),
    ("licenses/by", "CC BY"),
    ("licenses/sampling+", "CC Sampling+"),
]


def normalizza_licenza(valore: str | None) -> str | None:
    """Normalizza una licenza espressa come URL o come nome libero.

    Args:
        valore: URL creativecommons.org (come nei metadati FSD50K/Freesound)
            oppure nome testuale ("CC BY 4.0", "ODbL"...).

    Returns:
        Forma normalizzata (es. "CC BY", "CC0", "CC BY-NC") oppure None se
        la licenza non è riconoscibile: in quel caso il clip va scartato.
    """
    if not valore:
        return None
    testo = valore.strip().lower()

    # Caso URL creativecommons.org
    for pattern, normalizzata in _PATTERN_URL:
        if pattern in testo:
            return normalizzata

    # Caso nome testuale (senza numero di versione)
    if testo.startswith("cc0") or "public domain" in testo:
        return "CC0"
    if testo.startswith("cc by-nc"):
        return "CC BY-NC"
    if testo.startswith("cc by-sa"):
        return "CC BY-SA"
    if testo.startswith("cc by"):
        return "CC BY"
    if "odbl" in testo or "dbcl" in testo:
        return "ODbL/DbCL"

    # Licenza non riconosciuta: nessuna indulgenza, si scarta
    return None


def licenza_ammessa(valore: str | None) -> tuple[bool, str | None]:
    """Verifica se una licenza (URL o nome) è ammessa nella pipeline.

    Args:
        valore: licenza grezza come nei metadati del corpus.

    Returns:
        Coppia (ammessa, forma_normalizzata). Se la licenza non è
        riconoscibile la coppia è (False, None).
    """
    normalizzata = normalizza_licenza(valore)
    return (normalizzata in LICENZE_AMMESSE, normalizzata)
