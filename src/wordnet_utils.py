"""WordNet helpers for the Raganato WSD evaluation framework.

The framework annotates every target word with one or more sense-keys
(e.g. ``house%1:06:00::``). We use NLTK's WordNet interface to recover the
synset and gloss for every sense-key, and to enumerate all candidate senses
for a given lemma + part-of-speech.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

from nltk.corpus import wordnet as wn

# Built lazily — touching wn.NOUN at import time forces NLTK to load the
# WordNet zip, which fails before the user runs `nltk.download('wordnet')`.
_POS_MAP: dict | None = None


def _pos_map() -> dict:
    global _POS_MAP
    if _POS_MAP is None:
        _POS_MAP = {
            "NOUN": wn.NOUN, "VERB": wn.VERB, "ADJ": wn.ADJ, "ADV": wn.ADV,
            "n": wn.NOUN,    "v": wn.VERB,    "a": wn.ADJ,   "r": wn.ADV,
        }
    return _POS_MAP


@lru_cache(maxsize=None)
def lemma_from_key(sense_key: str):
    """Return the NLTK Lemma object for a WordNet sense-key, or None."""
    try:
        return wn.lemma_from_key(sense_key)
    except Exception:  # pragma: no cover - corrupt keys
        return None


@lru_cache(maxsize=None)
def gloss_for_key(sense_key: str) -> str | None:
    lem = lemma_from_key(sense_key)
    if lem is None:
        return None
    return lem.synset().definition()


@lru_cache(maxsize=None)
def candidate_keys(lemma: str, pos: str) -> tuple[str, ...]:
    """All WordNet sense-keys for (lemma, pos), preserving WordNet's
    canonical ordering (most-frequent sense first)."""
    wn_pos = _pos_map().get(pos)
    if wn_pos is None:
        return ()
    keys: List[str] = []
    for syn in wn.synsets(lemma, pos=wn_pos):
        for lem in syn.lemmas():
            if lem.name().lower().replace(" ", "_") == lemma.lower().replace(" ", "_"):
                keys.append(lem.key())
    # fall back: relax the exact-lemma match
    if not keys:
        for syn in wn.synsets(lemma, pos=wn_pos):
            keys.extend(lem.key() for lem in syn.lemmas())
    # de-duplicate while preserving order
    seen: set[str] = set()
    unique = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            unique.append(k)
    return tuple(unique)
