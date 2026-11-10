"""Detection of severe distress in a transcript.

Deliberately simple and over-inclusive: a false positive only means the assistant
points to support resources instead of suggesting a breathing exercise.
"""

from __future__ import annotations

import re
import unicodedata

HELPLINE_NUMBERS: tuple[str, ...] = ("3114", "15", "112")

_PATTERNS: tuple[str, ...] = (
    # French
    r"suicid",
    r"\bme (tuer|pendre|foutre en l'air|jeter (sous|du haut))",
    r"\b(envie|besoin|veux|vais|voudrais) d?'?en finir\b",
    r"\ben finir avec (la|ma) vie\b",
    r"\bmettre fin a (mes jours|ma vie)\b",
    r"\b(envie|veux|voudrais|prefere(rais)?) (de )?mourir\b",
    r"\b(plus|pas) (envie|la force) de vivre\b",
    r"\bne (sert|vaut) (a|plus) rien de vivre\b",
    r"\bme faire du mal\b",
    r"\bme (scarifier|mutiler|blesser expres)\b",
    r"\b(disparaitre|partir) pour toujours\b",
    # English
    r"\bkill(ing)? myself\b",
    r"\bend(ing)? (my|it all|my own) li(fe|ves)\b",
    r"\bwant(s)? to die\b",
    r"\b(hurt|harm|cut)(ing)? myself\b",
    r"\bself[- ]harm",
)
_REGEX = re.compile("|".join(f"(?:{p})" for p in _PATTERNS))


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    without_accents = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[’`´]", "'", without_accents)


def mentions_severe_distress(text: str) -> bool:
    """Return True if the text contains wording associated with severe distress."""
    return bool(_REGEX.search(_normalize(text)))


def mentions_helpline(text: str) -> bool:
    """Return True if the text already points to the national suicide-prevention line."""
    return "3114" in text
