"""Emotion cues in the user's own words ("je suis très énervé").

The voice alone often misses what the user states explicitly, especially before the
voice baseline is learned. This small lexicon spots explicit emotion words, handles
simple negations ("pas stressé") and intensifiers ("très triste"), and returns a
strength per label. It is a hint combined with the voice, never a judgement.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from voxemo.emotion.types import EmotionLabel

# Patterns are matched on lower-case text without accents, at the start of a word;
# the match is extended to the end of the word ("enerv" → "énervée").
LEXICONS: dict[str, dict[EmotionLabel, tuple[str, ...]]] = {
    "fr": {
        "stress": (
            r"stress",
            r"angoiss",
            r"anxi",
            r"inquiet",
            r"inquiete",
            r"inquietud",
            r"panique",
            r"nerveu",
            r"enerv",
            r"tendue?\b",
            r"sous pression",
            r"debord",
            r"submerg",
            r"colere",
            r"furieu",
            r"agac",
            r"irrit",
            r"j'en ai marre",
            r"ras le bol",
            r"peur",
            r"flipp",
            r"a bout\b",
            r"putain",
            r"(?:fait|fais|font) chier",
            r"chiant",
            r"(?:me|m') ?(?:saoule|soule|gonfle|enerve)",
            r"relou",
            r"merde\b",
            r"galere",
        ),
        "tristesse": (
            r"trist",
            r"deprim",
            r"malheureu",
            r"decourag",
            r"decue?s?\b",
            r"pleur",
            r"cafard",
            r"(?:pas|plus) (?:le )?moral",
            r"(?:suis|sens) (?:tout |si |tres )?seule?s?\b",
            r"(?:suis|sens) (?:vraiment |trop )?(?:nul|nulle|zero|0)\b",
            r"(?:zero|0) en\b",
            r"desesper",
            r"chagrin",
            r"abattue?\b",
            r"(?:ca|tout) va mal",
            r"(?:ca|rien) ne va (?:pas|plus)",
            r"echou",
        ),
        "joie": (
            r"content",
            r"heureu(?:x|se)\b",
            r"ravie?\b",
            r"genial",
            r"trop bien",
            r"joie",
            r"enchante",
            r"excellent",
            r"reussi",
            r"fiere?\b",
            r"youpi",
            r"(?:ca|tout) va (?:tres |super |vraiment )?bien",
            r"(?:ca|tout) s'est (?:tres |super |vraiment )?bien passe",
            r"(?:c'etait|c'etait|etait) (?:tres |super |vraiment )?bien passe",
            r"j'adore",
        ),
    },
    "en": {
        "stress": (
            r"stress",
            r"anxi",
            r"worr",
            r"nervous",
            r"panic",
            r"angry",
            r"annoyed",
            r"furious",
            r"overwhelm",
            r"scared",
            r"afraid",
            r"tense\b",
        ),
        "tristesse": (r"sad", r"depress", r"unhappy", r"lonely", r"down\b", r"hopeless", r"cry"),
        "joie": (r"happy", r"glad", r"great\b", r"excited", r"delighted", r"awesome", r"proud"),
    },
}
NEGATIONS: dict[str, frozenset[str]] = {
    "fr": frozenset({"pas", "plus", "jamais", "sans", "aucun", "aucune", "ni"}),
    "en": frozenset({"not", "never", "no", "without", "isn't", "don't", "aren't", "wasn't"}),
}
INTENSIFIERS: dict[str, frozenset[str]] = {
    "fr": frozenset(
        {"tres", "trop", "vraiment", "tellement", "super", "hyper", "completement", "grave"}
        | {"extremement", "totalement", "si"}
    ),
    "en": frozenset({"very", "really", "so", "extremely", "totally", "too"}),
}
NEGATED_AS: dict[EmotionLabel, EmotionLabel] = {"joie": "tristesse"}
"""A negated label that suggests another one ("pas réussi", "pas content")."""
NEGATED_STRENGTH = 0.5
NEGATION_WINDOW = 3  # words looked back, within the same clause
INTENSIFIER_WINDOW = 2
MAX_STRENGTH = 3.0
_CLAUSE_BREAK = re.compile(r"[,.;:!?…]|\b(?:mais|but)\b")
_WORD = re.compile(r"[\w']+")


@dataclass(frozen=True)
class TextCue:
    strengths: dict[EmotionLabel, float] = field(default_factory=dict)
    """Evidence per label: 1 per plain mention, 2 when intensified (capped)."""
    terms: tuple[str, ...] = ()
    """The words found, as written by the user."""

    def __bool__(self) -> bool:
        return bool(self.strengths)


def fold(text: str) -> str:
    """Lower-case, strip accents, normalize apostrophes; keeps one char per input char."""
    out = []
    for ch in text:
        decomposed = unicodedata.normalize("NFKD", ch.lower())
        base = "".join(c for c in decomposed if not unicodedata.combining(c))
        out.append(base[:1] if len(base) >= 1 else " ")
    return "".join(out).translate(str.maketrans("’`´", "'''"))


def _compile(patterns: tuple[str, ...]) -> re.Pattern[str]:
    return re.compile(r"(?<![\w])(?:" + "|".join(patterns) + r")[\w]*")


_COMPILED: dict[str, dict[EmotionLabel, re.Pattern[str]]] = {
    lang: {label: _compile(p) for label, p in lex.items()} for lang, lex in LEXICONS.items()
}


def _preceding_words(folded: str, start: int) -> list[str]:
    before = folded[:start]
    breaks = list(_CLAUSE_BREAK.finditer(before))
    clause = before[breaks[-1].end() :] if breaks else before
    return _WORD.findall(clause)


def _display_term(text: str, folded: str, start: int, end: int, negated: bool) -> str:
    term = text[start:end]
    if not negated:
        return term
    # Show the negation too ("pas réussi"), from the nearest negation word.
    words = list(_WORD.finditer(folded[:start]))
    for word in reversed(words[-NEGATION_WINDOW:]):
        if word.group(0) in _ALL_NEGATIONS:
            return text[word.start() : end]
    return term


_ALL_NEGATIONS = frozenset().union(*NEGATIONS.values())


def text_cues(text: str, language: str = "fr") -> TextCue:
    """Find explicit emotion words; empty for unknown languages."""
    lexicon = _COMPILED.get(language)
    if not lexicon or not text.strip():
        return TextCue()
    folded = fold(text)
    negations, intensifiers = NEGATIONS[language], INTENSIFIERS[language]
    strengths: dict[EmotionLabel, float] = {}
    terms: list[str] = []
    for label, regex in lexicon.items():
        for match in regex.finditer(folded):
            words = _preceding_words(folded, match.start())
            matched = match.group(0)
            built_in_negation = matched.split()[0] in negations  # e.g. "pas le moral"
            negated = not built_in_negation and any(
                w in negations for w in words[-NEGATION_WINDOW:]
            )
            target: EmotionLabel | None = label
            boost = 2.0 if any(w in intensifiers for w in words[-INTENSIFIER_WINDOW:]) else 1.0
            if negated:
                target, boost = NEGATED_AS.get(label), NEGATED_STRENGTH
            if target is None:
                continue
            strengths[target] = min(MAX_STRENGTH, strengths.get(target, 0.0) + boost)
            terms.append(_display_term(text, folded, match.start(), match.end(), negated))
    return TextCue(strengths, tuple(terms))
