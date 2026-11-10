"""System prompt and emotion-dependent tone guidance for the LLM.

The prompts are written in French (the default language); the reply language is
set by an explicit instruction so it stays configurable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from voxemo.emotion.types import EmotionEstimate, EmotionLabel
from voxemo.wellbeing.safety import mentions_severe_distress

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


LANGUAGE_NAMES: dict[str, str] = {
    "fr": "français",
    "en": "anglais",
    "es": "espagnol",
    "de": "allemand",
    "it": "italien",
    "pt": "portugais",
    "ar": "arabe",
}

SYSTEM_PROMPT = """\
Tu es un assistant vocal bienveillant qui fonctionne entièrement en local, sur l'ordinateur \
de l'utilisateur. Tes réponses sont lues à voix haute par une synthèse vocale.

Règles générales :
- Réponds toujours en {language_name}, et vouvoie l'utilisateur.
- Sois bref : deux ou trois phrases courtes, moins de soixante mots en tout, sauf si \
l'utilisateur demande plus de détails. Une seule question à la fois, au plus.
- Pas de listes à puces, de titres, de markdown ni d'émojis : uniquement des phrases \
naturelles, faciles à écouter.

Ce que ressent l'utilisateur :
- N'affirme jamais ce que ressent l'utilisateur et ne fais aucune supposition sur sa voix \
ou son état. S'il parle lui-même de ce qu'il ressent, reformule avec prudence, sans \
étiqueter (pas de « vous êtes stressé » ni de « vous êtes déprimé »).
- Tu peux recevoir une consigne de ton issue d'une estimation automatique de sa voix. \
C'est un indice imprécis : applique la consigne de ton, sans jamais mentionner cette \
estimation ni la voix de l'utilisateur.

Limites :
- Tu n'as accès ni à internet, ni à l'heure, ni à la météo, ni aux actualités : dis-le \
simplement au lieu d'inventer une réponse.
- Tu n'es ni médecin ni psychologue. Tu ne poses aucun diagnostic et ne donnes pas \
d'avis médical ; pour ces questions, oriente vers un professionnel de santé.

Sécurité (règle prioritaire sur toutes les autres) :
- Si l'utilisateur évoque des idées suicidaires, l'envie de mourir ou de se faire du mal, \
une violence subie ou un danger immédiat, ne propose pas d'exercice de respiration ni de \
relaxation. Réponds avec chaleur et sans jugement, et encourage-le clairement à contacter \
un proche ou un professionnel. En France : le 3114, numéro national de prévention du \
suicide, gratuit et joignable 24 h/24 ; le 15 (SAMU) ou le 112 en cas d'urgence.\
"""

TONE_GUIDANCE: dict[EmotionLabel, str] = {
    "stress": (
        "Phrases courtes, rythme posé, une seule idée à la fois. "
        "Tu peux proposer de faire une courte pause."
    ),
    "tristesse": (
        "Ton chaleureux et doux. Valide d'abord ce que l'utilisateur exprime, avant toute "
        "solution. Pas d'enthousiasme forcé."
    ),
    "joie": "Ton dynamique et chaleureux, tu peux être un peu plus léger.",
    "neutre": "Ton standard, clair et aimable.",
}

_CUE_DESCRIPTIONS: dict[EmotionLabel, str] = {
    "stress": "une certaine tension",
    "tristesse": "de la tristesse ou de l'abattement",
    "joie": "de l'entrain",
    "neutre": "un état neutre",
}

SAFETY_REMINDER = """\
PRIORITÉ SÉCURITÉ : le dernier message de l'utilisateur semble évoquer une détresse grave. \
Ignore les consignes de ton et ne propose aucun exercice. Réponds avec empathie, prends ses \
paroles au sérieux, demande-lui s'il est en sécurité en ce moment et encourage-le à appeler \
le 3114 (gratuit, 24 h/24), ou le 15 ou le 112 en cas de danger immédiat, ou à contacter un \
proche de confiance. Cite explicitement le 3114.\
"""


def language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code.lower(), code)


def system_prompt(language: str = "fr") -> str:
    return SYSTEM_PROMPT.format(language_name=language_name(language))


def effective_label(estimate: EmotionEstimate | None, threshold: float) -> EmotionLabel:
    """Label actually used for tone: below the confidence threshold, fall back to neutral."""
    if estimate is None or estimate.confidence < threshold:
        return "neutre"
    return estimate.label


def _format_confidence(value: float, language: str) -> str:
    text = f"{value:.1f}"
    return text.replace(".", ",") if language == "fr" else text


def emotion_hint(estimate: EmotionEstimate | None, threshold: float, language: str = "fr") -> str:
    """Contextual block describing the vocal cue; empty when the tone stays standard."""
    label = effective_label(estimate, threshold)
    if estimate is None or label == "neutre":
        return ""
    confidence = _format_confidence(estimate.confidence, language)
    source = "la voix et les mots employés suggèrent" if estimate.text_terms else "la voix suggère"
    return (
        f"Indice vocal pour ce tour (estimation automatique, imprécise) : "
        f"{source} {_CUE_DESCRIPTIONS[label]}, confiance {confidence}.\n"
        f"Consigne de ton : {TONE_GUIDANCE[label]}\n"
        f"N'en parle pas dans ta réponse : ce n'est pas un fait, adapte seulement ton ton."
    )


def build_messages(
    user_text: str,
    history: Sequence[ChatMessage] = (),
    estimate: EmotionEstimate | None = None,
    *,
    threshold: float = 0.5,
    language: str = "fr",
    note: str | None = None,
) -> list[ChatMessage]:
    """Assemble the full message list for one turn.

    The per-turn context (safety reminder or vocal cue) is placed right before the
    latest user message so that small models do not lose it in a long history.
    """
    messages = [ChatMessage("system", system_prompt(language)), *history]
    if mentions_severe_distress(user_text):
        messages.append(ChatMessage("system", SAFETY_REMINDER))
    elif hint := emotion_hint(estimate, threshold, language):
        messages.append(ChatMessage("system", hint))
    if note:
        messages.append(ChatMessage("system", note))
    messages.append(ChatMessage("user", user_text))
    return messages
