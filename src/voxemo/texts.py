"""User-facing strings (French by default), grouped in one place."""

from __future__ import annotations

# --- LLM errors ---
OLLAMA_UNREACHABLE = (
    "Impossible de joindre Ollama sur {host}. Vérifiez qu'il est lancé (commande : ollama serve)."
)
OLLAMA_MODEL_MISSING = (
    "Le modèle « {model} » n'est pas installé dans Ollama. Installez-le avec : ollama pull {model}"
)
OLLAMA_TIMEOUT = "Ollama met trop de temps à répondre (plus de {timeout:.0f} s)."
OLLAMA_ERROR = "Ollama a renvoyé une erreur : {detail}"

# --- Safety ---
SAFETY_FALLBACK = (
    "Ce que vous traversez semble très lourd, et vous n'avez pas à le porter seul. "
    "Parler à un proche ou à un professionnel peut vraiment aider : le 3114 est joignable "
    "gratuitement, jour et nuit. En cas de danger immédiat, appelez le 15 ou le 112."
)

# --- Text chat (command line) ---
CLI_WELCOME = (
    "Mode texte — tapez votre message.\n"
    "Commandes : /emotion <neutre|joie|tristesse|stress> [confiance], /reset, /quit"
)
CLI_PROMPT = "Vous : "
CLI_ASSISTANT = "Assistant : "
CLI_EMOTION_SET = "Indice vocal simulé : {label} (confiance {confidence})."
CLI_EMOTION_CLEARED = "Indice vocal simulé désactivé."
CLI_EMOTION_USAGE = "Usage : /emotion <neutre|joie|tristesse|stress> [confiance entre 0 et 1]"
CLI_RESET = "Conversation réinitialisée."
CLI_UNKNOWN_COMMAND = "Commande inconnue : {command}"
CLI_GOODBYE = "À bientôt."

# --- Audio input ---
PORTAUDIO_MISSING = (
    "La bibliothèque PortAudio est introuvable : le micro ne peut pas être utilisé. "
    "Installez-la avec « sudo apt install libportaudio2 » (Linux) ou « brew install portaudio » "
    "(macOS)."
)
MICROPHONE_UNAVAILABLE = "Impossible d'ouvrir le micro : {detail}"

# --- Transcription (command line) ---
STT_LISTENING = "Je vous écoute… (Ctrl+C pour quitter)"
STT_NO_SPEECH = "Je n'ai rien entendu, je continue d'écouter."
STT_NOTHING_UNDERSTOOD = "(parole détectée, mais rien de compréhensible)"
STT_RESULT = "[{start:.1f} s, {duration:.1f} s, {language}, transcrit en {elapsed:.1f} s] {text}"

# --- Emotion analysis (command line) ---
EMOTION_RESULT = (
    "[{start:.1f} s] émotion estimée : {label} (confiance {confidence:.2f}, "
    "activation {arousal:.2f}) | {scores} | pitch {pitch:.0f}±{pitch_std:.0f} Hz, "
    "débit {rate:.1f}/s, pauses {pauses:.0%} | {elapsed:.1f} s"
)

# --- Voice assistant (command line) ---
PIPELINE_LOADING = "Chargement des modèles (la première fois, cela peut prendre une minute)…"
PIPELINE_READY = "Prêt (modèles chargés en {elapsed:.0f} s)."
PIPELINE_WARMUP_WARNING = "Attention : {detail}"
PIPELINE_USER = "Vous : {text}"
PIPELINE_CUES = (
    "  (émotion estimée : {label}{detail} · {calibration}transcription {stt} s, voix {emotion} s)"
)
PIPELINE_CUE_CONFIDENCE = ", confiance {confidence}"
PIPELINE_CUE_NOTHING_MARKED = ", rien de marqué (le plus proche : {label} {score})"
PIPELINE_CUE_WORDS = ", mots relevés : {terms}"
PIPELINE_CUE_CALIBRATION = "apprentissage de votre voix {n}/{total} · "
PIPELINE_DETAILS = (
    "  [scores {scores} | modèle brut {model} | activation {arousal} | "
    "pitch {pitch:.0f}±{pitch_std:.0f} Hz, débit {rate}/s, pauses {pauses:.0%}]"
)
PIPELINE_AUDIO_SAVED = "  [audio enregistré : {path}]"
PIPELINE_STORING_AUDIO = "L'audio de chaque tour est enregistré dans {path} (STORE_RAW_AUDIO=true)."
PIPELINE_CUES_UNAVAILABLE = "  (indices vocaux indisponibles · transcription {stt} s)"
PIPELINE_LATENCY = "  (début de réponse {first} s après la fin de votre phrase, total {total} s)"
PIPELINE_HISTORY = (
    "Historique enregistré localement dans {path} ({n} échanges précédents servent de "
    "référence pour votre voix ; --no-history pour ne rien enregistrer)."
)

# --- History and trends (command line) ---
WEEKDAYS = ("lun.", "mar.", "mer.", "jeu.", "ven.", "sam.", "dim.")
TRACKING_EMPTY = "Aucun historique pour l'instant ({path})."
TRACKING_SUMMARY = "{interactions} échanges enregistrés sur {sessions} session(s) ({path})."
TRACKING_DISCLAIMER = (
    "Ces chiffres décrivent des émotions estimées à partir d'indices vocaux et des mots "
    "employés, pas un diagnostic."
)
TRACKING_SHARES = "Répartition des émotions estimées : {shares}"
TRACKING_DAYS_HEADER = "7 derniers jours :"
TRACKING_DAY = "  {weekday} {day:%d/%m} : {count} échange(s){detail}"
TRACKING_TREND_HEADER = "Tendance (score moyen des 7 derniers jours, comparé aux 7 précédents) :"
TRACKING_TREND = "  {label} : {recent} (avant : {previous}), {direction}"
TRACKING_TREND_UNKNOWN = "Tendance sur 7 jours : pas encore assez d'échanges pour comparer."
TRACKING_CONFIRM_CLEAR = (
    "Effacer tout l'historique ({count} échanges) ? Tapez « oui » pour confirmer : "
)
TRACKING_CLEARED = "Historique effacé ({count} échanges supprimés)."
TRACKING_CLEAR_CANCELLED = "Rien n'a été effacé."

# --- Exercises ---
EXERCISE_REASONS = {
    "stress": "Vos derniers échanges semblent un peu tendus, mais je peux me tromper.",
    "tristesse": "Vos derniers échanges semblent un peu lourds, mais je peux me tromper.",
}
EXERCISE_OFFER = (
    "{reason} Si vous en avez envie, je peux vous guider dans un court exercice : "
    "{name} ({duration}). Voulez-vous essayer ? Vous pouvez tout à fait dire non."
)
EXERCISE_OFFER_NOTE = (
    "Après ta réponse, l'application proposera elle-même un court exercice de relaxation à "
    "l'utilisateur. N'en propose pas toi-même et ne termine pas par une question."
)
EXERCISE_ACCEPT_BUTTON = "Oui, guidez-moi"
EXERCISE_DECLINE_BUTTON = "Non merci"
EXERCISE_DECLINED = "D'accord, pas de souci. Je reste là si vous voulez continuer à discuter."
EXERCISE_STARTING = "Très bien. {name}, {duration}. {summary} {caution}"
EXERCISE_CYCLE = "  Cycle {n}/{total}"
EXERCISE_STEP = "  {instruction}…"
EXERCISE_DONE = "C'est terminé. Prenez un instant pour remarquer comment vous vous sentez."
EXERCISE_STOPPED = "Exercice interrompu, ce n'est pas grave. On peut reprendre la discussion."
EXERCISE_HISTORY_NOTE = "(Exercice guidé : {name}.)"
EXERCISE_LIST_ITEM = "  {id:<24} {name} ({duration})"
EXERCISE_LIST_HEADER = "Exercices disponibles (python -m voxemo.wellbeing <nom>) :"

# --- Text-to-speech ---
TTS_UNAVAILABLE = "Synthèse vocale désactivée : {detail}"
TTS_OUTPUT_UNAVAILABLE = "Impossible d'utiliser le haut-parleur : {detail}"
TTS_DOWNLOADING = "Téléchargement de la voix {voice} (une seule fois)…"

# --- Web interface ---
UI_NOT_UNDERSTOOD = "Je n'ai pas compris ce qui a été dit. Pouvez-vous répéter ?"
UI_PRIVACY = (
    "Tout reste sur cette machine : l'audio n'est pas conservé, seuls la transcription, "
    "la réponse et les scores estimés sont enregistrés localement."
)

# --- Web API ---
API_LOADING = "Chargement des modèles en cours, un instant…"
API_LOAD_FAILED = "Les modèles n'ont pas pu être chargés : {detail}"
API_TOO_LONG = "Message trop long : {max_s:.0f} secondes au maximum."
API_BAD_MESSAGE = "Message non reconnu."
API_BUSY = "Un tour est déjà en cours."
SIMULATED_HISTORY = "Exemple simulé : ces strates ne sont pas les vôtres."

# Cautious phrasing of an estimate (never a diagnosis)
EMOTION_PHRASES = {
    "stress": "Voix plus tendue que d'habitude",
    "tristesse": "Voix plus basse et plus lente que d'habitude",
    "joie": "Voix plus vive que d'habitude",
    "neutre": "Voix comme d'habitude",
}
EMOTION_PHRASE_WORDS = "vos mots vont dans ce sens"
EMOTION_PHRASE_LEARNING = "j'apprends encore votre voix"
