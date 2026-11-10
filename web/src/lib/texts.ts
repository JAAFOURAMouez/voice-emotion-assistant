// User-facing text of the web interface (French).
export const T = {
  wordmark: "voxemo",
  tagline: "empreinte vocale",
  local: "tout reste sur cette machine",
  talk: "Maintenir pour parler",
  talkRecording: "Je vous écoute…",
  talkStop: "Relâcher pour envoyer",
  talkHint: "ou maintenez la barre d'espace · un appui bref démarre, un second arrête",
  thinking: "Je réfléchis…",
  idleNote: "Votre voix dessine cette forme. Parlez, elle vous écoute.",
  micDenied:
    "Le micro est refusé. Autorisez-le depuis l'icône à gauche de la barre d'adresse, puis réessayez.",
  micUnavailable:
    "Aucun micro utilisable. Hors de localhost, le navigateur n'autorise le micro qu'en HTTPS.",
  micRequesting: "Autorisez le micro pour commencer.",
  loading: "Chargement des modèles, un instant… La première fois, cela peut prendre quelques minutes.",
  offline: "Connexion au serveur perdue. Nouvelle tentative…",
  silence: "Je n'ai rien entendu. Parlez un peu plus près du micro, en maintenant la touche.",
  retry: "Réessayer",
  confidence: "confiance",
  nothingMarked: "rien de marqué · le plus proche :",
  estimate: "estimation",
  learning: "j'apprends encore votre voix",
  typed: "écrit",
  offerYes: "Oui, guidez-moi",
  offerNo: "Non merci",
  typeLabel: "ou écrire",
  typePlaceholder: "un message, puis Entrée",
  newConversation: "nouvelle conversation",
} as const;

export function decimal(value: number, digits = 2): string {
  return value.toFixed(digits).replace(".", ",");
}

/** French typography: a thin non-breaking space before ? ! ; : and inside « » (no orphan "?"). */
export function typeset(text: string): string {
  return text
    .replace(/\s+([?!;:»])/g, "\u202f$1")
    .replace(/«\s+/g, "«\u202f");
}
