// User-facing text of the web interface (French).
export const T = {
  wordmark: "voxemo",
  tagline: "empreinte vocale",
  local: "tout reste sur cette machine",
  localRemote: "tout reste sur votre ordinateur",
  cloudNotice: "texte envoyé à {provider} · audio et historique restent chez vous",
  modelLink: "modèle · {model}",
  settings: "réglages",
  settingsClose: "fermer",
  serverTitle: "serveur",
  serverHelp:
    "L'adresse de votre ordinateur, par son tunnel HTTPS (par exemple https://mon-pc.mon-reseau.ts.net). Laissez vide si cette page est servie par votre ordinateur.",
  serverAddress: "adresse",
  accessCode: "code d'accès",
  accessCodeHelp: "La valeur de ACCESS_TOKEN dans le fichier .env de votre ordinateur. Gardée dans ce navigateur seulement.",
  connect: "se connecter",
  serverNeeded: "Cette page est hébergée en ligne : indiquez l'adresse de votre ordinateur pour commencer.",
  accessDenied: "Code d'accès refusé. Saisissez celui défini sur votre ordinateur (ACCESS_TOKEN).",
  offlineRemote: "Votre ordinateur ne répond pas. Est-il allumé, avec voxemo et le tunnel lancés ? Nouvelle tentative…",
  badAddress: "Adresse invalide : une adresse https complète est attendue.",
  modelTitle: "modèle de langage",
  modelLocal: "sur cet ordinateur · Ollama",
  modelCloud: "fournisseur externe · votre clé",
  modelNone: "Aucun modèle trouvé dans Ollama. Installez-en un, par exemple : ollama pull llama3.2",
  modelThinking: "raisonnement désactivé pour répondre vite",
  modelUnavailable: "Les réglages du modèle seront disponibles une fois connecté au serveur.",
  provider: "fournisseur",
  providerUrl: "adresse de l'API",
  apiKey: "clé d'API",
  keyStored: "clé enregistrée {hint} · laisser vide pour la garder",
  keyLocation: "La clé est gardée sur votre ordinateur (data/llm.json), jamais dans cette page ni dans l'historique.",
  forgetKey: "oublier la clé",
  listModels: "lister ses modèles",
  listing: "recherche…",
  modelName: "modèle",
  modelPlaceholder: "nom du modèle",
  useModel: "utiliser ce modèle",
  cloudWarning:
    "Avec un fournisseur externe, le texte de vos échanges lui est envoyé : transcriptions, réponses et indice d'émotion estimée. L'audio et l'historique restent sur votre ordinateur. L'usage peut être payant selon le fournisseur ; certains proposent un palier gratuit.",
  cloudDisabled: "Les fournisseurs externes sont désactivés sur ce serveur (ALLOW_CLOUD_LLM=false).",
  modelChosen: "Le prochain message sera traité par {model}.",
  currentModel: "en ce moment",
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
  weekdays: ["lun.", "mar.", "mer.", "jeu.", "ven.", "sam.", "dim."],
  strataTitle: "strates",
  strataSubtitle: "7 derniers jours · une bande par échange",
  strataHow: "épaisseur = confiance de l'estimation · le plus récent en haut",
  strataEmpty: "Aucune strate pour l'instant : chaque échange en déposera une.",
  strataTyped: "message écrit · sans indice vocal",
  strataHint: "Survolez une bande, ou parcourez-les avec les flèches.",
  clearHistory: "effacer mon historique",
  clearConfirm: "Effacer définitivement {n} échange(s) ? L'audio n'est jamais conservé ; ceci efface les transcriptions et les estimations.",
  clearYes: "effacer",
  clearNo: "annuler",
  cleared: "Historique effacé.",
  trendUp: "en hausse sur 7 jours",
  trendDown: "en baisse sur 7 jours",
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
