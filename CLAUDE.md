# CLAUDE.md

Instructions pour Claude Code sur ce dépôt.

## Le projet

Assistant vocal qui analyse **ce que dit** l'utilisateur (transcription) et **comment il le dit** (caractéristiques de la voix), puis adapte ses réponses. Quatre fonctionnalités :

1. Détecter des indices vocaux associés au stress, à la joie ou à la tristesse.
2. Adapter le ton et le style des réponses du LLM.
3. Proposer des exercices de respiration ou de relaxation quand c'est pertinent.
4. Suivre l'évolution des émotions estimées au fil des interactions.

## Contraintes non négociables

- **Tout est gratuit et local.** Aucune API payante, aucune clé API, aucun envoi d'audio ou de texte vers un service externe. Le LLM passe par Ollama (`http://localhost:11434`).
- **Vocabulaire prudent.** On parle toujours d'émotion *estimée* ou d'*indices vocaux*, jamais de diagnostic. L'assistant n'est pas un outil médical ni psychologique ; ne jamais lui faire affirmer « vous êtes stressé », plutôt « votre voix semble plus tendue ».
- **Vie privée.** L'audio brut n'est pas conservé par défaut ; seuls la transcription, les scores d'émotion et l'horodatage sont stockés (SQLite local). La conservation de l'audio est une option désactivée par défaut.
- **Sécurité de l'utilisateur.** Si la transcription évoque une détresse grave, l'assistant ne propose pas un simple exercice de respiration : il encourage à contacter un proche ou un professionnel (en France : 3114, 15 ou 112 en cas d'urgence). Cette règle vit dans le prompt système et dans un test.
- **Langue.** Interface et réponses en français par défaut, langue configurable.

## Stack

| Rôle | Outil | Remarque |
|---|---|---|
| LLM | Ollama (`llama3.2` par défaut) | modèle configurable via `OLLAMA_MODEL` |
| Transcription | `faster-whisper` (modèle `small`) | `compute_type="int8"` sur CPU |
| Détection de parole | `silero-vad` | découpe les tours de parole |
| Prosodie | `librosa` | pitch (pyin), énergie RMS, débit, pauses, jitter approximé |
| Émotion (modèle) | `transformers` + `superb/wav2vec2-base-superb-er` | 4 classes : neutre, joie, colère, tristesse |
| Synthèse vocale | `piper-tts` | voix `fr_FR-siwis-medium` |
| Serveur | `fastapi` + `uvicorn` | WebSocket pour un tour, REST pour l'historique et les exercices |
| Interface | Vite + Svelte 5 + TypeScript (`web/`) | sans bibliothèque de composants ni framework CSS ; polices locales |
| Stockage | `sqlite3` (bibliothèque standard) | |

Avant d'installer ou de figer une version, vérifie la version actuelle de chaque paquet et l'existence des modèles cités : ces noms ont été choisis de mémoire, sans installation de test. Si un modèle n'existe plus, propose un équivalent gratuit et note le changement dans le README.

Le modèle d'émotion est entraîné sur de l'anglais (IEMOCAP) : ses résultats sur du français sont approximatifs. C'est pourquoi le score final fusionne le modèle **et** la prosodie, et que le « stress » est une estimation dérivée (voir plus bas), pas une classe du modèle.

## Architecture cible

```
src/voxemo/
├── config.py            # Settings pydantic, lus depuis .env
├── audio/
│   ├── recorder.py      # WAV, capture micro des outils CLI (sounddevice, optionnel)
│   └── vad.py           # découpage en tours de parole
├── stt/transcriber.py   # faster-whisper → texte + langue
├── emotion/
│   ├── features.py      # caractéristiques prosodiques (librosa)
│   ├── classifier.py    # modèle wav2vec2 → probabilités par classe
│   └── fusion.py        # fusion → EmotionEstimate
├── llm/
│   ├── prompts.py       # prompt système + consignes de ton par émotion
│   └── client.py        # appel Ollama en streaming
├── wellbeing/exercises.py  # catalogue d'exercices + règle de déclenchement
├── tts/speaker.py       # Piper → audio
├── tracking/
│   ├── store.py         # SQLite : sessions, interactions
│   └── trends.py        # agrégats et tendances
├── pipeline.py          # orchestration d'un tour de conversation
└── api/                 # FastAPI : app.py (routes), session.py (WebSocket), services.py
web/                     # frontend Svelte : micro (getUserMedia + AudioWorklet 16 kHz),
                         # « empreinte vocale » (Canvas), conversation, frise des strates
tests/
```

Flux d'un tour : audio → VAD → (transcription ∥ analyse d'émotion) → fusion → prompt adapté → Ollama → TTS → enregistrement dans SQLite.

### Contrat central

```python
@dataclass(frozen=True)
class EmotionEstimate:
    label: Literal["neutre", "joie", "tristesse", "stress"]
    confidence: float          # 0–1
    scores: dict[str, float]   # score par label
    arousal: float             # 0–1, dérivé de la prosodie
    features: ProsodyFeatures  # pitch moyen/écart-type, énergie, débit, taux de pauses
```

Tous les modules en aval (prompts, exercices, suivi) ne dépendent que de cette structure.

### Règles de fusion (point de départ, à ajuster)

- `joie` et `tristesse` : probabilités du modèle, pondérées par la prosodie (tristesse : énergie et débit bas, pitch peu variable ; joie : pitch et énergie élevés et variables).
- `stress` : estimé à partir d'un arousal élevé (pitch au-dessus de la ligne de base de l'utilisateur, débit rapide, peu de pauses, jitter élevé) combiné à la probabilité de « colère » du modèle et à l'absence de joie.
- **Ligne de base par utilisateur** : les caractéristiques sont comparées à la moyenne glissante de ses interactions précédentes, pas à des seuils absolus. Sans historique suffisant (< 5 tours), baisser la confiance.
- Sous un seuil de confiance (0,5 par défaut), le label retenu est `neutre` et le ton reste standard.

### Adaptation du ton

`llm/prompts.py` contient un prompt système fixe plus un bloc injecté selon l'émotion estimée :

- **stress** : phrases courtes, rythme posé, une seule idée à la fois, proposer une pause.
- **tristesse** : ton chaleureux, validation avant toute solution, pas d'enthousiasme forcé.
- **joie** : ton dynamique, peut être plus léger.
- **neutre** : ton standard.

Le LLM reçoit l'estimation comme un indice (« la voix suggère… confiance 0,7 »), avec la consigne de ne pas la citer comme un fait et de ne pas la commenter à chaque tour.

### Exercices

Catalogue statique dans `wellbeing/exercises.py` (cohérence cardiaque 5-5, respiration 4-7-8, respiration carrée, ancrage 5-4-3-2-1, relaxation musculaire courte). Règle de déclenchement : stress ou tristesse estimés avec confiance suffisante sur **deux tours consécutifs**, pas plus d'une proposition toutes les N minutes, et toujours sous forme de proposition que l'utilisateur peut refuser. L'interface guide l'exercice avec un minuteur visuel.

### Suivi

Table `interactions(id, session_id, timestamp, transcript, reply, label, confidence, scores_json, features_json)`. L'interface affiche une courbe par émotion dans le temps, la répartition par session et la tendance sur 7 jours. Prévoir un bouton « effacer mon historique ».

## Ordre de réalisation

Avance par étapes, chacune testée avant la suivante :

1. Squelette du paquet, `config.py`, `.env.example`, `pyproject.toml`, `.gitignore`.
2. Boucle texte seule : `llm/client.py` + prompts, testable sans micro.
3. Audio : enregistrement, VAD, transcription.
4. Émotion : `features.py`, puis `classifier.py`, puis `fusion.py`.
5. `pipeline.py` reliant le tout, en ligne de commande.
6. Stockage et tendances.
7. Exercices.
8. Synthèse vocale.
9. Interface web : API FastAPI + WebSocket, capture micro et forme réactive, tour complet,
   frise des strates, exercice guidé, Docker (`docker compose up`). Le micro est capté dans
   le navigateur ; après chaque sous-étape, vérifier dans un navigateur (Playwright) et
   montrer une capture.

## Commandes

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.2
pytest                       # tests
ruff check . && ruff format . # lint et format
python -m voxemo.api         # interface sur http://localhost:8000 (après npm run build)
cd web && npm install && npm run build   # ou npm run dev (proxy vers l'API)
cd web && npm test && npm run check      # tests et types du frontend
```

## Conventions de code

- Python 3.11, annotations de type partout, `dataclass` ou modèles pydantic pour les données échangées entre modules.
- Code, noms de variables et docstrings en anglais ; textes visibles par l'utilisateur en français, regroupés dans un module dédié.
- Chaque composant lourd (Whisper, wav2vec2, Ollama, Piper) est derrière une petite interface (`Protocol`) pour pouvoir le simuler dans les tests et le remplacer.
- Les modèles sont chargés une seule fois (chargement paresseux, mis en cache), jamais à chaque tour.
- Les tests ne nécessitent ni micro, ni Ollama, ni téléchargement de modèle : utiliser des signaux synthétiques (sinusoïdes, bruit) et des doublures. Les tests d'intégration réels sont marqués `@pytest.mark.integration`.
- Pas de nouvelle dépendance sans l'ajouter à `requirements.txt` et vérifier qu'elle est gratuite et utilisable hors ligne.
- Viser une latence de bout en bout inférieure à ~5 s sur CPU : streaming de la réponse Ollama, transcription et analyse d'émotion en parallèle.
