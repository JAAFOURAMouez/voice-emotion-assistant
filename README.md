# voxemo — assistant vocal sensible aux émotions

Un assistant vocal qui écoute **ce que vous dites** et **comment vous le dites** (hauteur de la voix, énergie, débit, pauses), puis adapte le ton de ses réponses. Tout fonctionne en local, avec des outils gratuits et open source : aucune clé d'API, aucun envoi d'audio ou de texte vers un service externe.

L'interface web est construite autour d'une seule forme, « l'empreinte vocale » : elle se déforme pendant que vous parlez, évolue lentement vers l'émotion estimée après chaque tour, pulse quand l'assistant répond, et deviendra le guide des exercices de respiration.

## Fonctionnalités

- **Indices vocaux** associés au stress, à la joie ou à la tristesse : prosodie (pitch, énergie, débit, pauses, jitter) + modèle wav2vec2 + mots d'émotion dits explicitement (« je suis très énervé »), comparés à **votre** voix habituelle, jamais à des seuils absolus.
- **Ton adapté** : le LLM reçoit l'estimation comme un indice prudent, jamais comme un fait.
- **Exercices** de respiration et de relaxation **proposés** (jamais imposés) après deux échanges tendus ou abattus ; jamais en cas de détresse grave, où l'assistant oriente vers le 3114, le 15 ou le 112.
- **Suivi** des émotions estimées dans une base SQLite locale, effaçable à tout moment.
- **Synthèse vocale** des réponses avec Piper, phrase par phrase.

## Architecture

```
Navigateur (web/, Svelte)                         Serveur (src/voxemo/, FastAPI)
┌─────────────────────────────┐   WebSocket    ┌──────────────────────────────────────────┐
│ micro → AudioWorklet 16 kHz │ ─── PCM ─────▶ │ VAD → Whisper ∥ prosodie + wav2vec2       │
│ empreinte vocale (Canvas)   │ ◀── texte ──── │      → fusion → prompt adapté → Ollama    │
│ texte qui s'écrit           │ ◀── WAV ────── │      → Piper (phrase par phrase)          │
│ lecture de la voix          │                │      → historique SQLite                  │
└─────────────────────────────┘   REST /api    └──────────────────────────────────────────┘
```

| Rôle | Outil |
|---|---|
| LLM local | [Ollama](https://ollama.com) avec `llama3.2` (configurable) |
| Transcription | faster-whisper (`small`, int8 sur CPU) |
| Détection de parole | Silero VAD |
| Prosodie | librosa (pYIN) |
| Émotion (modèle) | `superb/wav2vec2-base-superb-er` via Transformers |
| Synthèse vocale | Piper, voix `fr_FR-siwis-medium` |
| Serveur | FastAPI + uvicorn (WebSocket + REST) |
| Interface | Vite + Svelte 5 + TypeScript, CSS écrit à la main, polices Fraunces et JetBrains Mono embarquées |
| Stockage | SQLite (bibliothèque standard) |

## Prérequis

- Python 3.11+ et Node.js 20+ (Node sert uniquement à construire l'interface)
- [Ollama](https://ollama.com/download) installé et lancé, puis `ollama pull llama3.2`
- 8 Go de RAM minimum, 16 Go conseillés ; un GPU accélère Ollama mais n'est pas nécessaire

## Installation

```bash
git clone https://github.com/JAAFOURAMouez/voice-emotion-assistant.git
cd voice-emotion-assistant

python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt    # développement : requirements-dev.txt

cd web && npm install && npm run build && cd ..
ollama pull llama3.2
```

Au premier lancement, Whisper, wav2vec2 et la voix Piper se téléchargent une seule fois. Ensuite l'application démarre **hors ligne** : quand tous les modèles sont présents, elle passe d'elle-même en mode `HF_HUB_OFFLINE`.

## Lancement

```bash
python -m voxemo.api        # puis ouvrir http://localhost:8000
```

Maintenez la **barre d'espace** (ou le bouton) pour parler, relâchez pour envoyer. Un appui bref démarre l'écoute, un second l'arrête. On peut aussi écrire un message au clavier. Le premier chargement des modèles prend quelques minutes ; la page l'indique.

En développement, avec rechargement à chaud de l'interface :

```bash
python -m voxemo.api                 # terminal 1 : API sur :8000
cd web && npm run dev                # terminal 2 : http://localhost:5173 (proxy vers l'API)
```

> **Micro et HTTPS** : les navigateurs n'autorisent le micro que sur `localhost` ou en HTTPS. Pour utiliser l'assistant depuis une autre machine, placez-le derrière un reverse proxy HTTPS (un exemple Caddy sera fourni avec le déploiement Docker) et déclarez l'origine dans `ALLOWED_ORIGINS`.

## Configuration

Les réglages se font dans un fichier `.env` (voir `.env.example`, entièrement commenté). Les principaux :

| Variable | Défaut | Rôle |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.2` | modèle de langage |
| `WHISPER_MODEL` | `small` | `base` sur une machine modeste, `medium` pour plus de précision |
| `EMOTION_MODEL_WEIGHT` | `0.35` | part du modèle wav2vec2 (le reste : prosodie) |
| `EMOTION_TEXT_GAIN` | `5.0` | poids des mots d'émotion explicites ; `0` = voix seule |
| `EMOTION_CONFIDENCE_THRESHOLD` | `0.5` | sous ce seuil, l'émotion retenue est « neutre » |
| `EXERCISE_CONSECUTIVE_TURNS` / `EXERCISE_COOLDOWN_MINUTES` | `2` / `15` | règle de proposition des exercices |
| `TTS_ENABLED` / `TTS_SPEED` | `true` / `1.0` | voix de l'assistant |
| `STORE_RAW_AUDIO` | `false` | conserver l'audio des tours (désactivé par défaut) |
| `API_HOST` / `API_PORT` | `127.0.0.1` / `8000` | le serveur n'écoute que cette machine par défaut |
| `ALLOWED_ORIGINS` | vide | origines autorisées derrière un reverse proxy |

## API

| Route | Rôle |
|---|---|
| `WS /api/turn` | un tour de conversation : le navigateur envoie `start`, l'audio PCM 16 kHz en binaire, puis `end` (ou `text`) ; le serveur renvoie `transcript`, `emotion`, `reply_chunk`…, l'audio de chaque phrase, `offer` / `exercise`, puis `done`. Détail du protocole dans `src/voxemo/api/session.py`. |
| `GET /api/status` | modèles chargés ou non, Ollama joignable, voix disponible |
| `GET /api/history` · `DELETE /api/history` | historique et tendances · effacer l'historique |
| `GET /api/history?demo=true` | une semaine **simulée**, marquée comme telle, jamais enregistrée |
| `GET /api/exercises` · `/api/exercises/{id}/speech/{n}` | catalogue · consignes lues par Piper (WAV) |

Seules les pages de l'application peuvent ouvrir le WebSocket (vérification de l'origine) ; la page `/docs` est désactivée car elle chargerait des scripts depuis un CDN.

## Outils en ligne de commande

Utiles pour tester et ajuster chaque brique sans l'interface (ils utilisent le micro via PortAudio : `sudo apt install libportaudio2`).

```bash
python -m voxemo --details            # assistant complet au micro, scores affichés
python -m voxemo --file tour.wav      # même chose sur un fichier
python -m voxemo.llm --emotion stress # conversation écrite, indice vocal simulé
python -m voxemo.stt --file tour.wav  # transcription seule
python -m voxemo.emotion --file tour.wav  # émotion estimée tour par tour
python -m voxemo.tracking [--clear]   # historique et tendances / effacer
python -m voxemo.wellbeing coherence_cardiaque  # un exercice guidé
```

`STORE_RAW_AUDIO=true python -m voxemo --details` garde l'audio des tours dans `data/audio/` (localement) pour ajuster les règles sur votre propre voix.

## Tests

```bash
pytest                         # tests Python, sans micro, sans Ollama, sans téléchargement
pytest -m integration          # avec les vrais modèles (téléchargés au préalable)
ruff check . && ruff format .
cd web && npm test && npm run check   # tests et vérification de types du frontend
```

## Structure

```
src/voxemo/
├── api/         serveur FastAPI : WebSocket, REST, sessions
├── audio/       VAD, capture micro (outils CLI), WAV
├── stt/         transcription (faster-whisper)
├── emotion/     prosodie, modèle wav2vec2, mots d'émotion, fusion
├── llm/         prompts et client Ollama
├── wellbeing/   détection de détresse, exercices et règle de proposition
├── tts/         synthèse vocale Piper
├── tracking/    historique SQLite et tendances
└── pipeline.py  orchestration d'un tour
web/
├── src/lib/audio/   micro (AudioWorklet), rééchantillonnage, lecture de la voix
├── src/lib/         empreinte vocale, conversation, composants
└── src/assets/fonts polices locales (licences OFL incluses)
tests/
```

## Avancement

| Étape | État |
|---|---|
| Pipeline (audio, transcription, émotion, LLM, historique, exercices, voix) | fait, testé |
| API FastAPI + WebSocket | fait |
| Interface : micro, empreinte vocale, conversation complète | fait |
| Frise des « strates » (historique de la semaine) | à faire |
| Exercice guidé par la forme | à faire (catalogue et règle déjà prêts côté serveur) |
| Docker (`docker compose up`) + exemple Caddy | à faire |

## Limites et éthique

- Les émotions sont des **estimations**, pas des certitudes. Le modèle d'émotion est entraîné sur de l'anglais et s'est montré biaisé (vers « joie ») et sensible au volume sur une vraie voix française : son poids est limité, les mots employés et la prosodie comptent davantage. Les mots sont repérés par une simple liste (négations comprises) : l'ironie lui échappe.
- Ce projet **n'est pas un dispositif médical** et ne remplace pas un professionnel. En cas de détresse : **3114** (prévention du suicide, France, gratuit, 24 h/24), **15** ou **112** en cas d'urgence.
- **Vie privée** : tout reste sur la machine. L'audio n'est pas conservé par défaut ; l'historique (transcriptions et scores) est dans `data/voxemo.db` et s'efface à tout moment.

## Développer avec Claude Code

`CLAUDE.md` décrit l'architecture, les contraintes et les conventions du projet.
