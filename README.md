# voxemo — assistant vocal sensible aux émotions

Un assistant vocal qui écoute **ce que vous dites** et **comment vous le dites** (hauteur de la voix, énergie, débit, pauses), puis adapte le ton de ses réponses. Par défaut, tout fonctionne en local, avec des outils gratuits et open source : aucune clé d'API, aucun envoi d'audio ou de texte vers un service externe. Si vous le souhaitez, vous pouvez choisir dans l'interface un autre modèle Ollama ou un fournisseur externe avec votre propre clé (voir [Choisir le modèle de langage](#choisir-le-modèle-de-langage)).

L'interface web est construite autour d'une seule forme, « l'empreinte vocale » : elle se déforme pendant que vous parlez, évolue lentement vers l'émotion estimée après chaque tour, pulse quand l'assistant répond, et deviendra le guide des exercices de respiration.

## Fonctionnalités

- **Indices vocaux** associés au stress, à la joie ou à la tristesse : prosodie (pitch, énergie, débit, pauses, jitter) + modèle wav2vec2 + mots d'émotion dits explicitement (« je suis très énervé »), comparés à **votre** voix habituelle, jamais à des seuils absolus.
- **Ton adapté** : le LLM reçoit l'estimation comme un indice prudent, jamais comme un fait.
- **Exercices** de respiration et de relaxation **proposés** (jamais imposés) après deux échanges tendus ou abattus ; jamais en cas de détresse grave, où l'assistant oriente vers le 3114, le 15 ou le 112.
- **Suivi** des émotions estimées dans une base SQLite locale, effaçable à tout moment.
- **Synthèse vocale** des réponses avec Piper, phrase par phrase.
- **Modèle au choix** : n'importe quel modèle installé dans Ollama, ou, sur demande explicite, un fournisseur compatible OpenAI (OpenAI, Anthropic, Mistral, Gemini, Groq, OpenRouter, ou un serveur local type LM Studio) avec votre clé.
- **Interface accessible depuis Internet** sans héberger les modèles : le site statique peut être publié gratuitement sur Render, le serveur reste sur votre ordinateur.

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

> **Micro et HTTPS** : les navigateurs n'autorisent le micro que sur `localhost` ou en HTTPS. Pour utiliser l'assistant depuis une autre machine, suivez la section suivante.

## Utiliser l'assistant depuis Internet (interface sur Render, serveur chez vous)

Les modèles (Whisper, wav2vec2, Piper, Ollama) demandent plusieurs Go de RAM : les héberger dans le cloud est payant et lent sans GPU. Ici, **seule l'interface** (HTML, JS, polices) est publiée comme site statique gratuit sur Render ; le navigateur parle directement à **votre ordinateur** à travers un tunnel HTTPS.

```
navigateur ──▶ voxemo.onrender.com (site statique, gratuit)
     └─ WebSocket + REST (HTTPS) ──tunnel──▶ votre ordinateur : python -m voxemo.api + Ollama + SQLite
```

L'historique et les modèles restent chez vous ; si l'ordinateur est éteint, la page l'indique. **Base de données : rien à prévoir.** SQLite (`data/voxemo.db`) reste sur votre ordinateur, c'est gratuit et privé. Une base dans le cloud ne servirait que si le serveur lui-même était hébergé en ligne.

**1. Sur votre ordinateur**, dans `.env` :

```bash
ACCESS_TOKEN=...            # python -c "import secrets; print(secrets.token_urlsafe(24))"
ALLOWED_ORIGINS=https://voxemo.onrender.com   # l'adresse de votre site Render
```

`ACCESS_TOKEN` est **obligatoire** : sans lui, toute personne qui trouve l'adresse du tunnel pourrait lire ou effacer votre historique. Laissez `API_HOST=127.0.0.1` : le tunnel se connecte localement.

**2. Le tunnel HTTPS** (gratuit) — au choix :

- **Tailscale Funnel** (conseillé : adresse fixe, le trafic n'est pas déchiffré par un intermédiaire). Installez [Tailscale](https://tailscale.com/download) (dans WSL : `curl -fsSL https://tailscale.com/install.sh | sh`), connectez-vous (`sudo tailscale up`), puis :
  ```bash
  sudo tailscale funnel --bg 8000     # → https://<machine>.<réseau>.ts.net
  ```
  La première fois, la commande donne un lien pour autoriser Funnel sur votre réseau Tailscale.
- **Cloudflare Tunnel** : `cloudflared tunnel --url http://localhost:8000` donne une adresse `https://….trycloudflare.com`, différente à chaque lancement (à ressaisir dans la page). Le trafic est déchiffré chez Cloudflare au passage.

**3. Le site sur Render** : poussez le dépôt sur GitHub, puis dans Render *New → Blueprint* et choisissez ce dépôt : `render.yaml` crée un *Static Site* gratuit (`web/`, `npm ci && npm run build`, `VITE_STANDALONE=true`). Facultatif : `VITE_API_URL=https://<machine>.<réseau>.ts.net` pré-remplit l'adresse. Les en-têtes de sécurité (CSP, micro autorisé uniquement pour la page) sont déclarés dans `render.yaml`.

**4. Ouvrez le site** : la page demande l'adresse du tunnel et le code d'accès (gardés dans ce navigateur seulement), puis fonctionne comme en local.

Le code d'accès voyage en `Authorization: Bearer` pour les appels REST et comme sous-protocole WebSocket (`token.<code>`) — jamais dans l'URL. Les appels REST venant d'une autre origine ne sont acceptés que depuis `ALLOWED_ORIGINS` (CORS).

## Choisir le modèle de langage

Le lien **« modèle · … »** en haut à gauche ouvre les réglages :

- **Sur cet ordinateur (Ollama)** : la liste des modèles installés (`ollama list`, modèles d'embedding exclus). Pour les modèles « à raisonnement » (qwen3, deepseek-r1…), le raisonnement est désactivé : une réponse vocale ne peut pas attendre une longue réflexion cachée. Un modèle de 7-8 B est plus juste mais plus lent et plus gourmand que `llama3.2` (3 B).
- **Fournisseur externe, avec votre clé** : OpenAI, Anthropic, Mistral, Google Gemini, Groq, OpenRouter, ou « Autre » (toute API compatible OpenAI : LM Studio, vLLM… ; `http` accepté seulement vers `localhost`). « Lister ses modèles » interroge le fournisseur.
  - La clé est stockée sur votre ordinateur dans `data/llm.json` (lisible par vous seul, non versionné), jamais renvoyée au navigateur (seuls ses 4 derniers caractères sont affichés), jamais écrite dans l'historique ; « oublier la clé » la supprime.
  - **Ce qui part chez le fournisseur** : le texte des échanges (transcriptions, réponses) et l'indice d'émotion estimée du prompt système. L'audio et l'historique ne quittent jamais la machine. L'interface l'affiche en permanence (« ○ texte envoyé à … »).
  - L'usage peut être payant selon le fournisseur ; certains proposent un palier gratuit (conditions variables, à vérifier chez eux).
  - `ALLOW_CLOUD_LLM=false` désactive totalement cette possibilité.

La règle de sécurité (détresse grave → 3114, 15, 112) ne dépend pas du modèle : si la réponse ne cite pas ces ressources, l'assistant les ajoute lui-même.

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
| `ALLOWED_ORIGINS` | vide | origines autorisées : site Render, reverse proxy (WebSocket et CORS) |
| `ACCESS_TOKEN` | vide | code d'accès exigé par l'API ; **obligatoire** dès que le serveur est joignable depuis Internet |
| `ALLOW_CLOUD_LLM` | `true` | autoriser le choix d'un fournisseur externe dans l'interface (jamais par défaut) |
| `LLM_CONFIG_PATH` | `data/llm.json` | modèle choisi dans l'interface et clés d'API |

## API

| Route | Rôle |
|---|---|
| `WS /api/turn` | un tour de conversation : le navigateur envoie `start`, l'audio PCM 16 kHz en binaire, puis `end` (ou `text`) ; le serveur renvoie `transcript`, `emotion`, `reply_chunk`…, l'audio de chaque phrase, `offer` / `exercise`, puis `done`. Détail du protocole dans `src/voxemo/api/session.py`. |
| `GET /api/status` | modèles chargés ou non, Ollama joignable, voix disponible |
| `GET /api/history` · `DELETE /api/history` | historique et tendances · effacer l'historique |
| `GET /api/history?demo=true` | une semaine **simulée**, marquée comme telle, jamais enregistrée |
| `GET /api/exercises` · `/api/exercises/{id}/speech/{n}` | catalogue · consignes lues par Piper (WAV) |
| `GET /api/llm` · `PUT /api/llm` | modèle actuel, modèles Ollama, fournisseurs (clés jamais renvoyées) · changer de modèle |
| `POST /api/llm/models` · `DELETE /api/llm/keys/{fournisseur}` | modèles proposés par un fournisseur · oublier une clé |

Seules les pages de l'application (et `ALLOWED_ORIGINS`) peuvent ouvrir le WebSocket (vérification de l'origine) ; avec `ACCESS_TOKEN`, chaque appel doit présenter le code. La page `/docs` est désactivée car elle chargerait des scripts depuis un CDN.

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
├── llm/         prompts, client Ollama, fournisseurs compatibles OpenAI, choix du modèle
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
| Frise des « strates » (historique de la semaine) | fait |
| Choix du modèle (Ollama ou fournisseur avec clé) · interface sur Render + tunnel | fait |
| Exercice guidé par la forme | à faire (catalogue et règle déjà prêts côté serveur) |
| Docker (`docker compose up`) + exemple Caddy | à faire |

## Limites et éthique

- Les émotions sont des **estimations**, pas des certitudes. Le modèle d'émotion est entraîné sur de l'anglais et s'est montré biaisé (vers « joie ») et sensible au volume sur une vraie voix française : son poids est limité, les mots employés et la prosodie comptent davantage. Les mots sont repérés par une simple liste (négations comprises) : l'ironie lui échappe.
- Ce projet **n'est pas un dispositif médical** et ne remplace pas un professionnel. En cas de détresse : **3114** (prévention du suicide, France, gratuit, 24 h/24), **15** ou **112** en cas d'urgence.
- **Vie privée** : par défaut, tout reste sur la machine. L'audio n'est pas conservé ; l'historique (transcriptions et scores) est dans `data/voxemo.db` et s'efface à tout moment. Si vous choisissez un fournisseur externe, le texte des échanges lui est envoyé (l'interface l'indique) ; avec l'interface sur Render, l'audio transite chiffré par le tunnel jusqu'à votre ordinateur.

## Développer avec Claude Code

`CLAUDE.md` décrit l'architecture, les contraintes et les conventions du projet.
