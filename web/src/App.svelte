<script lang="ts">
  import { onMount } from "svelte";
  import { Microphone, type MicState } from "./lib/audio/microphone";
  import { SpeechPlayer } from "./lib/audio/player";
  import { applyMessage, type Turn } from "./lib/conversation";
  import Dialogue from "./lib/Dialogue.svelte";
  import TypedInput from "./lib/TypedInput.svelte";
  import { TurnSocket, type Connection } from "./lib/api/socket";
  import type { EmotionLabel, Estimate, Exercise, LlmState, ServerMessage } from "./lib/api/types";
  import { cueText } from "./lib/emotions";
  import PushToTalk from "./lib/PushToTalk.svelte";
  import { decimal, T, typeset } from "./lib/texts";
  import Voiceprint from "./lib/Voiceprint.svelte";
  import type { VoiceprintMode } from "./lib/voiceprint";

  type Phase = "idle" | "recording" | "processing" | "replying";

  let connection = $state<Connection>("connecting");
  let serverState = $state<"loading" | "ready" | "error">("loading");
  let serverMessage = $state<string | null>(null);
  let llm = $state<LlmState>({ ok: true, message: null });
  let micState = $state<MicState>("idle");
  let phase = $state<Phase>("idle");
  let notice = $state<string | null>(null);
  let estimate = $state<Estimate | null>(null);
  let turns = $state<Turn[]>([]);
  let speaking = $state(false);
  let exercise = $state<Exercise | null>(null);
  let turnId = 0;

  const mic = new Microphone((state) => (micState = state));
  const player = new SpeechPlayer((playing) => (speaking = playing));
  let socket: TurnSocket;

  const label: EmotionLabel = $derived(estimate?.label ?? "neutre");
  const confidence = $derived(estimate?.confidence ?? 0);
  const mode: VoiceprintMode = $derived(
    phase === "recording" ? "listening" : speaking ? "speaking" : phase === "processing" ? "thinking" : "idle",
  );
  // The same shape follows the user's voice, then the assistant's.
  const live = $derived(phase === "recording" ? () => mic.read() : speaking ? () => player.read() : null);

  // Highest-priority state first: what keeps the user from talking right now.
  const blocking: string | null = $derived(
    connection === "closed"
      ? T.offline
      : serverState === "loading"
        ? (serverMessage ?? T.loading)
        : serverState === "error"
          ? serverMessage
          : micState === "denied"
            ? T.micDenied
            : micState === "unavailable"
              ? T.micUnavailable
              : null,
  );
  const canTalk = $derived(connection === "open" && serverState === "ready" && micState !== "denied" && micState !== "unavailable");

  $effect(() => {
    document.documentElement.dataset.emotion = label;
  });

  function readingTag(e: Estimate): string {
    const text = cueText(e, decimal, T.nothingMarked);
    return e.label === "neutre" ? text : text.replace(" · ", ` · ${T.confidence} `);
  }

  function handle(message: ServerMessage): void {
    turns = applyMessage(turns, message, () => ++turnId);
    switch (message.type) {
      case "status":
        serverState = message.state;
        serverMessage = message.message;
        llm = message.llm;
        break;
      case "silence":
        notice = T.silence;
        phase = "idle";
        break;
      case "emotion":
        estimate = message;
        break;
      case "reply_start":
        phase = "replying";
        break;
      case "exercise":
        exercise = message;
        break;
      case "not_understood":
        notice = message.message;
        break;
      case "error":
        notice = message.message;
        break;
      case "done":
        phase = "idle";
        break;
    }
  }

  async function startTalking(): Promise<void> {
    if (!canTalk || phase === "recording") return;
    player.unlock(); // a user gesture: audio playback is allowed from now on
    player.stop(); // speaking over the assistant interrupts it
    notice = null;
    phase = "recording";
    socket.send({ type: "start", sample_rate: 16000 });
    const ok = await mic.start((pcm) => socket.sendAudio(pcm));
    if (!ok) {
      socket.send({ type: "cancel" });
      phase = "idle";
    }
  }

  async function stopTalking(): Promise<void> {
    if (phase !== "recording") return;
    await mic.stop();
    phase = "processing";
    socket.send({ type: "end" });
  }

  function sendText(text: string): void {
    player.unlock();
    player.stop();
    notice = null;
    phase = "processing";
    socket.send({ type: "text", text });
  }

  function answerOffer(accept: boolean): void {
    player.unlock();
    turns = turns.map((t, i) => (i === turns.length - 1 ? { ...t, offer: null } : t));
    phase = "processing";
    socket.send({ type: "offer", accept });
  }

  function newConversation(): void {
    player.stop();
    turns = [];
    estimate = null;
    socket.send({ type: "reset" });
  }

  onMount(() => {
    socket = new TurnSocket(handle, (wav) => player.enqueue(wav), (state) => (connection = state));
    socket.connect();
    return () => socket.close();
  });
</script>

<div class="page" data-phase={phase}>
  <header class="masthead">
    <p class="wordmark"><span>{T.wordmark}</span> <em>{T.tagline}</em></p>
    <p class="mono local">● {T.local}</p>
    {#if turns.length}
      <button type="button" class="mono reset" onclick={newConversation}>{T.newConversation}</button>
    {/if}
  </header>

  <section class="stage" aria-label="Empreinte vocale">
    <div class="shape">
      <Voiceprint {mode} {label} {confidence} {live} />
    </div>
    <p class="reading" aria-live="polite">
      {#if estimate}
        <span class="mono tag">{readingTag(estimate)}</span>
        <span class="phrase">{typeset(estimate.phrase)}</span>
      {:else}
        <span class="mono tag">{T.estimate} · —</span>
        <span class="phrase">{T.idleNote}</span>
      {/if}
    </p>
  </section>

  <section class="text" aria-label="Conversation">
    <Dialogue {turns} thinking={phase === "processing"} onoffer={answerOffer} />
  </section>

  <aside class="notes" aria-live="assertive">
    {#if blocking}
      <p class="note">{blocking}</p>
    {:else if !llm.ok && llm.message}
      <p class="note">{llm.message}</p>
    {/if}
    {#if notice}
      <p class="note soft">{notice}</p>
    {/if}
  </aside>

  <footer class="controls">
    <PushToTalk recording={phase === "recording"} disabled={!canTalk || phase === "processing" || phase === "replying"} onstart={startTalking} onstop={stopTalking} />
    <TypedInput disabled={connection !== "open" || serverState !== "ready" || phase === "processing" || phase === "replying"} onsend={sendText} />
  </footer>
</div>

<style>
  .page {
    min-height: 100vh;
    display: grid;
    grid-template-columns: minmax(16rem, 30rem) 1fr;
    grid-template-rows: auto minmax(0, 1fr) auto;
    height: 100vh;
    grid-template-areas:
      "mast stage"
      "text stage"
      "ctrl stage";
    column-gap: clamp(24px, 5vw, 96px);
    padding: var(--gutter);
  }
  .masthead {
    grid-area: mast;
    border-top: 1px solid var(--ink);
    padding-top: 0.6rem;
  }
  .wordmark {
    margin: 0;
    font-size: 1.6rem;
    font-weight: 600;
    font-variation-settings: "opsz" 72, "SOFT" 100, "WONK" 1;
    letter-spacing: -0.01em;
  }
  .wordmark em {
    font-weight: 300;
    font-style: italic;
    color: var(--ink-2);
  }
  .local {
    margin: 0.35rem 0 0;
    color: var(--ink-2);
  }
  .stage {
    grid-area: stage;
    position: relative;
    display: grid;
    grid-template-rows: 1fr auto;
    min-height: 70vh;
  }
  .shape {
    position: relative;
    min-height: 320px;
  }
  .reading {
    margin: 0 0 0 12%;
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
    max-width: var(--measure);
    border-left: 1px solid var(--rule);
    padding-left: 0.9rem;
  }
  .tag {
    color: var(--ink-2);
    text-transform: lowercase;
  }
  .phrase {
    font-style: italic;
    font-size: 1.15rem;
    font-variation-settings: "opsz" 36;
  }
  .text {
    grid-area: text;
    min-height: 0;
    max-width: var(--measure);
    padding: 1.5rem 0 1.2rem;
  }
  .reset {
    margin-top: 0.6rem;
    background: none;
    border: 0;
    padding: 0;
    color: var(--ink-2);
    border-bottom: 1px solid var(--rule);
    cursor: pointer;
  }
  .notes {
    grid-area: text;
    align-self: start;
    margin-top: 6rem;
    max-width: var(--measure);
  }
  .note {
    margin: 0 0 0.6rem;
    padding-left: 0.9rem;
    border-left: 2px solid var(--ink);
    font-size: 1rem;
  }
  .note.soft {
    border-left-color: var(--rule);
    color: var(--ink-2);
  }
  .controls {
    grid-area: ctrl;
    padding-top: 1rem;
    border-top: 1px solid var(--rule);
  }

  @media (max-width: 860px) {
    .page {
      grid-template-columns: 1fr;
      grid-template-rows: auto auto auto auto;
      grid-template-areas: "mast" "stage" "text" "ctrl";
      height: auto;
      padding-bottom: 14rem;
    }
    .stage {
      min-height: 48vh;
    }
    .shape {
      min-height: 38vh;
    }
    .reading {
      margin-left: 0;
    }
    .notes {
      margin-top: 0;
      grid-area: text;
    }
    .text {
      padding: 1rem 0;
    }
    .controls {
      position: fixed;
      left: 0;
      right: 0;
      bottom: 0;
      padding: 0.9rem var(--gutter) 1rem;
      background: var(--paper);
    }
  }
</style>
