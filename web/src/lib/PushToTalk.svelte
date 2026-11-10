<script lang="ts">
  import { onMount } from "svelte";
  import { T } from "./texts";

  interface Props {
    recording: boolean;
    disabled: boolean;
    onstart: () => void;
    onstop: () => void;
  }
  let { recording, disabled, onstart, onstop }: Props = $props();

  const TAP_MS = 350; // a short press toggles instead of hold-to-talk
  let pressedAt = 0;
  let latched = $state(false);

  function press(): void {
    if (disabled) return;
    if (latched) {
      latched = false;
      onstop();
      return;
    }
    pressedAt = performance.now();
    onstart();
  }

  function release(): void {
    if (!recording || latched) return;
    if (performance.now() - pressedAt < TAP_MS) {
      latched = true; // tap: keep listening until the next press
      return;
    }
    onstop();
  }

  function typing(target: EventTarget | null): boolean {
    return target instanceof HTMLElement && (target.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName));
  }

  onMount(() => {
    const down = (event: KeyboardEvent) => {
      if (event.code !== "Space" || typing(event.target)) return;
      event.preventDefault();
      if (!event.repeat) press();
    };
    const up = (event: KeyboardEvent) => {
      if (event.code !== "Space" || typing(event.target)) return;
      event.preventDefault();
      release();
    };
    const blur = () => {
      if (recording && !latched) onstop();
    };
    window.addEventListener("keydown", down, { capture: true });
    window.addEventListener("keyup", up, { capture: true });
    window.addEventListener("blur", blur);
    return () => {
      window.removeEventListener("keydown", down, { capture: true });
      window.removeEventListener("keyup", up, { capture: true });
      window.removeEventListener("blur", blur);
    };
  });

  $effect(() => {
    if (!recording) latched = false;
  });
</script>

<div class="ptt">
  <button
    type="button"
    class:recording
    aria-pressed={recording}
    aria-describedby="ptt-hint"
    {disabled}
    onpointerdown={(event) => {
      if (event.button !== 0) return;
      event.currentTarget.setPointerCapture(event.pointerId);
      press();
    }}
    onpointerup={release}
    onpointercancel={release}
    onkeydown={(event) => {
      if (event.key === "Enter" && !event.repeat) {
        event.preventDefault();
        press();
      }
    }}
    onkeyup={(event) => {
      if (event.key === "Enter") release();
    }}
    oncontextmenu={(event) => event.preventDefault()}
  >
    <span class="dot" aria-hidden="true"></span>
    {recording ? (latched ? T.talkRecording : T.talkStop) : T.talk}
  </button>
  <p id="ptt-hint" class="mono hint">{T.talkHint}</p>
</div>

<style>
  .ptt {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    align-items: flex-start;
  }
  button {
    display: inline-flex;
    align-items: center;
    gap: 0.75rem;
    min-height: 52px;
    padding: 0.7rem 1.4rem 0.75rem 1.1rem;
    background: transparent;
    border: 1px solid var(--ink);
    border-radius: 0;
    font-family: var(--serif);
    font-size: 1.05rem;
    font-variation-settings: "opsz" 24;
    cursor: pointer;
    touch-action: none;
    user-select: none;
    -webkit-user-select: none;
    -webkit-touch-callout: none;
    transition: background-color 240ms ease, color 240ms ease;
  }
  button:hover:not(:disabled) {
    background: var(--paper-deep);
  }
  button.recording {
    background: var(--ink);
    color: var(--paper);
  }
  button:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }
  .dot {
    width: 0.6rem;
    height: 0.6rem;
    border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 0 1px var(--ink);
  }
  .recording .dot {
    background: var(--paper);
    animation: beat 1.1s ease-in-out infinite;
  }
  @keyframes beat {
    50% {
      transform: scale(1.35);
    }
  }
  .hint {
    margin: 0;
    color: var(--ink-3);
    max-width: 28rem;
  }
</style>
