<script lang="ts">
  import type { Turn } from "./conversation";
  import { cueText } from "./emotions";
  import { decimal, T, typeset } from "./texts";
  import Typewriter from "./Typewriter.svelte";

  interface Props {
    turns: Turn[];
    thinking: boolean;
    onoffer: (accept: boolean) => void;
  }
  let { turns, thinking, onoffer }: Props = $props();

  const visible = $derived(turns.slice(-6));
</script>

<div class="dialogue">
  <ol class="turns">
    {#each visible as turn, index (turn.id)}
      {@const age = visible.length - 1 - index}
      <li class="turn" style:--age={age} aria-hidden={age > 2 ? "true" : undefined}>
        {#if turn.said}
          <p class="said">
            {#if turn.typed}<span class="mono kind">{T.typed}</span>{/if}
            {typeset(turn.said)}
          </p>
          {#if turn.estimate}
            <p class="mono cue">{cueText(turn.estimate, decimal, T.nothingMarked)}</p>
          {/if}
        {/if}
        {#if turn.reply}
          <p class="answer">
            {#if age === 0 && !turn.complete}<Typewriter text={typeset(turn.reply)} />{:else}{typeset(turn.reply)}{/if}
          </p>
        {:else if age === 0 && thinking}
          <p class="answer pending">{T.thinking}</p>
        {/if}
        {#if turn.error}
          <p class="note">{turn.error}</p>
        {/if}
        {#if turn.offer && age === 0 && turn.complete}
          <div class="offer" role="group" aria-label={turn.offer.exercise.name}>
            <button type="button" class="choice" onclick={() => onoffer(true)}>{T.offerYes}</button>
            <button type="button" class="choice quiet" onclick={() => onoffer(false)}>{T.offerNo}</button>
          </div>
        {/if}
      </li>
    {/each}
  </ol>
  <!-- Screen readers get the latest reply once complete, not every character. -->
  <p class="visually-hidden" aria-live="polite">
    {#if turns.at(-1)?.complete}{turns.at(-1)?.reply}{/if}
  </p>
</div>

<style>
  .dialogue {
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    overflow: hidden;
    /* Earlier turns fade towards the top, like ink drying on the page. */
    mask-image: linear-gradient(to bottom, transparent 0, #000 38%);
    -webkit-mask-image: linear-gradient(to bottom, transparent 0, #000 38%);
  }
  .turns {
    list-style: none;
    margin: 0;
    padding: 0;
  }
  .turn {
    margin-top: 1.6rem;
    opacity: calc(1 - var(--age) * 0.2);
    transform: translateY(calc(var(--age) * -2px));
    transition: opacity 900ms cubic-bezier(0.2, 0.7, 0.2, 1);
  }
  .said {
    margin: 0;
    color: var(--ink-2);
    font-style: italic;
    font-variation-settings: "opsz" 18;
  }
  .kind {
    font-style: normal;
    margin-right: 0.4rem;
    color: var(--ink-3);
  }
  .cue {
    margin: 0.2rem 0 0.55rem;
    color: var(--ink-3);
  }
  .cue::before {
    content: "";
    display: inline-block;
    width: 1.4rem;
    height: 1px;
    margin-right: 0.5rem;
    vertical-align: middle;
    background: var(--accent);
  }
  .answer {
    margin: 0.35rem 0 0;
    font-size: 1.28rem;
    line-height: 1.42;
    font-variation-settings: "opsz" 32;
    hanging-punctuation: first;
  }
  .turn:not(:last-child) .answer {
    font-size: 1.05rem;
  }
  .pending {
    color: var(--ink-3);
    font-style: italic;
  }
  .note {
    margin: 0.5rem 0 0;
    padding-left: 0.8rem;
    border-left: 2px solid var(--ink);
    font-size: 0.95rem;
  }
  .offer {
    display: flex;
    gap: 1.4rem;
    margin-top: 0.9rem;
  }
  .choice {
    background: none;
    border: 0;
    border-bottom: 1px solid var(--ink);
    padding: 0.15rem 0;
    font-size: 1.05rem;
    cursor: pointer;
  }
  .choice.quiet {
    border-bottom-color: var(--ink-3);
    color: var(--ink-2);
  }
  .choice:hover {
    border-bottom-width: 2px;
  }
</style>
