<script lang="ts">
  import { onMount } from "svelte";

  // Reveals streamed text at a steady, slightly accelerating pace (no bursts).
  let { text, instant = false }: { text: string; instant?: boolean } = $props();
  let shown = $state(0);
  let reduced = false;

  onMount(() => {
    reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
    let last = performance.now();
    let frame = 0;
    const tick = (now: number) => {
      const dt = (now - last) / 1000;
      last = now;
      const backlog = text.length - shown;
      if (instant || reduced) shown = text.length;
      else if (backlog > 0) shown = Math.min(text.length, shown + Math.max(1, dt * Math.max(38, backlog * 2.6)));
      else if (backlog < 0) shown = text.length;
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  });
</script>

<span>{text.slice(0, Math.floor(shown))}</span><span class="rest" aria-hidden="true">{text.slice(Math.floor(shown))}</span>

<style>
  /* The unrevealed text keeps its place, so lines never reflow while writing. */
  .rest {
    color: transparent;
  }
</style>
