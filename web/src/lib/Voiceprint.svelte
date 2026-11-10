<script lang="ts">
  import { onMount } from "svelte";
  import type { EmotionLabel } from "./api/types";
  import { VoiceprintEngine, type VoiceprintMode } from "./voiceprint";

  interface Props {
    mode: VoiceprintMode;
    label: EmotionLabel;
    confidence: number;
    live?: (() => { level: number; pitch: number }) | null;
    breath?: number;
  }
  let { mode, label, confidence, live = null, breath = 0 }: Props = $props();

  let canvas: HTMLCanvasElement;
  let engine: VoiceprintEngine | null = $state(null);

  function readColors(): void {
    if (!engine) return;
    const style = getComputedStyle(document.documentElement);
    engine.setColors(
      style.getPropertyValue("--accent").trim(),
      style.getPropertyValue("--ink").trim(),
      style.getPropertyValue("--paper").trim(),
    );
  }

  onMount(() => {
    engine = new VoiceprintEngine(canvas);
    const motion = matchMedia("(prefers-reduced-motion: reduce)");
    const dark = matchMedia("(prefers-color-scheme: dark)");
    const update = () => {
      if (engine) engine.reducedMotion = motion.matches;
      readColors();
    };
    update();
    motion.addEventListener("change", update);
    dark.addEventListener("change", update);
    engine.start();
    return () => {
      motion.removeEventListener("change", update);
      dark.removeEventListener("change", update);
      engine?.stop();
    };
  });

  $effect(() => {
    if (!engine) return;
    engine.mode = mode;
    engine.liveSource = live;
  });
  $effect(() => {
    if (!engine) return;
    engine.setMood(label, confidence);
    // The accent follows the emotion (set on <html> by the page): re-read it.
    requestAnimationFrame(readColors);
  });
  $effect(() => {
    engine?.setBreath(breath);
  });
</script>

<canvas bind:this={canvas} class="voiceprint" aria-hidden="true"></canvas>

<style>
  .voiceprint {
    display: block;
    width: 100%;
    height: 100%;
  }
</style>
