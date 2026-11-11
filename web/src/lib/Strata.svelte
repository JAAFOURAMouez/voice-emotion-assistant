<script lang="ts">
  import type { HistoryItem, TrendItem } from "./api/history";
  import type { EmotionLabel } from "./api/types";
  import { LABEL_WORDS, LABELS } from "./emotions";
  import { edgeClip, layoutStrata, type Band } from "./strata";
  import { decimal, T, typeset } from "./texts";

  interface Props {
    items: HistoryItem[];
    trend: TrendItem[];
    simulated: boolean;
    notice: string | null;
    onclear: () => Promise<void>;
  }
  let { items, trend, simulated, notice, onclear }: Props = $props();

  let coreHeight = $state(520);
  let active = $state<number | null>(null); // interaction id
  let activeTop = $state(0);
  let confirming = $state(false);
  let clearing = $state(false);
  let seen = new Set<number>();
  let fresh = $state(new Set<number>());

  const layout = $derived(layoutStrata(items, new Date(), { maxHeight: coreHeight }));
  const bands: Band[] = $derived(layout.layers.flatMap((l) => l.bands));
  const activeBand = $derived(bands.find((b) => b.item.id === active) ?? null);
  const focusId = $derived(active ?? bands[0]?.item.id ?? null);
  const realCount = $derived(simulated ? 0 : items.length);

  // Only bands that appear while the page is open are "deposited" (animated).
  $effect(() => {
    const ids = items.map((i) => i.id);
    const isFirst = seen.size === 0;
    const added = ids.filter((id) => !seen.has(id));
    seen = new Set(ids);
    fresh = isFirst || simulated ? new Set() : new Set(added);
  });

  const notable = $derived(
    simulated ? [] : trend.filter((t) => t.label !== "neutre" && (t.direction === "hausse" || t.direction === "baisse")),
  );

  function show(band: Band, element: HTMLElement): void {
    active = band.item.id;
    activeTop = element.offsetTop + element.offsetHeight / 2;
  }

  function move(event: KeyboardEvent, index: number): void {
    const keys: Record<string, number> = { ArrowDown: index + 1, ArrowUp: index - 1, Home: 0, End: bands.length - 1 };
    const next = keys[event.key];
    if (next === undefined) return;
    event.preventDefault();
    const target = Math.max(0, Math.min(bands.length - 1, next));
    const element = document.querySelector<HTMLElement>(`[data-band="${bands[target]!.item.id}"]`);
    element?.focus();
  }

  function describe(band: Band): string {
    const item = band.item;
    const when = new Date(item.timestamp);
    const time = when.toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit" });
    const cue = item.label ? `${LABEL_WORDS[item.label]}, ${T.confidence} ${decimal(item.confidence ?? 0)}` : T.strataTyped;
    return `${time} — ${cue} — « ${item.transcript} »`;
  }

  function label(item: HistoryItem): EmotionLabel | "typed" {
    return item.label ?? "typed";
  }

  async function confirmClear(): Promise<void> {
    clearing = true;
    try {
      await onclear();
    } finally {
      clearing = false;
      confirming = false;
      active = null;
    }
  }
</script>

<section class="strata" class:simulated aria-labelledby="strata-title">
  <header class="head">
    <h2 id="strata-title" class="mono">{T.strataTitle}</h2>
    <p class="mono sub">{T.strataSubtitle}</p>
  </header>

  {#if simulated && notice}
    <p class="sim">{notice}</p>
  {/if}

  <div class="core" bind:clientHeight={coreHeight} role="group" aria-label={T.strataSubtitle} aria-describedby="strata-how">
    {#each layout.layers as layer (layer.key)}
      <div class="day" role="presentation">
        <span class="mono day-title">{layer.title}</span>
      </div>
      {#each layer.bands as band (band.item.id)}
        {@const index = bands.indexOf(band)}
        <button
          type="button"
          class="band emo-{label(band.item)}"
          class:fresh={fresh.has(band.item.id)}
          class:on={active === band.item.id}
          data-band={band.item.id}
          tabindex={focusId === band.item.id ? 0 : -1}
          style:height="{band.thickness}px"
          style:margin-left="{band.insetLeft}%"
          style:margin-right="{band.insetRight}%"
          style:clip-path={edgeClip(band.item.id, band.thickness)}
          aria-label={describe(band)}
          onmouseenter={(e) => show(band, e.currentTarget)}
          onmouseleave={() => {
            if (document.activeElement?.getAttribute("data-band") !== String(band.item.id)) active = null;
          }}
          onfocus={(e) => show(band, e.currentTarget)}
          onblur={() => (active = null)}
          onkeydown={(e) => move(e, index)}
        ></button>
      {/each}
    {/each}
    {#if bands.length === 0}
      <p class="empty">{T.strataEmpty}</p>
    {/if}

    {#if activeBand}
      <div class="detail" style:--top="{activeTop}px" aria-hidden="true">
        <p class="mono when">
          {new Date(activeBand.item.timestamp).toLocaleString("fr-FR", { weekday: "short", day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" })}
        </p>
        <p class="mono cue">
          {#if activeBand.item.label}
            <span class="chip emo-{activeBand.item.label}" aria-hidden="true"></span>
            {LABEL_WORDS[activeBand.item.label]} · {T.confidence} {decimal(activeBand.item.confidence ?? 0)}
          {:else}
            {T.strataTyped}
          {/if}
        </p>
        <p class="said">{typeset(activeBand.item.transcript)}</p>
      </div>
    {/if}
  </div>

  <p id="strata-how" class="mono how">{T.strataHow}. {T.strataHint}</p>

  <ul class="legend" aria-label="Légende">
    {#each LABELS as l}
      <li class="mono"><span class="chip emo-{l}" aria-hidden="true"></span>{LABEL_WORDS[l]}</li>
    {/each}
  </ul>

  {#if notable.length}
    <p class="mono trend">
      {#each notable as t, i}{i ? " · " : ""}{LABEL_WORDS[t.label]} {t.direction === "hausse" ? T.trendUp : T.trendDown}{/each}
    </p>
  {/if}

  <div class="clear">
    {#if confirming}
      <p class="confirm">{T.clearConfirm.replace("{n}", String(realCount))}</p>
      <div class="actions">
        <button type="button" class="link danger" disabled={clearing} onclick={confirmClear}>{T.clearYes}</button>
        <button type="button" class="link" onclick={() => (confirming = false)}>{T.clearNo}</button>
      </div>
    {:else}
      <button type="button" class="link mono" disabled={realCount === 0} onclick={() => (confirming = true)}>{T.clearHistory}</button>
    {/if}
  </div>
</section>

<style>
  .strata {
    display: flex;
    flex-direction: column;
    min-height: 0;
    height: 100%;
    border-top: 1px solid var(--ink);
    padding-top: 0.6rem;
  }
  h2 {
    margin: 0;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.14em;
  }
  .sub,
  .how {
    margin: 0.25rem 0 0;
    color: var(--ink-3);
  }
  .how {
    margin-top: 0.6rem;
  }
  .sim {
    margin: 0.7rem 0 0;
    padding-left: 0.7rem;
    border-left: 2px dashed var(--ink);
    font-size: 0.9rem;
    font-style: italic;
  }
  .core {
    position: relative;
    flex: 1 1 0;
    min-height: 220px;
    margin-top: 0.9rem;
    display: flex;
    flex-direction: column;
    overflow: visible;
  }
  .day {
    height: 22px;
    display: flex;
    align-items: flex-end;
    border-bottom: 1px solid var(--rule);
    margin-bottom: 3px;
    flex: none;
  }
  .day-title {
    color: var(--ink-3);
    font-size: 0.66rem;
    padding-bottom: 2px;
  }
  .band {
    flex: none;
    display: block;
    padding: 0;
    margin-top: 0;
    margin-bottom: 1px;
    border: 0;
    border-radius: 0;
    background-color: var(--c);
    cursor: default;
    transform-origin: top;
    transition: transform 400ms cubic-bezier(0.2, 0.7, 0.2, 1);
  }
  .band:hover,
  .band.on {
    transform: scaleX(1.06);
    box-shadow: inset 0 0 0 1px var(--ink);
  }
  .band:focus-visible {
    outline: 2px solid var(--ink);
    outline-offset: 1px;
  }
  /* The uneven edges would clip the focus ring: straighten the active band. */
  .band.on,
  .band:hover,
  .band:focus-visible {
    clip-path: none !important;
  }
  .band.fresh {
    animation: deposit 1400ms cubic-bezier(0.16, 0.9, 0.24, 1.08);
  }
  @keyframes deposit {
    from {
      transform: scaleY(0);
      opacity: 0;
    }
  }

  /* One hue AND one texture per estimated emotion: never colour alone. */
  .emo-neutre { --c: var(--emo-neutre); }
  .emo-joie {
    --c: var(--emo-joie);
    background-image: radial-gradient(color-mix(in srgb, var(--paper) 70%, transparent) 0.9px, transparent 1.1px);
    background-size: 4px 4px;
  }
  .emo-tristesse {
    --c: var(--emo-tristesse);
    background-image: repeating-linear-gradient(to bottom, transparent 0 2px, color-mix(in srgb, var(--paper) 55%, transparent) 2px 3px);
  }
  .emo-stress {
    --c: var(--emo-stress);
    background-image: repeating-linear-gradient(-55deg, transparent 0 2px, color-mix(in srgb, var(--paper) 55%, transparent) 2px 3px);
  }
  .emo-typed {
    --c: var(--ink-3);
  }

  .empty {
    margin: 1rem 0 0;
    color: var(--ink-2);
    font-style: italic;
    font-size: 0.95rem;
  }
  .detail {
    position: absolute;
    top: var(--top);
    right: calc(100% + 1.2rem);
    width: min(19rem, 40vw);
    transform: translateY(-50%);
    padding: 0.6rem 0.8rem 0.7rem;
    background: var(--paper);
    border-left: 1px solid var(--ink);
    pointer-events: none;
    z-index: 5;
  }
  .detail p {
    margin: 0;
  }
  .when {
    color: var(--ink-3);
  }
  .cue {
    margin-top: 0.15rem !important;
    color: var(--ink-2);
  }
  .said {
    margin-top: 0.35rem !important;
    font-style: italic;
    font-size: 1rem;
    line-height: 1.35;
  }
  .legend {
    list-style: none;
    margin: 0.7rem 0 0;
    padding: 0;
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem 0.9rem;
  }
  .legend li {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    color: var(--ink-2);
  }
  .chip {
    display: inline-block;
    width: 1.1rem;
    height: 0.6rem;
    background-color: var(--c);
  }
  .trend {
    margin: 0.6rem 0 0;
    color: var(--ink-2);
  }
  .clear {
    margin-top: 0.9rem;
    padding-top: 0.6rem;
    border-top: 1px solid var(--rule);
  }
  .confirm {
    margin: 0 0 0.5rem;
    font-size: 0.9rem;
  }
  .actions {
    display: flex;
    gap: 1.2rem;
  }
  .link {
    background: none;
    border: 0;
    padding: 0.1rem 0;
    border-bottom: 1px solid var(--ink-3);
    color: var(--ink-2);
    cursor: pointer;
    font-size: 0.72rem;
  }
  .link:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }
  .link.danger {
    color: var(--ink);
    border-bottom: 2px solid var(--ink);
    font-size: 0.95rem;
  }
  .actions .link:not(.danger) {
    font-size: 0.95rem;
  }
  .simulated .band {
    opacity: 0.55;
  }

  @media (max-width: 860px) {
    .strata {
      height: auto;
    }
    .core {
      min-height: 0;
    }
    .detail {
      position: static;
      order: -1;
      transform: none;
      width: auto;
      margin: 0 0 0.8rem;
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .band.fresh {
      animation: none;
    }
  }
</style>
