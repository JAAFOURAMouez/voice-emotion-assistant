<script lang="ts">
  import { T } from "./texts";

  let { disabled, onsend }: { disabled: boolean; onsend: (text: string) => void } = $props();
  let value = $state("");
</script>

<form
  class="typed"
  onsubmit={(event) => {
    event.preventDefault();
    const text = value.trim();
    if (!text || disabled) return;
    onsend(text);
    value = "";
  }}
>
  <label class="mono" for="typed-input">{T.typeLabel}</label>
  <input id="typed-input" type="text" bind:value autocomplete="off" placeholder={T.typePlaceholder} {disabled} />
</form>

<style>
  .typed {
    display: flex;
    align-items: baseline;
    gap: 0.75rem;
    margin-top: 0.9rem;
  }
  label {
    color: var(--ink-3);
    white-space: nowrap;
  }
  input {
    flex: 1;
    min-width: 0;
    background: transparent;
    border: 0;
    border-bottom: 1px solid var(--rule);
    padding: 0.3rem 0;
    font: inherit;
    font-size: 1rem;
    color: var(--ink);
  }
  input::placeholder {
    color: var(--ink-3);
    font-style: italic;
  }
  input:focus {
    outline: none;
    border-bottom-color: var(--ink);
  }
</style>
