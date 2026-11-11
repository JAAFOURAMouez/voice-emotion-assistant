<script lang="ts">
  import { chooseLlm, fetchLlm, forgetKey, listRemoteModels, type CurrentModel, type LlmSettings } from "./api/llm";
  import { AccessDenied, accessCode, normalizeAddress, serverAddress, standalone } from "./api/server";
  import { decimal, T } from "./texts";

  interface Props {
    open: boolean;
    reason: string | null;
    connected: boolean;
    onserver: (address: string, code: string) => void;
    onmodel: (current: CurrentModel) => void;
  }
  let { open = $bindable(), reason, connected, onserver, onmodel }: Props = $props();

  let dialog: HTMLDialogElement;
  let address = $state(serverAddress());
  let code = $state(accessCode());
  let serverError = $state<string | null>(null);

  let settings = $state<LlmSettings | null>(null);
  let source = $state<"local" | "cloud">("local");
  let localModel = $state("");
  let providerId = $state("openai");
  let customUrl = $state("");
  let apiKey = $state("");
  let remoteModel = $state("");
  let remoteList = $state<string[]>([]);
  let busy = $state(false);
  let listing = $state(false);
  let message = $state<string | null>(null);
  let error = $state<string | null>(null);

  const provider = $derived(settings?.providers.find((p) => p.id === providerId) ?? null);
  const cloudProviders = $derived(settings?.providers ?? []);

  function explain(e: unknown): string {
    return e instanceof AccessDenied ? T.accessDenied : e instanceof Error ? e.message : String(e);
  }

  function adopt(next: LlmSettings): void {
    settings = next;
    const { current } = next;
    source = current.local ? "local" : "cloud";
    if (current.local) localModel = current.model;
    else {
      providerId = current.provider;
      remoteModel = current.model;
    }
    if (!localModel) localModel = next.ollama_models[0]?.name ?? "";
    customUrl = next.providers.find((p) => p.id === "custom")?.base_url ?? "";
  }

  async function load(): Promise<void> {
    if (!connected) {
      settings = null;
      return;
    }
    try {
      adopt(await fetchLlm());
      error = null;
    } catch (e) {
      settings = null;
      error = explain(e);
    }
  }

  $effect(() => {
    if (!dialog) return;
    if (open && !dialog.open) {
      address = serverAddress();
      code = accessCode();
      message = null;
      dialog.showModal();
    } else if (!open && dialog.open) dialog.close();
  });

  $effect(() => {
    if (open && connected) void load();
  });

  function saveServer(event: SubmitEvent): void {
    event.preventDefault();
    try {
      const normalized = normalizeAddress(address);
      serverError = null;
      onserver(normalized, code);
    } catch {
      serverError = T.badAddress;
    }
  }

  async function listModels(): Promise<void> {
    listing = true;
    error = null;
    try {
      remoteList = await listRemoteModels({
        provider: providerId,
        api_key: apiKey || undefined,
        base_url: providerId === "custom" ? customUrl : undefined,
      });
      if (!remoteModel) remoteModel = remoteList[0] ?? "";
    } catch (e) {
      error = explain(e);
    } finally {
      listing = false;
    }
  }

  async function useModel(event: SubmitEvent): Promise<void> {
    event.preventDefault();
    busy = true;
    error = null;
    message = null;
    try {
      const local = source === "local";
      const next = await chooseLlm({
        provider: local ? "ollama" : providerId,
        model: local ? localModel : remoteModel.trim(),
        api_key: !local && apiKey ? apiKey : undefined,
        base_url: !local && providerId === "custom" ? customUrl : undefined,
      });
      adopt(next);
      apiKey = "";
      onmodel(next.current);
      message = T.modelChosen.replace("{model}", next.current.model);
    } catch (e) {
      error = explain(e);
    } finally {
      busy = false;
    }
  }

  async function dropKey(): Promise<void> {
    try {
      const next = await forgetKey(providerId);
      adopt(next);
      onmodel(next.current);
    } catch (e) {
      error = explain(e);
    }
  }
</script>

<dialog bind:this={dialog} onclose={() => (open = false)} aria-labelledby="settings-title">
  <header>
    <h2 id="settings-title">{T.settings}</h2>
    <button type="button" class="link mono" onclick={() => (open = false)}>{T.settingsClose}</button>
  </header>

  {#if reason}
    <p class="note" role="alert">{reason}</p>
  {/if}

  {#if standalone || serverAddress() || reason}
  <section aria-labelledby="server-title">
    <h3 id="server-title" class="mono">{T.serverTitle}</h3>
    <form onsubmit={saveServer}>
      <p class="help">{T.serverHelp}</p>
      <label class="field">
        <span class="mono">{T.serverAddress}</span>
        <input type="url" bind:value={address} placeholder={standalone ? "https://…" : location.origin} autocomplete="url" spellcheck="false" />
      </label>
      <label class="field">
        <span class="mono">{T.accessCode}</span>
        <input type="password" bind:value={code} autocomplete="current-password" spellcheck="false" />
      </label>
      <p class="help">{T.accessCodeHelp}</p>
      {#if serverError}<p class="error" role="alert">{serverError}</p>{/if}
      <button type="submit" class="act">{T.connect}</button>
    </form>
  </section>
  {/if}

  <section aria-labelledby="model-title">
    <h3 id="model-title" class="mono">{T.modelTitle}</h3>
    {#if !settings}
      <p class="help">{error ?? T.modelUnavailable}</p>
    {:else}
      <p class="current">
        <span class="mono">{T.currentModel}</span>
        {settings.current.provider_name} · <em>{settings.current.model}</em>
      </p>
      <form onsubmit={useModel}>
        <fieldset class="sources">
          <legend class="visually-hidden">{T.modelTitle}</legend>
          <label class="choice"><input type="radio" bind:group={source} value="local" /> {T.modelLocal}</label>
          <label class="choice" class:off={!settings.cloud_allowed}>
            <input type="radio" bind:group={source} value="cloud" disabled={!settings.cloud_allowed} />
            {T.modelCloud}
          </label>
        </fieldset>
        {#if !settings.cloud_allowed}<p class="help">{T.cloudDisabled}</p>{/if}

        {#if source === "local"}
          {#if settings.ollama_models.length}
            <fieldset class="models">
              <legend class="visually-hidden">{T.modelLocal}</legend>
              {#each settings.ollama_models as m (m.name)}
                <label class="model">
                  <input type="radio" bind:group={localModel} value={m.name} />
                  <span class="name">{m.name}</span>
                  <span class="mono meta">
                    {m.parameters ? `${m.parameters} · ` : ""}{decimal(m.size_gb, 1)} Go{m.thinking ? ` · ${T.modelThinking}` : ""}
                  </span>
                </label>
              {/each}
            </fieldset>
          {:else}
            <p class="help">{T.modelNone}</p>
          {/if}
        {:else}
          <label class="field">
            <span class="mono">{T.provider}</span>
            <select bind:value={providerId} onchange={() => ((remoteList = []), (remoteModel = ""))}>
              {#each cloudProviders as p (p.id)}
                <option value={p.id}>{p.name}</option>
              {/each}
            </select>
          </label>
          {#if providerId === "custom"}
            <label class="field">
              <span class="mono">{T.providerUrl}</span>
              <input type="url" bind:value={customUrl} placeholder="https://…/v1" spellcheck="false" />
            </label>
          {/if}
          <label class="field">
            <span class="mono">{T.apiKey}</span>
            <input
              type="password"
              bind:value={apiKey}
              autocomplete="off"
              spellcheck="false"
              placeholder={provider?.has_key ? T.keyStored.replace("{hint}", provider.key_hint ?? "") : ""}
            />
          </label>
          <p class="help">
            {T.keyLocation}
            {#if provider?.has_key}
              <button type="button" class="link mono" onclick={dropKey}>{T.forgetKey}</button>
            {/if}
          </p>
          <div class="field row">
            <label class="grow">
              <span class="mono">{T.modelName}</span>
              <input list="remote-models" bind:value={remoteModel} placeholder={T.modelPlaceholder} spellcheck="false" autocomplete="off" />
            </label>
            <button type="button" class="link mono" disabled={listing} onclick={listModels}>{listing ? T.listing : T.listModels}</button>
          </div>
          <datalist id="remote-models">
            {#each remoteList as id (id)}<option value={id}></option>{/each}
          </datalist>
          <p class="warning">{T.cloudWarning}</p>
        {/if}

        <div aria-live="polite">
          {#if error}<p class="error">{error}</p>{/if}
          {#if message}<p class="done">{message}</p>{/if}
        </div>
        <button
          type="submit"
          class="act"
          disabled={busy || (source === "local" ? !localModel : !remoteModel.trim())}>{T.useModel}</button
        >
      </form>
    {/if}
  </section>
</dialog>

<style>
  dialog {
    margin: 0 0 0 auto;
    width: min(34rem, 100vw);
    max-width: 100vw;
    height: 100vh;
    max-height: 100vh;
    padding: var(--gutter);
    padding-top: 1.4rem;
    border: 0;
    border-left: 1px solid var(--ink);
    background: var(--paper);
    color: var(--ink);
    overflow-y: auto;
  }
  dialog[open] {
    animation: slide 420ms cubic-bezier(0.2, 0.7, 0.2, 1);
  }
  dialog::backdrop {
    background: color-mix(in srgb, var(--paper) 55%, transparent);
  }
  @keyframes slide {
    from {
      transform: translateX(2.5rem);
      opacity: 0;
    }
  }
  header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    border-top: 1px solid var(--ink);
    padding-top: 0.6rem;
  }
  h2 {
    margin: 0;
    font-size: 1.6rem;
    font-weight: 400;
    font-style: italic;
    font-variation-settings: "opsz" 72, "SOFT" 100;
  }
  section {
    margin-top: 1.8rem;
    padding-top: 0.6rem;
    border-top: 1px solid var(--rule);
  }
  h3 {
    margin: 0 0 0.6rem;
    font-weight: 500;
    color: var(--ink-2);
  }
  .help {
    margin: 0.3rem 0 0.8rem;
    font-size: 0.9rem;
    color: var(--ink-2);
  }
  .field {
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
    margin-bottom: 0.9rem;
  }
  .field span,
  .grow span {
    color: var(--ink-3);
  }
  .row {
    flex-direction: row;
    align-items: flex-end;
    gap: 1rem;
  }
  .grow {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
  }
  input[type="url"],
  input[type="password"],
  input:not([type]),
  select {
    width: 100%;
    background: transparent;
    border: 0;
    border-bottom: 1px solid var(--rule);
    border-radius: 0;
    padding: 0.35rem 0;
    font: inherit;
    font-size: 1rem;
    color: var(--ink);
  }
  select option {
    background: var(--paper);
  }
  input:focus-visible,
  select:focus-visible {
    outline: none;
    border-bottom: 2px solid var(--ink);
  }
  input::placeholder {
    color: var(--ink-3);
    font-style: italic;
  }
  fieldset {
    border: 0;
    margin: 0 0 0.8rem;
    padding: 0;
  }
  .choice {
    display: block;
    padding: 0.25rem 0;
    cursor: pointer;
  }
  .choice.off {
    color: var(--ink-3);
    cursor: not-allowed;
  }
  .models {
    border-left: 1px solid var(--rule);
    padding-left: 0.9rem;
  }
  .model {
    display: grid;
    grid-template-columns: auto 1fr;
    column-gap: 0.6rem;
    padding: 0.3rem 0;
    cursor: pointer;
  }
  .model .meta {
    grid-column: 2;
    color: var(--ink-3);
  }
  input[type="radio"] {
    accent-color: var(--ink);
  }
  .current {
    margin: 0 0 0.9rem;
  }
  .current .mono {
    color: var(--ink-3);
    margin-right: 0.5rem;
  }
  .warning,
  .note {
    margin: 0.4rem 0 1rem;
    padding-left: 0.9rem;
    border-left: 2px solid var(--ink);
    font-size: 0.95rem;
  }
  .note {
    margin-top: 1.2rem;
  }
  .error {
    margin: 0.3rem 0 0.8rem;
    color: var(--emo-stress);
    font-size: 0.95rem;
  }
  .done {
    margin: 0.3rem 0 0.8rem;
    font-style: italic;
  }
  .act {
    background: none;
    border: 1px solid var(--ink);
    border-radius: 0;
    padding: 0.45rem 1.1rem;
    cursor: pointer;
    font-size: 1rem;
  }
  .act:hover:not(:disabled) {
    background: var(--ink);
    color: var(--paper);
  }
  .act:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }
  .link {
    background: none;
    border: 0;
    padding: 0.1rem 0;
    border-bottom: 1px solid var(--ink-3);
    color: var(--ink-2);
    cursor: pointer;
    white-space: nowrap;
  }
  .link:disabled {
    opacity: 0.5;
    cursor: wait;
  }
  .help .link {
    margin-left: 0.4rem;
  }
  @media (max-width: 860px) {
    dialog {
      width: 100vw;
      border-left: 0;
    }
  }
</style>
