import { api } from "./server";

export interface LocalModel {
  name: string;
  size_gb: number;
  parameters: string;
  thinking: boolean;
}

export interface ProviderInfo {
  id: string;
  name: string;
  base_url: string;
  needs_key: boolean;
  has_key: boolean;
  key_hint: string | null;
}

export interface CurrentModel {
  provider: string;
  provider_name: string;
  model: string;
  local: boolean;
}

export interface LlmSettings {
  current: CurrentModel;
  cloud_allowed: boolean;
  ollama_models: LocalModel[];
  providers: ProviderInfo[];
}

export interface LlmRequest {
  provider: string;
  model?: string;
  api_key?: string;
  base_url?: string;
}

export const fetchLlm = () => api<LlmSettings>("/api/llm");

export const chooseLlm = (request: LlmRequest) =>
  api<LlmSettings>("/api/llm", { method: "PUT", body: JSON.stringify(request) });

export const listRemoteModels = async (request: LlmRequest) =>
  (await api<{ models: string[] }>("/api/llm/models", { method: "POST", body: JSON.stringify(request) })).models;

export const forgetKey = (provider: string) =>
  api<LlmSettings>(`/api/llm/keys/${encodeURIComponent(provider)}`, { method: "DELETE" });
