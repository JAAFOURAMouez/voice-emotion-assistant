import { api } from "./server";
import type { EmotionLabel } from "./types";

export interface HistoryItem {
  id: number;
  session_id: number;
  timestamp: string; // ISO 8601
  transcript: string;
  label: EmotionLabel | null; // null: typed message, no vocal cue
  confidence: number | null;
}

export interface TrendItem {
  label: EmotionLabel;
  recent: number;
  previous: number;
  direction: "hausse" | "baisse" | "stable" | "inconnu";
}

export interface History {
  simulated: boolean;
  notice: string | null;
  interactions: HistoryItem[];
  trend: TrendItem[];
}

export async function fetchHistory(options: { demo?: boolean; days?: number } = {}): Promise<History> {
  const params = new URLSearchParams();
  if (options.demo) params.set("demo", "true");
  if (options.days) params.set("days", String(options.days));
  return api<History>(`/api/history?${params}`);
}

export async function clearHistory(): Promise<number> {
  return (await api<{ deleted: number }>("/api/history", { method: "DELETE" })).deleted;
}
