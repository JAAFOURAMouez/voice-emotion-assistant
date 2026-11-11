export type EmotionLabel = "neutre" | "joie" | "tristesse" | "stress";

export interface Estimate {
  label: EmotionLabel;
  confidence: number;
  scores: Record<EmotionLabel, number>;
  arousal: number;
  text_terms: string[];
  phrase: string;
  learning: boolean;
  baseline_turns: number;
}

export interface Step {
  instruction: string;
  seconds: number;
  kind: "inspire" | "expire" | "hold" | "free";
}

export interface Exercise {
  id: string;
  name: string;
  summary: string;
  caution: string;
  cycles: number;
  total_seconds: number;
  duration_text: string;
  steps: Step[];
  introduction: string;
  spoken?: string[];
}

export interface LlmState {
  ok: boolean;
  message: string | null;
  provider?: string;
  provider_name?: string;
  model?: string;
  local?: boolean;
}

export type ServerMessage =
  | { type: "status"; state: "loading" | "ready" | "error"; message: string | null; llm: LlmState; voice: boolean; warnings: string[]; notice?: string }
  | { type: "recording" }
  | { type: "silence" }
  | { type: "transcript"; text: string; typed: boolean; duration_s?: number }
  | ({ type: "emotion" } & Estimate)
  | { type: "reply_start" }
  | { type: "reply_chunk"; text: string }
  | { type: "speech"; index: number; bytes: number }
  | { type: "reply_end"; text: string; error: string | null }
  | { type: "offer"; label: EmotionLabel; text: string; exercise: Exercise }
  | ({ type: "exercise" } & Exercise)
  | { type: "not_understood"; message: string }
  | { type: "error"; code: string; message: string }
  | { type: "done"; timings?: Record<string, number> };
