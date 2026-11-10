import type { EmotionLabel, Estimate } from "./api/types";

export const LABELS: EmotionLabel[] = ["neutre", "joie", "tristesse", "stress"];

/** Words shown next to the shape: the emotion is never conveyed by colour alone. */
export const LABEL_WORDS: Record<EmotionLabel, string> = {
  neutre: "calme",
  joie: "entrain",
  tristesse: "abattement",
  stress: "tension",
};

/**
 * How the voiceprint looks for each estimated emotion (blended by confidence):
 * stress = taut, spiky, fast; joy = ample, lively; sadness = slow, sagging;
 * neutral = a calm circle.
 */
export interface Mood {
  scale: number; // overall size
  wobble: number; // low-frequency deformation
  tension: number; // high-frequency, sharp contour (0–1)
  sag: number; // droop of the lower half (0–1)
  speed: number; // how fast the contour moves
}

export const MOODS: Record<EmotionLabel, Mood> = {
  neutre: { scale: 1.0, wobble: 0.018, tension: 0.0, sag: 0.0, speed: 0.12 },
  joie: { scale: 1.12, wobble: 0.085, tension: 0.08, sag: -0.04, speed: 0.55 },
  tristesse: { scale: 0.9, wobble: 0.04, tension: 0.0, sag: 0.2, speed: 0.06 },
  stress: { scale: 0.95, wobble: 0.03, tension: 0.85, sag: 0.0, speed: 1.35 },
};

export function blendMood(label: EmotionLabel, confidence: number): Mood {
  const target = MOODS[label];
  const base = MOODS.neutre;
  const t = Math.max(0, Math.min(1, 0.35 + 0.65 * confidence)); // always visible a bit
  const mix = (a: number, b: number) => a + (b - a) * t;
  return {
    scale: mix(base.scale, target.scale),
    wobble: mix(base.wobble, target.wobble),
    tension: mix(base.tension, target.tension),
    sag: mix(base.sag, target.sag),
    speed: mix(base.speed, target.speed),
  };
}

/** "tension · 0,72"; a neutral estimate shows its closest other cue instead of a confidence. */
export function cueText(e: Estimate, decimal: (v: number) => string, nothingMarked: string): string {
  if (e.label !== "neutre") return `${LABEL_WORDS[e.label]} · ${decimal(e.confidence)}`;
  const others = (Object.entries(e.scores) as [EmotionLabel, number][]).filter(([k]) => k !== "neutre");
  if (others.length === 0) return LABEL_WORDS.neutre;
  const [closest, score] = others.reduce((a, b) => (b[1] > a[1] ? b : a));
  return `${LABEL_WORDS.neutre} · ${nothingMarked} ${LABEL_WORDS[closest]} ${decimal(score)}`;
}
