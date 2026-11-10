import type { EmotionLabel } from "./api/types";
import { blendMood, type Mood } from "./emotions";
import { Spring } from "./spring";

export type VoiceprintMode = "idle" | "listening" | "thinking" | "speaking" | "breathing";

const POINTS = 180;
// Harmonics of the contour: low ones make it breathe, high ones make it taut.
const LOW = [2, 3, 5];
const HIGH = [9, 13, 17, 23];

/**
 * The single organic shape: the user's voice while they speak, the estimated
 * emotion between turns, the assistant's voice while it answers, and the breathing
 * guide during an exercise. Drawn in Canvas 2D, driven by springs.
 */
export class VoiceprintEngine {
  mode: VoiceprintMode = "idle";
  reducedMotion = false;
  /** Read every frame while listening or speaking (microphone or playback level). */
  liveSource: (() => { level: number; pitch: number }) | null = null;

  private readonly springs = {
    scale: new Spring(1, 1, 18, 7),
    wobble: new Spring(0.018, 0.018, 8, 5),
    tension: new Spring(0, 0, 8, 5),
    sag: new Spring(0, 0, 6, 4.5),
    speed: new Spring(0.12, 0.12, 6, 4.5),
    live: new Spring(0, 0, 140, 18), // input/output loudness, fast
    pitch: new Spring(0.5, 0.5, 40, 12),
    breath: new Spring(0, 0, 10, 6),
  };
  private phase = Math.random() * 100;
  private seeds = [...LOW, ...HIGH].map(() => Math.random() * Math.PI * 2);
  private frame = 0;
  private last = 0;
  private accent = "#008670";
  private ink = "#1c1a17";
  private paper = "#f3ede2";

  constructor(private readonly canvas: HTMLCanvasElement) {}

  start(): void {
    const loop = (now: number) => {
      const dt = Math.min(0.05, this.last ? (now - this.last) / 1000 : 0.016);
      this.last = now;
      this.step(dt);
      this.draw();
      this.frame = requestAnimationFrame(loop);
    };
    this.frame = requestAnimationFrame(loop);
  }

  stop(): void {
    cancelAnimationFrame(this.frame);
  }

  setColors(accent: string, ink: string, paper: string): void {
    this.accent = accent;
    this.ink = ink;
    this.paper = paper;
  }

  /** Slowly evolve towards the estimated emotion (after a turn). */
  setMood(label: EmotionLabel, confidence: number): void {
    const mood: Mood = blendMood(label, confidence);
    this.springs.scale.target = mood.scale;
    this.springs.wobble.target = mood.wobble;
    this.springs.tension.target = mood.tension;
    this.springs.sag.target = mood.sag;
    this.springs.speed.target = mood.speed;
  }

  /** Live signal: the user's microphone, or the assistant's voice (0–1). */
  setLive(level: number, pitchHz = 0): void {
    this.springs.live.target = level;
    if (pitchHz > 0) this.springs.pitch.target = Math.max(0, Math.min(1, (pitchHz - 80) / 260));
  }

  /** Breathing guide: -1 (exhaled) … 1 (inhaled). */
  setBreath(value: number): void {
    this.springs.breath.target = value;
  }

  private step(dt: number): void {
    if (this.liveSource && (this.mode === "listening" || this.mode === "speaking")) {
      const frame = this.liveSource();
      this.setLive(frame.level, frame.pitch);
    } else {
      this.springs.live.target = 0;
    }
    for (const spring of Object.values(this.springs)) spring.step(dt);
    const motion = this.reducedMotion ? 0.04 : 1;
    const liveBoost = this.mode === "listening" || this.mode === "speaking" ? this.springs.live.value : 0;
    this.phase += dt * motion * (this.springs.speed.value + 1.8 * liveBoost);
  }

  private draw(): void {
    const canvas = this.canvas;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const s = this.springs;
    const reduced = this.reducedMotion;
    const live = this.mode === "listening" || this.mode === "speaking" ? s.live.value : 0;
    const liveGain = reduced ? 0.15 : 1;
    const breath = this.mode === "breathing" ? s.breath.value : 0;
    const radius =
      Math.min(width, height) *
      0.3 *
      s.scale.value *
      (1 + 0.16 * live * liveGain) *
      (1 + 0.22 * breath);
    const cx = width / 2;
    const cy = height / 2 - radius * 0.04;

    const wobble = (s.wobble.value + 0.07 * live * liveGain) * (this.mode === "breathing" ? 0.4 : 1);
    const tension = s.tension.value;
    const sag = s.sag.value;
    const pitchShift = 1 + 0.6 * (s.pitch.value - 0.5) * live;
    const t = this.phase;

    const points: [number, number][] = [];
    for (let i = 0; i < POINTS; i++) {
      const a = (i / POINTS) * Math.PI * 2;
      let r = 1;
      LOW.forEach((k, j) => {
        r += (wobble / (1 + j * 0.6)) * Math.sin(k * a + this.seeds[j]! + t * (0.7 + 0.3 * j) * pitchShift);
      });
      HIGH.forEach((k, j) => {
        const wave = Math.sin(k * a + this.seeds[LOW.length + j]! - t * (1.5 + j * 0.45));
        // Sharpened peaks: a taut, nervous contour.
        const sharp = Math.sign(wave) * Math.abs(wave) ** 0.55;
        r += tension * (0.026 / (1 + j * 0.35)) * sharp * (1 + 0.8 * live);
      });
      let x = Math.cos(a) * r;
      let y = Math.sin(a) * r;
      // Sagging: the lower half drops and widens, the top flattens.
      if (sag !== 0) {
        const down = Math.max(0, y);
        y += sag * (0.32 * down * down + 0.12);
        x *= 1 + sag * 0.22 * down;
        if (y < 0) y *= 1 - sag * 0.35;
      }
      points.push([cx + x * radius, cy + y * radius]);
    }

    const path = new Path2D();
    const n = points.length;
    for (let i = 0; i < n; i++) {
      const p0 = points[(i - 1 + n) % n]!;
      const p1 = points[i]!;
      const p2 = points[(i + 1) % n]!;
      const p3 = points[(i + 2) % n]!;
      if (i === 0) path.moveTo(p1[0], p1[1]);
      // Catmull-Rom → Bézier: a smooth, closed contour.
      path.bezierCurveTo(
        p1[0] + (p2[0] - p0[0]) / 6,
        p1[1] + (p2[1] - p0[1]) / 6,
        p2[0] - (p3[0] - p1[0]) / 6,
        p2[1] - (p3[1] - p1[1]) / 6,
        p2[0],
        p2[1],
      );
    }
    path.closePath();

    // Printed look: the colour plate slightly off-register from the ink line.
    ctx.save();
    ctx.translate(radius * 0.025, radius * 0.02);
    ctx.fillStyle = this.accent;
    ctx.globalAlpha = 0.88;
    ctx.fill(path);
    ctx.restore();

    // Fine hatching inside the shape: texture, and a cue that is not colour.
    ctx.save();
    ctx.clip(path);
    ctx.strokeStyle = this.paper;
    ctx.globalAlpha = 0.16;
    ctx.lineWidth = 1;
    const step = 7;
    const extent = radius * 1.6;
    for (let d = -extent; d < extent; d += step) {
      ctx.beginPath();
      ctx.moveTo(cx + d - extent, cy - extent);
      ctx.lineTo(cx + d + extent, cy + extent);
      ctx.stroke();
    }
    ctx.restore();

    ctx.strokeStyle = this.ink;
    ctx.lineWidth = 1.25;
    ctx.globalAlpha = 1;
    ctx.stroke(path);
  }
}
