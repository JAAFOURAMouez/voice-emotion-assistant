import { describe, expect, it } from "vitest";
import { level, pitch, rms } from "./analysis";
import { Resampler, toInt16 } from "./resample";

const sine = (freq: number, rate: number, seconds: number, amp = 0.5) =>
  Float32Array.from({ length: Math.round(rate * seconds) }, (_, i) => amp * Math.sin((2 * Math.PI * freq * i) / rate));

describe("Resampler", () => {
  it("turns 48 kHz into 16 kHz, streaming, with the right length", () => {
    const r = new Resampler(48000);
    const input = sine(440, 48000, 1);
    let total = 0;
    for (let i = 0; i < input.length; i += 128) total += r.push(input.subarray(i, i + 128)).length;
    expect(Math.abs(total - 16000)).toBeLessThan(40);
  });

  it("keeps a speech-band tone and removes what would alias", () => {
    const keep = new Resampler(48000).push(sine(440, 48000, 0.5));
    const alias = new Resampler(48000).push(sine(11000, 48000, 0.5));
    expect(rms(keep.subarray(100))).toBeGreaterThan(0.3);
    expect(rms(alias.subarray(100))).toBeLessThan(0.02);
  });

  it("handles 44.1 kHz", () => {
    const out = new Resampler(44100).push(sine(220, 44100, 1));
    expect(Math.abs(out.length - 16000)).toBeLessThan(40);
  });
});

describe("analysis", () => {
  it("estimates the pitch of a voiced tone and ignores silence", () => {
    expect(pitch(sine(200, 16000, 0.064), 16000)).toBeCloseTo(200, -1);
    expect(pitch(new Float32Array(1024), 16000)).toBe(0);
  });

  it("maps loudness to 0–1", () => {
    expect(level(new Float32Array(512))).toBe(0);
    expect(level(sine(200, 16000, 0.05, 0.25))).toBeGreaterThan(0.7);
  });

  it("converts to 16-bit PCM with clipping", () => {
    expect(Array.from(toInt16(Float32Array.from([0, 1, -1, 2])))).toEqual([0, 32767, -32768, 32767]);
  });
});
