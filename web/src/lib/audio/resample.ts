/**
 * Streaming resampler to 16 kHz mono with an anti-aliasing low-pass filter.
 * Used inside the AudioWorklet (the browser's own rate is usually 44.1 or 48 kHz;
 * asking the AudioContext for 16 kHz is not portable across browsers).
 */
export class Resampler {
  readonly ratio: number;
  private readonly taps: Float32Array;
  private history: Float32Array;
  private position = 0; // absolute fractional read position in the filtered stream
  private offset = 0; // absolute index of filtered[0]
  private filtered: number[] = [];

  constructor(
    readonly inputRate: number,
    readonly outputRate = 16000,
    tapCount = 63,
  ) {
    this.ratio = inputRate / outputRate;
    this.taps = Resampler.lowPass(tapCount, (0.45 * outputRate) / inputRate);
    this.history = new Float32Array(tapCount - 1);
  }

  /** Windowed-sinc low-pass (Blackman), cutoff as a fraction of the input rate. */
  static lowPass(count: number, cutoff: number): Float32Array {
    const taps = new Float32Array(count);
    const middle = (count - 1) / 2;
    let sum = 0;
    for (let i = 0; i < count; i++) {
      const x = i - middle;
      const sinc = x === 0 ? 2 * cutoff : Math.sin(2 * Math.PI * cutoff * x) / (Math.PI * x);
      const window =
        0.42 - 0.5 * Math.cos((2 * Math.PI * i) / (count - 1)) +
        0.08 * Math.cos((4 * Math.PI * i) / (count - 1));
      taps[i] = sinc * window;
      sum += taps[i]!;
    }
    for (let i = 0; i < count; i++) taps[i]! /= sum;
    return taps;
  }

  /** Feed input samples, get the output samples that are now complete. */
  push(input: Float32Array): Float32Array {
    if (this.ratio === 1) return input.slice();
    const n = this.taps.length;
    const buffer = new Float32Array(this.history.length + input.length);
    buffer.set(this.history);
    buffer.set(input, this.history.length);
    for (let i = 0; i + n <= buffer.length; i++) {
      let acc = 0;
      for (let k = 0; k < n; k++) acc += buffer[i + k]! * this.taps[k]!;
      this.filtered.push(acc);
    }
    this.history = buffer.slice(buffer.length - (n - 1));

    const out: number[] = [];
    for (;;) {
      const i = Math.floor(this.position) - this.offset;
      if (i + 1 >= this.filtered.length) break;
      const frac = this.position - Math.floor(this.position);
      out.push(this.filtered[i]! * (1 - frac) + this.filtered[i + 1]! * frac);
      this.position += this.ratio;
    }
    const drop = Math.max(0, Math.min(Math.floor(this.position) - this.offset, this.filtered.length));
    this.filtered = this.filtered.slice(drop);
    this.offset += drop;
    return Float32Array.from(out);
  }
}

/** Float [-1, 1] → 16-bit little-endian PCM (what the server expects). */
export function toInt16(samples: Float32Array): Int16Array {
  const out = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]!));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}
