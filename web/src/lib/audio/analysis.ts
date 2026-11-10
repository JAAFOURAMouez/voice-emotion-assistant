/** Light, real-time voice analysis for the visual only (the server does the real one). */

export function rms(frame: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < frame.length; i++) sum += frame[i]! * frame[i]!;
  return Math.sqrt(sum / Math.max(1, frame.length));
}

/** Loudness mapped to 0–1 on a log scale (-60 dBFS → 0, -12 dBFS → 1). */
export function level(frame: Float32Array): number {
  const db = 20 * Math.log10(rms(frame) + 1e-9);
  return Math.max(0, Math.min(1, (db + 60) / 48));
}

/**
 * Fundamental frequency by normalized autocorrelation, or 0 when unvoiced.
 * Good enough to make the shape follow the voice; not used for the estimate.
 */
export function pitch(frame: Float32Array, sampleRate: number, fmin = 70, fmax = 400): number {
  const n = frame.length;
  const minLag = Math.floor(sampleRate / fmax);
  const maxLag = Math.min(n - 1, Math.ceil(sampleRate / fmin));
  if (rms(frame) < 0.005 || maxLag <= minLag) return 0;
  let energy = 0;
  for (let i = 0; i < n; i++) energy += frame[i]! * frame[i]!;
  let bestLag = 0;
  let best = 0;
  for (let lag = minLag; lag <= maxLag; lag++) {
    let acc = 0;
    for (let i = 0; i + lag < n; i++) acc += frame[i]! * frame[i + lag]!;
    const normalized = acc / energy;
    if (normalized > best) {
      best = normalized;
      bestLag = lag;
    }
  }
  return best > 0.45 && bestLag > 0 ? sampleRate / bestLag : 0;
}
