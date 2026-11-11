import type { HistoryItem } from "./api/history";
import { T } from "./texts";

export interface Band {
  item: HistoryItem;
  thickness: number; // px: the confidence of the estimate
  insetLeft: number; // % — irregular edges, like real sediment
  insetRight: number;
}

export interface DayLayer {
  key: string; // yyyy-mm-dd, local time
  title: string; // "mar. 10/11"
  bands: Band[]; // newest first (deposited on top)
}

export const LAYOUT = {
  minThickness: 3,
  maxThickness: 15,
  hairline: 1.5, // typed message: no estimate, just a trace
  dayHeader: 25, // 22 px label + 3 px gap
  floor: 1.5,
};

/** Small deterministic hash: the same interaction always gets the same edges. */
function noise(id: number, salt: number): number {
  let x = Math.imul(id ^ (salt * 0x9e3779b9), 0x85ebca6b);
  x ^= x >>> 13;
  x = Math.imul(x, 0xc2b2ae35);
  x ^= x >>> 16;
  return (x >>> 0) / 0xffffffff;
}

function dayKey(date: Date): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
}

export function dayTitle(date: Date): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${T.weekdays[(date.getDay() + 6) % 7]} ${pad(date.getDate())}/${pad(date.getMonth() + 1)}`;
}

export function thicknessOf(item: HistoryItem): number {
  if (item.label === null || item.confidence === null) return LAYOUT.hairline;
  const c = Math.max(0, Math.min(1, item.confidence));
  return LAYOUT.minThickness + c * (LAYOUT.maxThickness - LAYOUT.minThickness);
}

/**
 * The week as a geological section: one layer per day (newest on top), each
 * interaction a band whose thickness is its confidence. Bands shrink together
 * if the week does not fit in ``maxHeight``.
 */
export function layoutStrata(
  items: HistoryItem[],
  now: Date,
  { days = 7, maxHeight = Infinity }: { days?: number; maxHeight?: number } = {},
): { layers: DayLayer[]; scale: number } {
  const layers: DayLayer[] = [];
  const byKey = new Map<string, DayLayer>();
  for (let offset = 0; offset < days; offset++) {
    const date = new Date(now.getFullYear(), now.getMonth(), now.getDate() - offset);
    const layer = { key: dayKey(date), title: dayTitle(date), bands: [] as Band[] };
    layers.push(layer);
    byKey.set(layer.key, layer);
  }
  const sorted = [...items].sort((a, b) => b.timestamp.localeCompare(a.timestamp));
  for (const item of sorted) {
    const layer = byKey.get(dayKey(new Date(item.timestamp)));
    if (!layer) continue;
    layer.bands.push({
      item,
      thickness: thicknessOf(item),
      insetLeft: 1 + 7 * noise(item.id, 1),
      insetRight: 1 + 7 * noise(item.id, 2),
    });
  }
  const content = layers.reduce((sum, l) => sum + l.bands.reduce((s, b) => s + b.thickness + 1, 0), 0);
  const available = maxHeight - layers.length * LAYOUT.dayHeader - 8; // 8 px of breathing room
  const scale = content > 0 && available < content ? Math.max(0.15, available / content) : 1;
  if (scale < 1) {
    for (const layer of layers)
      for (const band of layer.bands) band.thickness = Math.max(LAYOUT.floor, band.thickness * scale);
  }
  return { layers, scale };
}

/**
 * Slightly uneven top and bottom edges, like sediment: a clip-path polygon whose
 * points move by at most a quarter of the band (stable for a given interaction).
 */
export function edgeClip(id: number, thickness: number): string {
  const amp = Math.min(1.4, thickness * 0.25);
  const steps = 8;
  const top: string[] = [];
  const bottom: string[] = [];
  for (let i = 0; i <= steps; i++) {
    const x = (i / steps) * 100;
    top.push(`${x.toFixed(1)}% ${(amp * noise(id, 10 + i)).toFixed(2)}px`);
    bottom.push(`${(100 - x).toFixed(1)}% calc(100% - ${(amp * noise(id, 30 + i)).toFixed(2)}px)`);
  }
  return `polygon(${[...top, ...bottom].join(", ")})`;
}
