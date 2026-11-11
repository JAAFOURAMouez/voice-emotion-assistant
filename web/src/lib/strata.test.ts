import { describe, expect, it } from "vitest";
import type { HistoryItem } from "./api/history";
import { edgeClip, LAYOUT, layoutStrata, thicknessOf } from "./strata";

const NOW = new Date(2026, 10, 11, 18, 0); // Wed 11 Nov 2026, local time

function item(id: number, daysAgo: number, hour: number, confidence: number | null, label: HistoryItem["label"] = "stress"): HistoryItem {
  const date = new Date(2026, 10, 11 - daysAgo, hour, 0);
  return { id, session_id: 1, timestamp: date.toISOString(), transcript: `t${id}`, label: confidence === null ? null : label, confidence };
}

describe("strata", () => {
  it("makes one layer per day, newest first, with empty days kept", () => {
    const { layers } = layoutStrata([item(1, 0, 9, 0.5), item(2, 2, 9, 0.8)], NOW);
    expect(layers).toHaveLength(7);
    expect(layers[0]!.title).toBe("mer. 11/11");
    expect(layers[0]!.bands.map((b) => b.item.id)).toEqual([1]);
    expect(layers[1]!.bands).toEqual([]);
    expect(layers[2]!.bands.map((b) => b.item.id)).toEqual([2]);
  });

  it("deposits the latest interaction on top of its day", () => {
    const { layers } = layoutStrata([item(1, 0, 9, 0.5), item(2, 0, 15, 0.5)], NOW);
    expect(layers[0]!.bands.map((b) => b.item.id)).toEqual([2, 1]);
  });

  it("makes thickness follow confidence, typed messages a hairline", () => {
    expect(thicknessOf(item(1, 0, 9, 1))).toBe(LAYOUT.maxThickness);
    expect(thicknessOf(item(1, 0, 9, 0))).toBe(LAYOUT.minThickness);
    expect(thicknessOf(item(1, 0, 9, null))).toBe(LAYOUT.hairline);
  });

  it("ignores interactions older than the week", () => {
    const { layers } = layoutStrata([item(1, 9, 9, 0.5)], NOW);
    expect(layers.every((l) => l.bands.length === 0)).toBe(true);
  });

  it("shrinks bands to fit, never below a visible floor", () => {
    const many = Array.from({ length: 80 }, (_, i) => item(i, i % 7, 8 + (i % 10), 1));
    const { layers, scale } = layoutStrata(many, NOW, { maxHeight: 400 });
    expect(scale).toBeLessThan(1);
    const bands = layers.flatMap((l) => l.bands);
    expect(bands.every((b) => b.thickness >= LAYOUT.floor)).toBe(true);
  });

  it("gives each interaction stable, irregular edges", () => {
    const a = layoutStrata([item(5, 0, 9, 0.5)], NOW).layers[0]!.bands[0]!;
    const b = layoutStrata([item(5, 0, 9, 0.5)], NOW).layers[0]!.bands[0]!;
    expect(a.insetLeft).toBe(b.insetLeft);
    expect(a.insetLeft).toBeGreaterThanOrEqual(1);
    expect(a.insetLeft).toBeLessThanOrEqual(8);
  });
});

describe("edges", () => {
  it("are stable and stay within the band", () => {
    expect(edgeClip(3, 10)).toBe(edgeClip(3, 10));
    expect(edgeClip(3, 10)).not.toBe(edgeClip(4, 10));
    const offsets = [...edgeClip(3, 2).matchAll(/([\d.]+)px/g)].map((m) => Number(m[1]));
    expect(Math.max(...offsets)).toBeLessThanOrEqual(0.5);
  });
});
