import { describe, expect, it } from "vitest";
import { blendMood, MOODS } from "./emotions";
import { Spring } from "./spring";

describe("moods", () => {
  it("blend towards the estimated emotion with confidence", () => {
    const weak = blendMood("stress", 0);
    const strong = blendMood("stress", 1);
    expect(strong.tension).toBeCloseTo(MOODS.stress.tension);
    expect(weak.tension).toBeGreaterThan(0);
    expect(weak.tension).toBeLessThan(strong.tension);
    expect(blendMood("neutre", 1)).toEqual(MOODS.neutre);
  });

  it("give each emotion a distinct silhouette", () => {
    expect(MOODS.tristesse.sag).toBeGreaterThan(MOODS.neutre.sag);
    expect(MOODS.joie.scale).toBeGreaterThan(MOODS.neutre.scale);
    expect(MOODS.stress.speed).toBeGreaterThan(MOODS.joie.speed);
  });
});

describe("spring", () => {
  it("settles on its target without jumping", () => {
    const s = new Spring(0, 1);
    const first = s.step(1 / 60);
    expect(first).toBeLessThan(0.1);
    for (let i = 0; i < 600; i++) s.step(1 / 60);
    expect(s.value).toBeCloseTo(1, 3);
  });
});
