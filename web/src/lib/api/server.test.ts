import { describe, expect, it } from "vitest";
import { normalizeAddress } from "./server";

describe("normalizeAddress", () => {
  it("keeps only the origin and adds https", () => {
    expect(normalizeAddress("mon-pc.reseau.ts.net/")).toBe("https://mon-pc.reseau.ts.net");
    expect(normalizeAddress(" https://exemple.org:8443/api ")).toBe("https://exemple.org:8443");
    expect(normalizeAddress("http://localhost:8000")).toBe("http://localhost:8000");
  });
  it("empty means this page's own server", () => {
    expect(normalizeAddress("  ")).toBe("");
  });
  it("rejects nonsense", () => {
    expect(() => normalizeAddress("https://")).toThrow();
  });
});
