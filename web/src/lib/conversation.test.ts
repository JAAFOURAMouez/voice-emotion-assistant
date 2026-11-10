import { describe, expect, it } from "vitest";
import type { Estimate, ServerMessage } from "./api/types";
import { applyMessage, type Turn } from "./conversation";

function run(messages: ServerMessage[]): Turn[] {
  let id = 0;
  return messages.reduce<Turn[]>((turns, m) => applyMessage(turns, m, () => ++id), []);
}

const estimate: Estimate = {
  label: "stress", confidence: 0.7, scores: { neutre: 0.1, joie: 0.1, tristesse: 0.1, stress: 0.7 },
  arousal: 0.7, text_terms: [], phrase: "Voix plus tendue que d'habitude", learning: false, baseline_turns: 6,
};

describe("conversation", () => {
  it("builds a turn from transcript, estimate and streamed reply", () => {
    const [turn] = run([
      { type: "transcript", text: "Bonjour", typed: false },
      { type: "emotion", ...estimate },
      { type: "reply_start" },
      { type: "reply_chunk", text: "Bon" },
      { type: "reply_chunk", text: "jour." },
      { type: "reply_end", text: "Bonjour.", error: null },
      { type: "done" },
    ]);
    expect(turn).toMatchObject({ said: "Bonjour", reply: "Bonjour.", complete: true });
    expect(turn!.estimate?.label).toBe("stress");
  });

  it("starts a turn without user words for an answer to an offer", () => {
    const turns = run([
      { type: "transcript", text: "a", typed: true },
      { type: "done" },
      { type: "reply_start" },
      { type: "reply_chunk", text: "Très bien." },
      { type: "done" },
    ]);
    expect(turns).toHaveLength(2);
    expect(turns[1]).toMatchObject({ said: null, reply: "Très bien.", complete: true });
  });

  it("keeps a bounded history", () => {
    const messages: ServerMessage[] = [];
    for (let i = 0; i < 30; i++) messages.push({ type: "transcript", text: `t${i}`, typed: true }, { type: "done" });
    const turns = run(messages);
    expect(turns.length).toBeLessThanOrEqual(12);
    expect(turns.at(-1)!.said).toBe("t29");
  });
});

import { cueText } from "./emotions";
import { typeset } from "./texts";

describe("display", () => {
  it("never leaves a question mark alone at the start of a line", () => {
    expect(typeset("n'est-ce pas ?")).toBe("n'est-ce pas\u202f?");
    expect(typeset("« oui »")).toBe("«\u202foui\u202f»");
  });

  it("shows the closest cue for a neutral estimate", () => {
    const d = (v: number) => v.toFixed(2).replace(".", ",");
    const neutral: Estimate = { ...estimate, label: "neutre", confidence: 0.14, scores: { neutre: 0.4, joie: 0.1, tristesse: 0.04, stress: 0.46 } };
    expect(cueText(neutral, d, "le plus proche :")).toBe("calme · le plus proche : tension 0,46");
    expect(cueText(estimate, d, "")).toBe("tension · 0,70");
  });
});
