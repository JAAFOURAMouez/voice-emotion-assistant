import type { Estimate, Exercise, ServerMessage } from "./api/types";

export interface Turn {
  id: number;
  said: string | null; // null: the assistant speaks on its own (answer to an offer)
  typed: boolean;
  estimate: Estimate | null;
  reply: string;
  error: string | null;
  offer: { text: string; exercise: Exercise } | null;
  complete: boolean;
}

const MAX_TURNS = 12;

/** Folds the server's events into a list of turns (pure, easy to test). */
export function applyMessage(turns: Turn[], message: ServerMessage, nextId: () => number): Turn[] {
  const last = turns.at(-1);
  const update = (patch: Partial<Turn>): Turn[] => [...turns.slice(0, -1), { ...last!, ...patch }];
  switch (message.type) {
    case "transcript":
      return [
        ...turns.slice(-(MAX_TURNS - 1)),
        { id: nextId(), said: message.text, typed: message.typed, estimate: null, reply: "", error: null, offer: null, complete: false },
      ];
    case "emotion":
      return last && !last.complete ? update({ estimate: message }) : turns;
    case "reply_start":
      if (last && !last.complete) return turns;
      return [
        ...turns.slice(-(MAX_TURNS - 1)),
        { id: nextId(), said: null, typed: false, estimate: null, reply: "", error: null, offer: null, complete: false },
      ];
    case "reply_chunk":
      return last && !last.complete ? update({ reply: last.reply + message.text }) : turns;
    case "reply_end":
      return last && !last.complete ? update({ reply: message.text, error: message.error }) : turns;
    case "offer":
      return last ? update({ offer: { text: message.text, exercise: message.exercise } }) : turns;
    case "done":
      return last && !last.complete ? update({ complete: true }) : turns;
    default:
      return turns;
  }
}
