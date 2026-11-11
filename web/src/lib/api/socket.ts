import { socketProtocols, socketUrl } from "./server";
import type { ServerMessage } from "./types";

/** "denied": wrong or missing access code (no automatic retry). */
export type Connection = "connecting" | "open" | "closed" | "denied";
const ACCESS_DENIED = 4401;

/**
 * The conversation WebSocket, reconnected automatically. Binary frames are the
 * spoken reply: each one follows a {"type": "speech"} header.
 */
export class TurnSocket {
  private ws: WebSocket | null = null;
  private retry = 0;
  private closedByUser = false;
  private timer: ReturnType<typeof setTimeout> | undefined;

  constructor(
    private readonly onMessage: (message: ServerMessage) => void,
    private readonly onAudio: (wav: ArrayBuffer) => void,
    private readonly onConnection: (state: Connection) => void,
  ) {}

  connect(): void {
    this.closedByUser = false;
    this.onConnection("connecting");
    let ws: WebSocket;
    try {
      ws = new WebSocket(socketUrl(), socketProtocols());
    } catch {
      this.onConnection("closed"); // malformed address
      return;
    }
    ws.binaryType = "arraybuffer";
    ws.onopen = () => {
      this.retry = 0;
      this.onConnection("open");
    };
    ws.onmessage = (event: MessageEvent<string | ArrayBuffer>) => {
      if (typeof event.data === "string") this.onMessage(JSON.parse(event.data) as ServerMessage);
      else this.onAudio(event.data);
    };
    ws.onclose = (event: CloseEvent) => {
      if (this.ws !== ws) return; // an older socket, replaced by restart()
      this.ws = null;
      if (event.code === ACCESS_DENIED) {
        this.onConnection("denied");
        return;
      }
      this.onConnection("closed");
      if (!this.closedByUser) {
        const delay = Math.min(10_000, 500 * 2 ** this.retry++);
        this.timer = setTimeout(() => this.connect(), delay);
      }
    };
    this.ws = ws;
  }

  get open(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  send(message: Record<string, unknown>): void {
    if (this.open) this.ws!.send(JSON.stringify(message));
  }

  sendAudio(pcm: ArrayBuffer): void {
    if (this.open) this.ws!.send(pcm);
  }

  close(): void {
    this.closedByUser = true;
    clearTimeout(this.timer);
    const ws = this.ws;
    this.ws = null;
    ws?.close();
  }

  /** New address or access code: reconnect now. */
  restart(): void {
    this.close();
    this.retry = 0;
    this.connect();
  }
}
