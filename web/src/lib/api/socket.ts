import type { ServerMessage } from "./types";

export type Connection = "connecting" | "open" | "closed";

/**
 * The conversation WebSocket, reconnected automatically. Binary frames are the
 * spoken reply: each one follows a {"type": "speech"} header.
 */
export class TurnSocket {
  private ws: WebSocket | null = null;
  private retry = 0;
  private closedByUser = false;

  constructor(
    private readonly onMessage: (message: ServerMessage) => void,
    private readonly onAudio: (wav: ArrayBuffer) => void,
    private readonly onConnection: (state: Connection) => void,
    private readonly url = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/api/turn`,
  ) {}

  connect(): void {
    this.closedByUser = false;
    this.onConnection("connecting");
    const ws = new WebSocket(this.url);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => {
      this.retry = 0;
      this.onConnection("open");
    };
    ws.onmessage = (event: MessageEvent<string | ArrayBuffer>) => {
      if (typeof event.data === "string") this.onMessage(JSON.parse(event.data) as ServerMessage);
      else this.onAudio(event.data);
    };
    ws.onclose = () => {
      this.ws = null;
      this.onConnection("closed");
      if (!this.closedByUser) {
        const delay = Math.min(10_000, 500 * 2 ** this.retry++);
        setTimeout(() => this.connect(), delay);
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
    this.ws?.close();
  }
}
