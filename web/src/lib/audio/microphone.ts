import workletUrl from "./capture-worklet.ts?worker&url";
import { level, pitch } from "./analysis";

export type MicState = "idle" | "requesting" | "ready" | "denied" | "unavailable";

export interface VoiceFrame {
  level: number; // 0–1
  pitch: number; // Hz, 0 when unvoiced
}

/**
 * Browser microphone: getUserMedia → AudioWorklet (16 kHz PCM for the server) and
 * an AnalyserNode (level and pitch for the shape). Permission is asked on first use.
 */
export class Microphone {
  state: MicState = "idle";
  private context: AudioContext | null = null;
  private node: AudioWorkletNode | null = null;
  private analyser: AnalyserNode | null = null;
  private frame: Float32Array<ArrayBuffer> | null = null;
  private onChunk: ((pcm: ArrayBuffer) => void) | null = null;
  private stopped: (() => void) | null = null;

  constructor(private readonly onState: (state: MicState) => void) {}

  private set(state: MicState): void {
    this.state = state;
    this.onState(state);
  }

  async open(): Promise<boolean> {
    if (this.state === "ready") return true;
    if (!navigator.mediaDevices?.getUserMedia) {
      this.set("unavailable");
      return false;
    }
    this.set("requesting");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true, // the assistant's own voice must not be heard back
          noiseSuppression: false, // keep the voice as it is: prosody matters
          autoGainControl: false,
        },
      });
      const context = new AudioContext();
      await context.audioWorklet.addModule(workletUrl);
      const source = context.createMediaStreamSource(stream);
      const node = new AudioWorkletNode(context, "voxemo-capture");
      node.port.onmessage = (event: MessageEvent<ArrayBuffer | string>) => {
        if (event.data === "stopped") {
          this.stopped?.();
          this.stopped = null;
        } else if (typeof event.data !== "string") {
          this.onChunk?.(event.data);
        }
      };
      const analyser = context.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);
      source.connect(node);
      this.context = context;
      this.node = node;
      this.analyser = analyser;
      this.frame = new Float32Array(analyser.fftSize);
      this.set("ready");
      return true;
    } catch (error) {
      const name = error instanceof DOMException ? error.name : "";
      this.set(name === "NotAllowedError" || name === "SecurityError" ? "denied" : "unavailable");
      return false;
    }
  }

  async start(onChunk: (pcm: ArrayBuffer) => void): Promise<boolean> {
    if (!(await this.open()) || !this.node || !this.context) return false;
    if (this.context.state === "suspended") await this.context.resume();
    this.onChunk = onChunk;
    this.node.port.postMessage("start");
    return true;
  }

  /** Stop recording; resolves once the last samples have been delivered. */
  stop(): Promise<void> {
    if (!this.node) return Promise.resolve();
    return new Promise((resolve) => {
      this.stopped = () => {
        this.onChunk = null;
        resolve();
      };
      this.node!.port.postMessage("stop");
    });
  }

  /** Current loudness and pitch, for the visual. */
  read(): VoiceFrame {
    if (!this.analyser || !this.frame || !this.context) return { level: 0, pitch: 0 };
    this.analyser.getFloatTimeDomainData(this.frame);
    return { level: level(this.frame), pitch: pitch(this.frame, this.context.sampleRate) };
  }
}
