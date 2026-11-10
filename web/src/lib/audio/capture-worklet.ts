/// <reference lib="webworker" />
// AudioWorklet: microphone at the context rate → 16 kHz mono 16-bit PCM chunks.
import { Resampler, toInt16 } from "./resample";

declare const sampleRate: number;
declare function registerProcessor(name: string, ctor: unknown): void;
declare class AudioWorkletProcessor {
  readonly port: MessagePort;
}

const CHUNK = 1600; // 100 ms at 16 kHz

class CaptureProcessor extends AudioWorkletProcessor {
  private resampler = new Resampler(sampleRate, 16000);
  private pending: number[] = [];
  private active = false;

  constructor() {
    super();
    this.port.onmessage = (event: MessageEvent<string>) => {
      if (event.data === "start") {
        this.resampler = new Resampler(sampleRate, 16000);
        this.pending = [];
        this.active = true;
      } else if (event.data === "stop") {
        this.active = false;
        this.flush(true);
        this.port.postMessage("stopped");
      }
    };
  }

  private flush(all: boolean): void {
    while (this.pending.length >= CHUNK || (all && this.pending.length > 0)) {
      const part = Float32Array.from(this.pending.splice(0, CHUNK));
      const pcm = toInt16(part);
      this.port.postMessage(pcm.buffer, [pcm.buffer]);
    }
  }

  process(inputs: Float32Array[][]): boolean {
    const channels = inputs[0];
    if (!this.active || !channels || channels.length === 0) return true;
    const length = channels[0]!.length;
    const mono = new Float32Array(length);
    for (const channel of channels) for (let i = 0; i < length; i++) mono[i]! += channel[i]! / channels.length;
    for (const sample of this.resampler.push(mono)) this.pending.push(sample);
    this.flush(false);
    return true;
  }
}

registerProcessor("voxemo-capture", CaptureProcessor);
