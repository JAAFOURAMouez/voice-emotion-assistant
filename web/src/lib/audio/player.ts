import { level } from "./analysis";

/**
 * Plays the assistant's spoken reply, sentence after sentence, without gaps, and
 * exposes its loudness so the shape can pulse with it.
 */
export class SpeechPlayer {
  private context: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private frame: Float32Array<ArrayBuffer> | null = null;
  private sources = new Set<AudioBufferSourceNode>();
  private endsAt = 0;
  private chain: Promise<void> = Promise.resolve();
  private generation = 0;

  constructor(private readonly onPlaying: (playing: boolean) => void) {}

  /** Must be called from a user gesture once (browsers block audio otherwise). */
  unlock(): void {
    if (!this.context) {
      this.context = new AudioContext();
      this.analyser = this.context.createAnalyser();
      this.analyser.fftSize = 1024;
      this.analyser.connect(this.context.destination);
      this.frame = new Float32Array(this.analyser.fftSize);
    }
    if (this.context.state === "suspended") void this.context.resume();
  }

  get playing(): boolean {
    return this.sources.size > 0;
  }

  enqueue(wav: ArrayBuffer): void {
    this.unlock();
    const generation = this.generation;
    // Decode in arrival order: sentences must never swap.
    this.chain = this.chain.then(async () => {
      const context = this.context!;
      let buffer: AudioBuffer;
      try {
        buffer = await context.decodeAudioData(wav);
      } catch {
        return;
      }
      if (generation !== this.generation) return;
      const source = context.createBufferSource();
      source.buffer = buffer;
      source.connect(this.analyser!);
      const start = Math.max(context.currentTime + 0.03, this.endsAt);
      source.start(start);
      this.endsAt = start + buffer.duration;
      this.sources.add(source);
      this.onPlaying(true);
      source.onended = () => {
        this.sources.delete(source);
        if (this.sources.size === 0) this.onPlaying(false);
      };
    });
  }

  /** Stop speaking now (the user starts talking, or a new turn begins). */
  stop(): void {
    this.generation++;
    for (const source of this.sources) {
      source.onended = null;
      try {
        source.stop();
      } catch {
        /* already stopped */
      }
    }
    this.sources.clear();
    this.endsAt = 0;
    this.onPlaying(false);
  }

  read(): { level: number; pitch: number } {
    if (!this.analyser || !this.frame) return { level: 0, pitch: 0 };
    this.analyser.getFloatTimeDomainData(this.frame);
    return { level: level(this.frame), pitch: 0 };
  }
}
