/** Damped spring: values move with inertia towards their target, never jump. */
export class Spring {
  velocity = 0;

  constructor(
    public value: number,
    public target = value,
    public stiffness = 60,
    public damping = 14,
  ) {}

  step(dt: number): number {
    const force = this.stiffness * (this.target - this.value) - this.damping * this.velocity;
    this.velocity += force * dt;
    this.value += this.velocity * dt;
    return this.value;
  }

  snap(value: number): void {
    this.value = this.target = value;
    this.velocity = 0;
  }
}
