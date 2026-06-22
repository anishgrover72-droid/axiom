// Presentation color contracts for the AXIOM console. Data lives in api.ts; this is purely visual.

// XD-PRM head channels — distinct instrument hues legible on the warm-dark canvas.
export const HEAD_COLORS: Record<string, string> = {
  logic: "#ff8a4c",
  commonsense: "#f2c14e",
  consistency: "#5fc9c2",
  efficiency: "#9bd64f",
  confidence: "#e98aa6",
};

// Adaptive-depth controller decisions.
export const DECISION_COLORS: Record<string, string> = {
  continue: "#9a8c7e",
  expand: "#f2c14e",
  exit: "#9bd64f",
};

// Verifier reward as signal intensity: dim oxide-red (weak) → bright ember-gold (strong).
const RAMP: [number, [number, number, number]][] = [
  [0.0, [122, 42, 30]],
  [0.5, [214, 96, 38]],
  [1.0, [255, 198, 110]],
];

export function signalColor(r: number): string {
  const x = Math.min(1, Math.max(0, r));
  for (let i = 1; i < RAMP.length; i++) {
    const [hi, hc] = RAMP[i];
    const [lo, lc] = RAMP[i - 1];
    if (x <= hi) {
      const t = (x - lo) / (hi - lo);
      const [r0, g0, b0] = lc;
      const [r1, g1, b1] = hc;
      return `rgb(${Math.round(r0 + (r1 - r0) * t)}, ${Math.round(g0 + (g1 - g0) * t)}, ${Math.round(b0 + (b1 - b0) * t)})`;
    }
  }
  const [, last] = RAMP[RAMP.length - 1];
  return `rgb(${last[0]}, ${last[1]}, ${last[2]})`;
}
