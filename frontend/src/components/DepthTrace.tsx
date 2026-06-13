import { StepEvent } from "../api";

// Ribbon of the adaptive-depth controller's per-step decisions.
const DECISION_COLOR: Record<string, string> = {
  continue: "#475569",
  expand: "#f59e0b",
  exit: "#22c55e",
};

export function DepthTrace({ steps }: { steps: StepEvent[] }) {
  if (steps.length === 0) return null;
  return (
    <div className="depth-trace">
      <span className="muted">depth control:</span>
      {steps.map((s, i) => (
        <span
          key={i}
          className="depth-pill"
          style={{ background: DECISION_COLOR[s.decision] ?? "#475569" }}
          title={`step ${i + 1}: ${s.decision} (u=${s.uncertainty.toFixed(2)})`}
        >
          {s.decision}
        </span>
      ))}
    </div>
  );
}
