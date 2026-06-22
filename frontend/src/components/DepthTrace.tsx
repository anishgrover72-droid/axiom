import { StepEvent } from "../api";
import { DECISION_COLORS } from "../theme";

// Depth profile: the adaptive-depth controller's per-step decisions, as tinted signal pills.
export function DepthTrace({ steps }: { steps: StepEvent[] }) {
  if (steps.length === 0) return null;
  return (
    <div className="depth">
      <span className="eyebrow">depth profile</span>
      <div className="depth-pills">
        {steps.map((s, i) => {
          const c = DECISION_COLORS[s.decision] ?? "#9a8c7e";
          return (
            <span
              key={i}
              className="depth-pill"
              style={{
                background: `color-mix(in srgb, ${c} 14%, transparent)`,
                borderColor: `color-mix(in srgb, ${c} 36%, transparent)`,
                color: c,
              }}
              title={`step ${i + 1}: ${s.decision} (u=${s.uncertainty.toFixed(2)})`}
            >
              <span className="depth-dot" style={{ background: c }} />
              {s.decision}
            </span>
          );
        })}
      </div>
    </div>
  );
}
