import { HEAD_COLORS, StepEvent } from "../api";

// The 5-head XD-PRM breakdown for the selected step — the core explainability panel.
export function HeadScores({ step }: { step: StepEvent | null }) {
  if (!step) return <div className="panel muted">Select a step to inspect its verifier scores.</div>;
  return (
    <div className="panel">
      <h3>
        Step {step.idx + 1} · R<sub>aggregate</sub> {step.r_aggregate.toFixed(2)}
      </h3>
      {Object.entries(HEAD_COLORS).map(([head, color]) => {
        const value = step.per_head[head];
        return (
          <div className="bar-row" key={head}>
            <span className="bar-label">{head}</span>
            <div className="bar-track">
              <div
                className="bar-fill"
                style={{ width: `${(value ?? 0) * 100}%`, background: color }}
              />
            </div>
            <span className="bar-value">{value === undefined ? "—" : value.toFixed(2)}</span>
          </div>
        );
      })}
      <p className="muted">
        uncertainty {step.uncertainty.toFixed(2)} · decision <b>{step.decision}</b>
        {step.pruned.length > 0 && ` · pruned ${step.pruned.length} weaker branch(es)`}
      </p>
    </div>
  );
}
