import { StepEvent } from "../api";
import { DECISION_COLORS, HEAD_COLORS, signalColor } from "../theme";

// Instrument dial for the aggregate reward, arc tinted on the signal ramp.
function Dial({ value }: { value: number }) {
  const radius = 34;
  const circ = 2 * Math.PI * radius;
  const filled = circ * Math.min(1, Math.max(0, value));
  const color = signalColor(value);
  return (
    <svg className="dial" viewBox="0 0 84 84">
      <circle className="dial-track" cx="42" cy="42" r={radius} />
      <circle
        className="dial-arc"
        cx="42"
        cy="42"
        r={radius}
        stroke={color}
        strokeDasharray={`${filled} ${circ}`}
        transform="rotate(-90 42 42)"
      />
      <text className="dial-value" x="42" y="47">
        {value.toFixed(2)}
      </text>
    </svg>
  );
}

// The 5-head XD-PRM breakdown for the selected step — the core explainability readout.
export function HeadScores({ step }: { step: StepEvent | null }) {
  if (!step) {
    return (
      <aside className="telemetry">
        <div className="tele-head">
          <span className="panel-title">Verifier telemetry</span>
        </div>
        <div className="ledger-empty">Select a step to inspect its verifier channels.</div>
      </aside>
    );
  }
  return (
    <aside className="telemetry">
      <div className="tele-head">
        <span className="panel-title">Verifier telemetry</span>
        <span className="panel-meta">Step {step.idx + 1}</span>
      </div>
      <div className="dial-wrap">
        <Dial value={step.r_aggregate} />
        <div className="dial-label">
          <span className="dial-title">Aggregate reward</span>
          <span className="dial-sub">signal strength across heads</span>
        </div>
      </div>
      <div className="channels">
        {Object.entries(HEAD_COLORS).map(([head, color]) => {
          const value = step.per_head[head];
          return (
            <div className="channel" key={head}>
              <span className="ch-name">{head}</span>
              <div className="ch-track">
                <div
                  className="ch-fill"
                  style={{ width: `${(value ?? 0) * 100}%`, background: color }}
                />
              </div>
              <span className="ch-val mono">{value === undefined ? "—" : value.toFixed(2)}</span>
            </div>
          );
        })}
      </div>
      <div className="tele-foot">
        <div>
          <span className="foot-label">Uncertainty</span>
          <span className="foot-val mono">{step.uncertainty.toFixed(2)}</span>
        </div>
        <div>
          <span className="foot-label">Decision</span>
          <span className="foot-val" style={{ color: DECISION_COLORS[step.decision] ?? "#9a8c7e" }}>
            {step.decision}
          </span>
        </div>
        {step.pruned.length > 0 && (
          <div>
            <span className="foot-label">Pruned</span>
            <span className="foot-val mono">{step.pruned.length}</span>
          </div>
        )}
      </div>
    </aside>
  );
}
