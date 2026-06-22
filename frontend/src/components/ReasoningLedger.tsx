import { StepEvent } from "../api";
import { DECISION_COLORS, signalColor } from "../theme";

const tint = (c: string, pct: number) => `color-mix(in srgb, ${c} ${pct}%, transparent)`;

// The reasoning rendered as a log: each step's left rail glows at its R_aggregate.
function topHead(per_head: Record<string, number>): string | null {
  const entries = Object.entries(per_head);
  if (entries.length === 0) return null;
  return entries.reduce((a, b) => (b[1] > a[1] ? b : a))[0];
}

export function ReasoningLedger({
  steps,
  selected,
  onSelect,
  running,
}: {
  steps: StepEvent[];
  selected: number | null;
  onSelect: (i: number) => void;
  running: boolean;
}) {
  return (
    <section className="ledger">
      <div className="ledger-head">
        <span className="panel-title">Reasoning log</span>
        <span className="panel-meta">{steps.length} steps</span>
      </div>
      <div className="ledger-body">
        {steps.length === 0 && (
          <div className="ledger-empty">
            {running ? "Acquiring signal…" : "Awaiting transmission — enter a question and reason."}
          </div>
        )}
        {steps.map((s, i) => {
          const color = signalColor(s.r_aggregate);
          const top = topHead(s.per_head);
          const dc = DECISION_COLORS[s.decision] ?? "#9a8c7e";
          return (
            <button
              key={i}
              className={`entry${selected === i ? " is-selected" : ""}`}
              onClick={() => onSelect(i)}
            >
              <span className="entry-idx">{String(i + 1).padStart(2, "0")}</span>
              <span
                className="entry-rail"
                style={{ background: color, boxShadow: `0 0 10px -1px ${color}` }}
              />
              <span className="entry-main">
                <span className="entry-text">{s.text}</span>
                <span className="entry-meta">
                  <span
                    className="decision"
                    style={{ color: dc, borderColor: tint(dc, 40), background: tint(dc, 12) }}
                  >
                    {s.decision}
                  </span>
                  {top && <span className="entry-lead">{top} lead</span>}
                  {s.pruned.length > 0 && (
                    <span className="entry-lead">−{s.pruned.length} pruned</span>
                  )}
                </span>
              </span>
              <span className="entry-score" style={{ color }}>
                {s.r_aggregate.toFixed(2)}
              </span>
            </button>
          );
        })}
      </div>
    </section>
  );
}
