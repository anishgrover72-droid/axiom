import { CSSProperties } from "react";
import { DatasetTable } from "../components/DatasetTable";
import { Icon } from "../components/Icon";
import { Reveal } from "../components/Reveal";
import { METRICS, STACK } from "../content";

const tint = (c: string, pct: number) => `color-mix(in srgb, ${c} ${pct}%, transparent)`;

export function Tech() {
  return (
    <div className="page-wrap">
      <Reveal>
        <span className="band-eyebrow mono">Under the hood</span>
        <h1 className="page-title">Stack, data &amp; targets</h1>
        <p className="page-lead">
          A pragmatic open-source stack — train in QLoRA, verify with a custom PRM, serve over vLLM.
        </p>
      </Reveal>

      <Reveal>
        <h2 className="section-h">Technology</h2>
      </Reveal>
      <div className="stack-grid">
        {STACK.map((g, i) => {
          const iconStyle: CSSProperties = {
            color: g.color,
            background: tint(g.color, 13),
            boxShadow: `inset 0 0 0 1px ${tint(g.color, 30)}`,
          };
          return (
            <Reveal key={g.label} delay={i * 0.05}>
              <article className="stack-card" style={{ borderTopColor: g.color }}>
                <div className="stack-head">
                  <span className="node-icon" style={iconStyle}>
                    <Icon name={g.icon} />
                  </span>
                  <h3>{g.label}</h3>
                  <span className="stack-count mono">{g.items.length}</span>
                </div>
                <div className="chips">
                  {g.items.map((it) => (
                    <span key={it} className="chip-tech mono">
                      {it}
                    </span>
                  ))}
                </div>
              </article>
            </Reveal>
          );
        })}
      </div>

      <Reveal>
        <h2 className="section-h">Datasets</h2>
      </Reveal>
      <DatasetTable />

      <Reveal>
        <h2 className="section-h">
          Targets <span className="section-note mono">— goals &amp; gates, to be measured</span>
        </h2>
      </Reveal>
      <div className="metric-grid">
        {METRICS.map((m, i) => (
          <Reveal key={m.label} delay={i * 0.05}>
            <article className="metric-card">
              <span className="metric-big">{m.value}</span>
              <span className="metric-label">{m.label}</span>
              <span className="metric-note">{m.note}</span>
            </article>
          </Reveal>
        ))}
      </div>
    </div>
  );
}
