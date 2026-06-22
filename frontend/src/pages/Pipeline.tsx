import { CSSProperties, Fragment } from "react";
import { Icon } from "../components/Icon";
import { Reveal } from "../components/Reveal";
import { PHASES, PIPELINE } from "../content";

const tint = (c: string, pct: number) => `color-mix(in srgb, ${c} ${pct}%, transparent)`;

export function Pipeline() {
  return (
    <div className="page-wrap wide">
      <Reveal>
        <span className="band-eyebrow mono">End-to-end</span>
        <h1 className="page-title">The AXIOM pipeline</h1>
        <p className="page-lead">
          Nine stages, left to right — follow the data as it&apos;s distilled, compressed, trained,
          verified and served.
        </p>
      </Reveal>

      <div className="legend">
        {PHASES.map((p) => (
          <span key={p.key} className="legend-item">
            <span className="legend-dot" style={{ background: p.color }} />
            {p.label}
          </span>
        ))}
      </div>

      <Reveal className="flow-h">
        <div className="flow-track">
          {PIPELINE.map((s, i) => {
            const phase = PHASES.find((p) => p.key === s.phase) ?? PHASES[0];
            const iconStyle: CSSProperties = {
              color: phase.color,
              background: tint(phase.color, 13),
              boxShadow: `inset 0 0 0 1px ${tint(phase.color, 30)}`,
            };
            return (
              <Fragment key={s.id}>
                <article className="hnode" style={{ borderTopColor: phase.color }}>
                  <header className="hnode-head">
                    <span className="hnode-id mono">{s.id}</span>
                    <span className="node-icon" style={iconStyle}>
                      <Icon name={s.icon} />
                    </span>
                  </header>
                  <h3>{s.title}</h3>
                  <p>{s.blurb}</p>
                  <div className="hnode-io">
                    <span className="io-chip">{s.from}</span>
                    <span className="io-arrow">↓</span>
                    <span className="io-chip out">{s.to}</span>
                  </div>
                </article>
                {i < PIPELINE.length - 1 && <span className="flow-arrow">→</span>}
              </Fragment>
            );
          })}
        </div>
      </Reveal>
    </div>
  );
}
