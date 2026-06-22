import { CSSProperties } from "react";
import { CountUp } from "../components/CountUp";
import { LiveReasoning } from "../components/LiveReasoning";
import { Reveal } from "../components/Reveal";
import { FEATURES, HEADS, Page, STATS } from "../content";

const TITLE = "AXIOM".split("");

export function Home({ onNavigate }: { onNavigate: (p: Page) => void }) {
  return (
    <div className="home">
      <section className="hero">
        <div className="mars" />
        <div className="hero-inner">
          <span className="landing-eyebrow">Explainable micro-reasoning</span>
          <h1 className="hero-title">
            {TITLE.map((c, i) => (
              <span key={i} style={{ animationDelay: `${0.25 + i * 0.11}s` } as CSSProperties}>
                {c}
              </span>
            ))}
          </h1>
          <p className="landing-sub">
            A verifier-guided reasoning engine. Watch every step transmit — scored, in real time, as
            it thinks.
          </p>
          <div className="hero-cta">
            <button className="enter primary" onClick={() => onNavigate("console")}>
              Launch console <span aria-hidden="true">→</span>
            </button>
            <button className="enter ghost" onClick={() => onNavigate("pipeline")}>
              See the pipeline
            </button>
          </div>
          <div className="landing-tags">
            <span className="tag-pill">XD-PRM cross-domain verifier</span>
            <span className="tag-pill">GRPO reinforcement learning</span>
            <span className="tag-pill">~40% sparse compression</span>
            <span className="tag-pill">Adaptive-depth decoding</span>
          </div>
        </div>
        <div className="scroll-hint mono">scroll to explore ↓</div>
      </section>

      <section className="band stat-band">
        <div className="stats">
          {STATS.map((s) => (
            <Reveal key={s.label} className="stat">
              <span className="stat-value">
                <CountUp
                  value={s.value}
                  prefix={s.prefix}
                  suffix={s.suffix}
                  decimals={s.decimals}
                />
              </span>
              <span className="stat-label mono">{s.label}</span>
            </Reveal>
          ))}
        </div>
      </section>

      <section className="band">
        <Reveal>
          <span className="band-eyebrow mono">Why AXIOM</span>
          <h2 className="band-title">
            Reasoning that <span className="accent">shows its work.</span>
          </h2>
        </Reveal>
        <div className="feature-grid">
          {FEATURES.map((f, i) => (
            <Reveal key={f.title} delay={i * 0.08}>
              <article className="feature-card">
                <span className="feature-idx mono">{String(i + 1).padStart(2, "0")}</span>
                <h3>{f.title}</h3>
                <p>{f.body}</p>
              </article>
            </Reveal>
          ))}
        </div>
      </section>

      <section className="band">
        <Reveal>
          <span className="band-eyebrow mono">The verifier</span>
          <h2 className="band-title">
            Five heads score <span className="accent">every step.</span>
          </h2>
          <p className="band-sub">
            XD-PRM doesn&apos;t just say a step is good — it says why, on five independent axes.
          </p>
        </Reveal>
        <div className="head-grid">
          {HEADS.map((h, i) => (
            <Reveal key={h.name} delay={i * 0.06}>
              <article className="head-card" style={{ borderTopColor: h.color }}>
                <span className="head-dot" style={{ background: h.color }} />
                <h3>{h.name}</h3>
                <p>{h.role}</p>
              </article>
            </Reveal>
          ))}
        </div>
      </section>

      <section className="band">
        <Reveal>
          <span className="band-eyebrow mono">Live</span>
          <h2 className="band-title">
            Watch it <span className="accent">reason.</span>
          </h2>
          <p className="band-sub">
            Every step streams in and is scored as it arrives — this is the real console, running a
            sample trace.
          </p>
        </Reveal>
        <Reveal>
          <LiveReasoning />
        </Reveal>
      </section>

      <section className="band cta-band">
        <Reveal>
          <h2 className="band-title">See it reason, live.</h2>
          <p className="band-sub">Stream the model through your own question in the console.</p>
          <button className="enter primary" onClick={() => onNavigate("console")}>
            Launch console <span aria-hidden="true">→</span>
          </button>
        </Reveal>
      </section>
    </div>
  );
}
