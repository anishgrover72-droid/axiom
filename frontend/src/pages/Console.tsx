import { FormEvent, useState } from "react";
import { DoneEvent, StepEvent, streamReason } from "../api";
import { DepthTrace } from "../components/DepthTrace";
import { HeadScores } from "../components/HeadScores";
import { ReasoningLedger } from "../components/ReasoningLedger";
import { streamSample } from "../sample";

const DEMO_QUESTION = "Can a vegetarian survive on Mars using current technology?";
const DOMAINS = ["commonsense", "math", "science", "multihop"];
const EXAMPLES: { q: string; d: string }[] = [
  { q: DEMO_QUESTION, d: "commonsense" },
  { q: "If a train covers 60 km in 45 minutes, what is its speed in km/h?", d: "math" },
  { q: "Why does ice float on liquid water?", d: "science" },
];

export function Console() {
  const [question, setQuestion] = useState(DEMO_QUESTION);
  const [domain, setDomain] = useState("commonsense");
  const [steps, setSteps] = useState<StepEvent[]>([]);
  const [done, setDone] = useState<DoneEvent | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [running, setRunning] = useState(false);
  const [sample, setSample] = useState(false);

  async function run(e: FormEvent) {
    e.preventDefault();
    if (running) return;
    setSteps([]);
    setDone(null);
    setSelected(null);
    setSample(false);
    setRunning(true);
    const onStep = (step: StepEvent) => setSteps((prev) => [...prev, step]);
    const onDone = (final: DoneEvent) => setDone(final);
    try {
      await streamReason({ question, domain, answer_type: "span" }, onStep, onDone);
    } catch {
      // Backend offline — fall back to a labeled sample trace so the demo still runs.
      setSample(true);
      await streamSample(onStep, onDone);
    } finally {
      setRunning(false);
    }
  }

  const status = running ? "transmitting" : done ? "signal acquired" : "standby";
  const linkClass = running ? "link live" : done ? "link ok" : "link";

  return (
    <div className="console">
      <div className="console-head">
        <div>
          <span className="band-eyebrow mono">Reasoning console</span>
          <p className="console-intro">
            Pose a question and watch the model reason step by step — each step scored live by the
            XD-PRM verifier.
          </p>
        </div>
        <div className={linkClass}>
          <span className="dot" />
          {status}
        </div>
      </div>

      <form className="command" onSubmit={run}>
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Transmit a question to the reasoner…"
        />
        <select value={domain} onChange={(e) => setDomain(e.target.value)}>
          {DOMAINS.map((d) => (
            <option key={d} value={d}>
              {d}
            </option>
          ))}
        </select>
        <button type="submit" disabled={running}>
          {running ? "Reasoning…" : "Reason"}
        </button>
      </form>

      <div className="examples">
        <span className="examples-label mono">try</span>
        {EXAMPLES.map((ex) => (
          <button
            key={ex.q}
            type="button"
            className="example-chip"
            disabled={running}
            onClick={() => {
              setQuestion(ex.q);
              setDomain(ex.d);
            }}
          >
            {ex.q}
          </button>
        ))}
      </div>

      {sample && (
        <div className="sample-note">
          <span className="dot" />
          Sample reasoning trace — start the FastAPI backend (08_serve) to reason on your own input.
        </div>
      )}

      <DepthTrace steps={steps} />

      <div className="deck">
        <ReasoningLedger steps={steps} selected={selected} onSelect={setSelected} running={running} />
        <HeadScores step={selected === null ? null : steps[selected]} />
      </div>

      {done && (
        <footer className="result">
          <div className="result-answer">
            <span className="eyebrow">Mission result</span>
            <span className="val">{done.answer ?? "—"}</span>
          </div>
          <div className="result-stats">
            <div>
              <span className="eyebrow">steps</span>
              <span className="mono">{done.n_steps}</span>
            </div>
            <div>
              <span className="eyebrow">tokens</span>
              <span className="mono">{done.n_tokens}</span>
            </div>
          </div>
        </footer>
      )}
    </div>
  );
}
