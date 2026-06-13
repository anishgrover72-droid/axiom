import { useState } from "react";
import { DoneEvent, StepEvent, streamReason } from "./api";
import { DepthTrace } from "./components/DepthTrace";
import { HeadScores } from "./components/HeadScores";
import { ReasoningGraph } from "./components/ReasoningGraph";

const DEMO_QUESTION = "Can a vegetarian survive on Mars using current technology?";

export default function App() {
  const [question, setQuestion] = useState(DEMO_QUESTION);
  const [domain, setDomain] = useState("commonsense");
  const [steps, setSteps] = useState<StepEvent[]>([]);
  const [done, setDone] = useState<DoneEvent | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [running, setRunning] = useState(false);

  async function run() {
    setSteps([]);
    setDone(null);
    setSelected(null);
    setRunning(true);
    try {
      await streamReason(
        { question, domain, answer_type: "span" },
        (step) => setSteps((prev) => [...prev, step]),
        (final) => setDone(final),
      );
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="app">
      <header>
        <h1>AXIOM</h1>
        <span className="tag">Explainable Micro-Reasoning · verifier-guided</span>
      </header>

      <div className="controls">
        <input value={question} onChange={(e) => setQuestion(e.target.value)} />
        <select value={domain} onChange={(e) => setDomain(e.target.value)}>
          {["commonsense", "math", "science", "multihop"].map((d) => (
            <option key={d} value={d}>
              {d}
            </option>
          ))}
        </select>
        <button onClick={run} disabled={running}>
          {running ? "Reasoning…" : "Reason"}
        </button>
      </div>

      <DepthTrace steps={steps} />

      <div className="layout">
        <ReasoningGraph steps={steps} selected={selected} onSelect={setSelected} />
        <HeadScores step={selected === null ? null : steps[selected]} />
      </div>

      {done && (
        <footer>
          <b>Answer:</b> {done.answer ?? "—"} · {done.n_steps} steps · {done.n_tokens} tokens
        </footer>
      )}
    </div>
  );
}
