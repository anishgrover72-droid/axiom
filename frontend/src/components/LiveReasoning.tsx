import { useEffect, useRef, useState } from "react";
import { StepEvent } from "../api";
import { streamSample } from "../sample";
import { HeadScores } from "./HeadScores";
import { ReasoningLedger } from "./ReasoningLedger";

// Auto-playing console preview for the home page: loops a sample trace while in view,
// always keeping the latest step selected so the verifier telemetry animates with it.
export function LiveReasoning() {
  const ref = useRef<HTMLDivElement>(null);
  const [steps, setSteps] = useState<StepEvent[]>([]);
  const [selected, setSelected] = useState<number | null>(null);
  const [active, setActive] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(([entry]) => setActive(entry.isIntersecting), {
      threshold: 0.25,
    });
    io.observe(el);
    return () => io.disconnect();
  }, []);

  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
    (async () => {
      while (!cancelled) {
        setSteps([]);
        setSelected(null);
        await streamSample(
          (step) => {
            if (cancelled) return;
            setSteps((prev) => [...prev, step]);
            setSelected(step.idx);
          },
          () => {},
          { stepMs: 560, doneMs: 0 },
        );
        if (cancelled) break;
        await sleep(3200);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [active]);

  return (
    <div ref={ref} className="deck live-deck">
      <ReasoningLedger
        steps={steps}
        selected={selected}
        onSelect={setSelected}
        running={active && steps.length < 5}
      />
      <HeadScores step={selected === null ? null : steps[selected]} />
    </div>
  );
}
