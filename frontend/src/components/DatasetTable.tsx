import { useEffect, useRef, useState } from "react";
import { DATASETS } from "../content";

const MAX = Math.max(...DATASETS.map((d) => d.count));
const fmt = (n: number) => (n >= 1000 ? `${(n / 1000).toFixed(n < 10000 ? 1 : 0)}K` : String(n));

// Datasets as a rich table: domain color tags + sqrt-scaled size bars that fill on scroll.
export function DatasetTable() {
  const ref = useRef<HTMLDivElement>(null);
  const [shown, setShown] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setShown(true);
          io.disconnect();
        }
      },
      { threshold: 0.2 },
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);

  return (
    <div className="ds-table" ref={ref}>
      <div className="ds-row ds-head mono">
        <span>Dataset</span>
        <span>Domain</span>
        <span>Answer type</span>
        <span>Examples</span>
      </div>
      {DATASETS.map((d, i) => {
        const pct = (Math.sqrt(d.count) / Math.sqrt(MAX)) * 100;
        return (
          <div className="ds-row" key={d.name}>
            <span className="ds-name mono">{d.name}</span>
            <span className="ds-domain">
              <span className="ds-dot" style={{ background: d.color }} />
              {d.domain}
            </span>
            <span className="ds-task">{d.task}</span>
            <span className="ds-size">
              <span className="ds-bar">
                <span
                  className="ds-bar-fill"
                  style={{
                    width: shown ? `${pct}%` : "0%",
                    background: d.color,
                    transitionDelay: `${i * 0.08}s`,
                  }}
                />
              </span>
              <span className="ds-count mono">{fmt(d.count)}</span>
              <span className="ds-disk mono">{d.size}</span>
            </span>
          </div>
        );
      })}
    </div>
  );
}
