import { useMemo } from "react";
import ReactFlow, { Background, Edge, Node } from "reactflow";
import "reactflow/dist/style.css";
import { StepEvent } from "../api";

// Reasoning rendered as a graph; each node's left bar is colored by its R_aggregate.
function scoreColor(r: number): string {
  return `rgb(${Math.round(210 * (1 - r))}, ${Math.round(80 + 130 * r)}, 90)`;
}

export function ReasoningGraph({
  steps,
  selected,
  onSelect,
}: {
  steps: StepEvent[];
  selected: number | null;
  onSelect: (i: number) => void;
}) {
  const nodes: Node[] = useMemo(
    () =>
      steps.map((s, i) => ({
        id: String(i),
        position: { x: 0, y: i * 92 },
        data: { label: `${i + 1}. ${s.text.slice(0, 64)}${s.text.length > 64 ? "…" : ""}` },
        style: {
          width: 320,
          padding: 8,
          borderRadius: 8,
          fontSize: 12,
          color: "#dbe4ff",
          background: "#10182b",
          border: selected === i ? "2px solid #fff" : "1px solid #2a3550",
          borderLeft: `6px solid ${scoreColor(s.r_aggregate)}`,
        },
      })),
    [steps, selected],
  );

  const edges: Edge[] = useMemo(
    () => steps.slice(1).map((_, i) => ({ id: `e${i}`, source: String(i), target: String(i + 1) })),
    [steps],
  );

  return (
    <div className="graph">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        fitView
        nodesDraggable={false}
        onNodeClick={(_, n) => onSelect(Number(n.id))}
      >
        <Background color="#1c2540" />
      </ReactFlow>
    </div>
  );
}
