import { DoneEvent, StepEvent } from "./api";

// A canned reasoning trace so the console demonstrates itself when the FastAPI
// backend (08_serve) isn't running. Always shown with a visible "sample" badge.
const SAMPLE_STEPS: StepEvent[] = [
  {
    idx: 0,
    text: "Mars has no breathable atmosphere and no liquid surface water, so any human needs a sealed, pressurized habitat to begin with.",
    per_head: { logic: 0.92, commonsense: 0.89, consistency: 0.9, efficiency: 0.83, confidence: 0.81 },
    r_aggregate: 0.88,
    uncertainty: 0.18,
    decision: "continue",
    surviving: [],
    pruned: [],
  },
  {
    idx: 1,
    text: "Food must be produced on-site or shipped; current mission designs rely on closed-loop hydroponics plus stored supplies.",
    per_head: { logic: 0.85, commonsense: 0.84, consistency: 0.87, efficiency: 0.71, confidence: 0.73 },
    r_aggregate: 0.8,
    uncertainty: 0.27,
    decision: "continue",
    surviving: [],
    pruned: [],
  },
  {
    idx: 2,
    text: "A vegetarian diet maps cleanly onto hydroponic crops — legumes, leafy greens and grains cover calories and most protein.",
    per_head: { logic: 0.81, commonsense: 0.83, consistency: 0.78, efficiency: 0.76, confidence: 0.67 },
    r_aggregate: 0.77,
    uncertainty: 0.41,
    decision: "expand",
    surviving: [],
    pruned: ["A weaker branch arguing meat is strictly required for protein"],
  },
  {
    idx: 3,
    text: "The real gap is vitamin B12: plants don't produce it, so supplementation is required regardless of crop choice.",
    per_head: { logic: 0.88, commonsense: 0.86, consistency: 0.91, efficiency: 0.8, confidence: 0.75 },
    r_aggregate: 0.84,
    uncertainty: 0.2,
    decision: "continue",
    surviving: [],
    pruned: [],
  },
  {
    idx: 4,
    text: "So with a pressurized habitat, hydroponic crops and B12 supplements, a vegetarian could survive within mission constraints.",
    per_head: { logic: 0.9, commonsense: 0.86, consistency: 0.92, efficiency: 0.84, confidence: 0.8 },
    r_aggregate: 0.86,
    uncertainty: 0.15,
    decision: "exit",
    surviving: [],
    pruned: [],
  },
];

const SAMPLE_DONE: DoneEvent = {
  answer: "Yes — with a sealed habitat, hydroponic crops and B12 supplements.",
  n_steps: 5,
  n_tokens: 612,
};

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export async function streamSample(
  onStep: (step: StepEvent) => void,
  onDone: (done: DoneEvent) => void,
  opts: { stepMs?: number; doneMs?: number } = {},
): Promise<void> {
  const stepMs = opts.stepMs ?? 620;
  const doneMs = opts.doneMs ?? 480;
  for (const step of SAMPLE_STEPS) {
    await sleep(stepMs);
    onStep(step);
  }
  await sleep(doneMs);
  onDone(SAMPLE_DONE);
}
