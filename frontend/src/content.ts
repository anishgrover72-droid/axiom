// Product content for the marketing/explainer pages. Mirrors the real repo pipeline,
// stack and datasets so the site stays accurate to what AXIOM actually ships.
import { HEAD_COLORS } from "./theme";

export type Page = "home" | "pipeline" | "tech" | "console";

export interface Feature {
  title: string;
  body: string;
}

export const FEATURES: Feature[] = [
  {
    title: "One verifier, every domain",
    body: "XD-PRM scores reasoning steps across math, science, commonsense and multi-hop with a single cross-domain process reward model.",
  },
  {
    title: "Reasoning you can audit",
    body: "Five scalar heads explain why each step is trusted — logic, commonsense, consistency, efficiency and confidence.",
  },
  {
    title: "Think less, not worse",
    body: "Adaptive-depth control and sparse compression cut roughly 40% of tokens while preserving the answer.",
  },
];

export interface Head {
  name: string;
  role: string;
  color: string;
}

export const HEADS: Head[] = [
  { name: "Logic", role: "Is this a valid inference from the previous step?", color: HEAD_COLORS.logic },
  { name: "Commonsense", role: "Is the claim plausible in the real world?", color: HEAD_COLORS.commonsense },
  { name: "Consistency", role: "Does it contradict an earlier step?", color: HEAD_COLORS.consistency },
  { name: "Efficiency", role: "Is the step compact and non-redundant?", color: HEAD_COLORS.efficiency },
  { name: "Confidence", role: "How sure is the model — and is that calibrated?", color: HEAD_COLORS.confidence },
];

export interface Phase {
  key: string;
  label: string;
  color: string;
}

export const PHASES: Phase[] = [
  { key: "data", label: "Data", color: "#5fc9c2" },
  { key: "compress", label: "Compress", color: "#f2c14e" },
  { key: "train", label: "Train", color: "#ff7a45" },
  { key: "deliver", label: "Deliver", color: "#9bd64f" },
];

export interface Stage {
  id: string;
  title: string;
  module: string;
  blurb: string;
  phase: string;
  icon: string;
  from: string;
  to: string;
}

export const PIPELINE: Stage[] = [
  {
    id: "00",
    title: "Ingest & difficulty-tag",
    module: "data/ · 00_download_data",
    blurb: "Load five reasoning datasets, normalize them to one schema and tag difficulty.",
    phase: "data",
    icon: "database",
    from: "5 HF datasets",
    to: "unified, tagged examples",
  },
  {
    id: "01",
    title: "Distill teacher traces",
    module: "distill/ · 01_build_traces",
    blurb: "Generate step-by-step reasoning traces from a DeepSeek teacher for every training question.",
    phase: "data",
    icon: "sparkles",
    from: "questions",
    to: "teacher reasoning traces",
  },
  {
    id: "02",
    title: "Sparse reasoning compression",
    module: "distill/compress · 02_compress",
    blurb: "Prune filler and merge redundant steps, answer preserved — the ~40% token win.",
    phase: "compress",
    icon: "scissors",
    from: "verbose traces",
    to: "sparse traces · ~40% fewer tokens",
  },
  {
    id: "03",
    title: "QLoRA supervised fine-tune",
    module: "sft/ · 03_sft",
    blurb: "4-bit QLoRA SFT teaches the student model the compressed step-by-step reasoning format.",
    phase: "train",
    icon: "sliders",
    from: "sparse traces",
    to: "QLoRA student",
  },
  {
    id: "04",
    title: "PRM label foundry",
    module: "prm/labeling · 04_prm_label",
    blurb: "Monte-Carlo rollouts score every step — logic, confidence and efficiency from one pass.",
    phase: "train",
    icon: "beaker",
    from: "student + questions",
    to: "per-step MC labels",
  },
  {
    id: "05",
    title: "Train XD-PRM",
    module: "prm/ · 05_prm_train",
    blurb: "A cross-domain process reward model: one backbone, five scalar heads verifying every reasoning step.",
    phase: "train",
    icon: "layers",
    from: "step labels",
    to: "XD-PRM · 5 heads",
  },
  {
    id: "06",
    title: "GRPO reinforcement",
    module: "rl/ · 06_grpo",
    blurb: "GRPO with a composite XD-PRM reward closes the loop — reasoning that is correct and compact.",
    phase: "train",
    icon: "repeat",
    from: "student + XD-PRM",
    to: "RL-tuned policy",
  },
  {
    id: "07",
    title: "Evaluate",
    module: "eval/ · 07_eval",
    blurb: "Benchmark step ROC-AUC, best-of-N lift, accuracy and calibration across every domain.",
    phase: "deliver",
    icon: "gauge",
    from: "policy + XD-PRM",
    to: "AUC · best-of-N · ECE",
  },
  {
    id: "08",
    title: "Adaptive-depth serving",
    module: "inference/ · serve/ · 08_serve",
    blurb: "Verifier-guided decoding and an adaptive-depth controller stream live reasoning over FastAPI + SSE.",
    phase: "deliver",
    icon: "broadcast",
    from: "policy + XD-PRM",
    to: "live SSE reasoning",
  },
];

export interface TechGroup {
  label: string;
  items: string[];
  color: string;
  icon: string;
}

export const STACK: TechGroup[] = [
  { label: "Modeling & training", color: "#ff7a45", icon: "sliders", items: ["PyTorch", "HF Transformers", "PEFT · QLoRA", "bitsandbytes 4-bit", "TRL · GRPO", "Accelerate"] },
  { label: "Inference & serving", color: "#5fc9c2", icon: "broadcast", items: ["vLLM", "FastAPI", "Uvicorn", "SSE-Starlette"] },
  { label: "Data & retrieval", color: "#f2c14e", icon: "database", items: ["HF Datasets", "sentence-transformers", "FAISS"] },
  { label: "Config & tracking", color: "#9bd64f", icon: "layers", items: ["Hydra", "OmegaConf", "Pydantic", "Weights & Biases"] },
  { label: "Frontend", color: "#e98aa6", icon: "sparkles", items: ["React", "Vite", "TypeScript"] },
];

export interface Dataset {
  name: string;
  domain: string;
  task: string;
  count: number;
  size: string;
  color: string;
}

export const DATASETS: Dataset[] = [
  { name: "GSM8K", domain: "Math", task: "numeric", count: 8790, size: "~10 MB", color: "#ff8a4c" },
  { name: "ARC-Challenge", domain: "Science", task: "multiple-choice", count: 2590, size: "~3 MB", color: "#5fc9c2" },
  { name: "CommonsenseQA", domain: "Commonsense", task: "multiple-choice", count: 12100, size: "~5 MB", color: "#f2c14e" },
  { name: "OpenBookQA", domain: "Science", task: "multiple-choice", count: 5960, size: "~3 MB", color: "#5fc9c2" },
  { name: "HotpotQA", domain: "Multi-hop", task: "span · EM/F1", count: 113000, size: "~1.2 GB", color: "#9bd64f" },
];

export interface Metric {
  value: string;
  label: string;
  note: string;
}

export interface Stat {
  value: number;
  prefix?: string;
  suffix?: string;
  decimals?: number;
  label: string;
}

export const STATS: Stat[] = [
  { value: 5, label: "verifier heads" },
  { value: 40, prefix: "~", suffix: "%", label: "fewer tokens" },
  { value: 9, label: "pipeline stages" },
  { value: 0.7, decimals: 2, prefix: ">", label: "step AUC target" },
];

// Targets / gates — framed as goals, not measured results.
export const METRICS: Metric[] = [
  { value: "AUC > 0.70", label: "Step verification", note: "XD-PRM separates good vs bad steps" },
  { value: "~40%", label: "Fewer tokens", note: "Sparse compression at matched accuracy" },
  { value: "ECE < 0.15", label: "Calibration", note: "The confidence head stays trustworthy" },
  { value: "Best-of-N", label: "Accuracy lift", note: "Verifier-guided decoding beats plain sampling" },
];
