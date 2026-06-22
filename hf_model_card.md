---
language:
- en
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
tags:
- reasoning
- reinforcement-learning
- process-reward-model
- chain-of-thought
- small-language-model
- grpo
- qlora
- math-reasoning
- adaptive-depth
- verifier-guided-decoding
pipeline_tag: text-generation
---

# axiom-qwen2.5-1.5b-reasoning

**AXIOM** (Adaptive eXplainable Intelligence for Optimized Micro-Reasoning) trained on
Qwen2.5-1.5B-Instruct via QLoRA SFT on compressed reasoning traces, with the full AXIOM inference
pipeline: adaptive-depth controller + verifier-guided decoding via XD-PRM.

Built for **Samsung ennovateX AX Hackathon 2026, Problem Statement 06** — Enhancing Reasoning in
Small Language Models (SLMs) using Reinforcement Learning.

## Repositories & Resources

| Resource | Link |
|---|---|
| GitHub (AXIOM code) | https://github.com/prabindersinghh/axiom-ax-2026 |
| Model (this) | https://huggingface.co/prabindersinghh/axiom-qwen2.5-1.5b-reasoning |
| Training Dataset | https://huggingface.co/datasets/prabindersinghh/axiom-reasoning-traces |
| Demo Video | https://youtu.be/xdE6rI9mULU?si=v4dzUQCTxAbbht6g |

---

## Model Description

AXIOM is a **verifier-centric** framework that turns Small Language Models into efficient,
self-checking, adaptive reasoners. This model checkpoint pairs Qwen2.5-1.5B-Instruct with:

1. **XD-PRM** — a cross-domain Process Reward Model with five scalar heads per reasoning step:
   - **Logic** — step-level correctness probability from MC rollouts
   - **Commonsense** — factual plausibility via NLI entailment
   - **Consistency** — logical coherence with prior steps (cross-encoder NLI)
   - **Efficiency** — step novelty vs. prior steps (embedding cosine)
   - **Confidence** — rollout answer agreement / variance
   
   Aggregate reward: `R_agg = 1.0·σ(logic) + 1.0·σ(commonsense) + 1.0·σ(consistency) + 0.5·σ(efficiency) + 0.5·σ(confidence)`

2. **Sparse reasoning compression** — traces are pruned and merged under a token budget with an
   answer-preservation check. Achieved **59.5% token reduction** on GSM8K traces.

3. **GRPO reinforcement learning** — composite reward blending correctness, XD-PRM aggregate,
   efficiency, and repetition penalties.

4. **Adaptive-depth + verifier-guided decoding** — XD-PRM confidence head drives a
   `continue / expand / exit` controller, spending tokens only where uncertainty is high.

### Architecture

```
Qwen2.5-1.5B-Instruct (student policy)
       │
       ▼   QLoRA SFT on 38 compressed GSM8K traces
       │
       ▼   GRPO RL (composite reward)
       │        R = 1.0×correct + 0.5×R_agg(XD-PRM) − 0.1×len_ratio − 0.2×rep_penalty
       │
       ▼   Inference
  ┌──────────────────────────────────────────────┐
  │  Adaptive-depth controller (confidence gate)  │
  │  Verifier-guided decoding  (score candidates) │
  │  XD-PRM: 5 heads per <|step|> sentinel        │
  └──────────────────────────────────────────────┘
```

---

## Training Details

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| SFT method | QLoRA (4-bit NF4, double quant, bfloat16 compute) |
| LoRA rank / alpha | r=16, α=32 |
| LoRA target modules | all linear layers |
| SFT training traces | 38 (compressed from 300 OpenR1-Math-220k rows) |
| Max sequence length | 2048 tokens |
| Batch size | 1 (gradient accumulation = 32) |
| Training hardware | Tesla T4 16 GB (Kaggle) |
| GRPO group size | G=4 (T4 constraint) |
| GRPO KL coefficient | 0.04 |
| XD-PRM backbone | `Qwen/Qwen2.5-0.5B-Instruct` + 5 scalar heads |
| XD-PRM G2 gate | AUC > 0.70, ECE < 0.15, head correlation < 0.90 |
| Step sentinel | `<|step|>` |
| Source traces | `open-r1/OpenR1-Math-220k` (Apache 2.0, 300-row sample) |

---

## Intended Uses

- **Math reasoning** (GSM8K-type arithmetic word problems)
- **Verifier-guided inference** with the AXIOM serving pipeline
- **Research** into process reward models, reasoning compression, and RL for SLMs
- **Educational** demonstration of cross-domain PRM + adaptive-depth decoding

---

## How to Use

### Standard inference (Transformers)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "prabindersinghh/axiom-qwen2.5-1.5b-reasoning"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

question = "A store sells apples for $0.80 each. If Maria buys 5 apples, how much does she pay?"

messages = [
    {"role": "system", "content": "You are a helpful math reasoning assistant. Think step by step."},
    {"role": "user", "content": question},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.0, do_sample=False)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

### With AXIOM verifier-guided decoding (full pipeline)

```bash
git clone https://github.com/prabindersinghh/axiom-ax-2026.git axiom && cd axiom
pip install -e .
make serve      # FastAPI on :8000 with XD-PRM scoring
# then open the React console at :5173 for live per-step telemetry
```

### With 4-bit quantization (low VRAM)

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype="bfloat16", bnb_4bit_use_double_quant=True)

model = AutoModelForCausalLM.from_pretrained(
    "prabindersinghh/axiom-qwen2.5-1.5b-reasoning",
    quantization_config=bnb_config,
    device_map="auto",
)
```

---

## Performance

| Benchmark | Baseline (Qwen2.5-1.5B-I, zero-shot) | AXIOM (this model) | Delta |
|---|---|---|---|
| GSM8K | 57.2%* | **62.4%*** | +5.2 pp |
| MMLU | 56.8%* | **61.1%*** | +4.3 pp |
| StrategyQA | 66.1%* | **70.5%*** | +4.4 pp |

*Baseline from Qwen2.5 model card. AXIOM values are architecture-predicted estimates based on
comparable GRPO + PRM results in the literature (Math-Shepherd, DeepSeekMath-RL); full measured
numbers require a complete A100/L4 training run.

**Efficiency:**

| KPI | Target | Measured / Expected |
|---|---|---|
| Token reduction (compression) | ~40% | **59.5%** ✅ measured |
| XD-PRM step ROC-AUC | > 0.70 | ~0.72 expected |
| Confidence calibration (ECE) | < 0.15 | ~0.13 expected |

---

## Limitations

- **Training corpus is small.** SFT was performed on 38 compressed traces. A typical SFT run
  uses thousands to tens-of-thousands of examples. Benchmark numbers are architecture predictions
  rather than measured results from a complete training run.

- **T4 constraints.** Training used `rollouts_k=1` (binary Logic labels), `batch_size=1`, and
  `max_seq_len=2048` due to 16 GB VRAM limits. Full-quality training requires ≥ 24 GB (L4 / A100).

- **Domain bias.** Current traces are arithmetic word problems (GSM8K). Cross-domain (science,
  commonsense, multi-hop) fine-tuning is supported by the architecture but not yet in this release.

- **vLLM not used.** The HFEngine fallback was used on Kaggle T4 (vLLM unstable on CUDA 12.1).
  KV-cache reuse during GRPO sampling is therefore not active in this checkpoint.

---

## Model Card Contact

Prabinder Singh — prabindersinghh@gmail.com  
Thapar Institute of Engineering & Technology, Patiala

---

## Citation

```bibtex
@misc{axiom2026,
  title     = {AXIOM: Adaptive eXplainable Intelligence for Optimized Micro-Reasoning},
  author    = {Prabinder Singh and Anish Grover},
  year      = {2026},
  note      = {Samsung ennovateX AX Hackathon 2026, Problem Statement 06},
  url       = {https://github.com/prabindersinghh/axiom-ax-2026}
}
```

---

## License

The model weights inherit the license of the base model: **Apache 2.0**.
All supporting code is Apache 2.0. The training dataset is CC-BY-4.0.

---

## Related Resources

| Resource | URL |
|---|---|
| AXIOM GitHub | https://github.com/prabindersinghh/axiom-ax-2026 |
| Training Dataset | https://huggingface.co/datasets/prabindersinghh/axiom-reasoning-traces |
| Base Model | https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct |
| XD-PRM Backbone | https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct |
| Compression Embeddings | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 |
| Demo Video | https://youtu.be/xdE6rI9mULU?si=v4dzUQCTxAbbht6g |

---

*Samsung ennovateX AX Hackathon 2026 · Problem Statement 06 ·
Thapar Institute of Engineering & Technology, Patiala*
