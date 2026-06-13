"""End-to-end smoke on a tiny real model (CPU-friendly). Marked slow: downloads ~1GB.

Run with: pytest -m slow  (after `pip install torch transformers peft`).
Exercises the fallback engine and the full XD-PRM path: dataset -> collate -> 5-head
forward -> masked multi-task loss -> save -> load -> score.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from axiom.common.config import load_config  # noqa: E402
from axiom.common.io import write_records  # noqa: E402
from axiom.common.vllm_pool import GenConfig, get_engine  # noqa: E402
from axiom.data.schemas import Step, StepLabel, Trace  # noqa: E402
from axiom.prm.score import PRMScorer  # noqa: E402
from axiom.prm.train_prm import train_prm  # noqa: E402

TINY = "Qwen/Qwen2.5-0.5B-Instruct"
CPU_PRM = [
    f"prm.backbone.hf_id={TINY}",
    "prm.backbone.load_in_4bit=false",
    "prm.backbone.dtype=float32",
]


@pytest.mark.slow
def test_fallback_engine_generates():
    engine = get_engine(TINY, "float32", 1024)
    out = engine.chat(
        [[{"role": "user", "content": "Reply with exactly one word."}]],
        GenConfig(n=1, max_tokens=8, temperature=0.0),
    )
    assert isinstance(out[0][0], str)


@pytest.mark.slow
def test_xdprm_train_and_score(tmp_path):
    # New per-dataset layout: data/prm/<dataset>/{traces,labels}.jsonl. Several traces so the
    # 80/20 train split is non-empty.
    prm_dir = tmp_path / "data" / "prm" / "smoke"
    prm_dir.mkdir(parents=True)
    traces, labels = [], []
    for i in range(5):
        tid = f"t{i}:0"
        traces.append(
            Trace(
                trace_id=tid,
                example_id=f"t{i}",
                dataset="gsm8k",
                domain="math",
                model="m",
                question="What is 2+2?",
                steps=[Step(idx=0, text="2 + 2 = 4."), Step(idx=1, text="The answer is 4.")],
            )
        )
        labels += [
            StepLabel(
                trace_id=tid,
                step_idx=0,
                domain="math",
                logic=1.0,
                consistency=1.0,
                commonsense=0.7,
                efficiency=0.8,
                confidence=1.0,
            ),
            StepLabel(
                trace_id=tid,
                step_idx=1,
                domain="math",
                logic=1.0,
                consistency=1.0,
                commonsense=0.9,
                efficiency=0.9,
                confidence=1.0,
            ),
        ]
    write_records(prm_dir / "traces.jsonl", traces)
    write_records(prm_dir / "labels.jsonl", labels)

    cfg = load_config(
        [
            *CPU_PRM,
            f"paths.data={tmp_path / 'data'}",
            f"paths.experiments={tmp_path / 'exp'}",
            "prm.train.epochs=1",
            "prm.train.batch_size=1",
            "prm.train.grad_accum=1",
        ]
    )
    ckpt = train_prm(cfg)

    scorer = PRMScorer(cfg.prm, str(ckpt), device="cpu")
    scores = scorer.score(["2 + 2 = 4.", "The answer is 4."], "math")
    assert len(scores.r_aggregate) == 2
    assert set(scores.per_head) == set(cfg.prm.heads.keys())
    assert all(r >= 0.0 for r in scores.r_aggregate)
