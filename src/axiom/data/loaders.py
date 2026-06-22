"""HuggingFace datasets → unified Example schema.

Each loader normalises one dataset's fields into the shared vocabulary. Streaming
is preferred so large splits never fully load into RAM; callers can materialise
with list() when needed.
"""

from __future__ import annotations

from collections.abc import Iterator

from omegaconf import DictConfig

from axiom.common.logging import get_logger
from axiom.data.schemas import Example

log = get_logger(__name__)

# HF splits: the "eval" logical split maps to the actual test/validation split name.
_EVAL_SPLIT: dict[str, str] = {
    "gsm8k": "test",
    "commonsense_qa": "validation",
    "arc_challenge": "test",
    "hotpot_qa": "validation",
    "openbookqa": "test",
    "mmlu": "test",
    "strategyqa": "test",
    "aqua_rat": "test",
}

# Datasets whose logical "train" maps to a non-"train" HF split (e.g. MMLU has no train).
_TRAIN_SPLIT: dict[str, str] = {"mmlu": "validation"}


def _hf_split(dataset_name: str, split: str) -> str:
    """Translate logical "eval"/"train" to the dataset-specific HF split name."""
    if split == "eval":
        return _EVAL_SPLIT.get(dataset_name, "test")
    return _TRAIN_SPLIT.get(dataset_name, split)


def _gsm8k_examples(rows, domain: str) -> Iterator[Example]:
    for i, row in enumerate(rows):
        yield Example(
            id=f"gsm8k:{i}",
            dataset="gsm8k",
            domain=domain,
            question=row["question"],
            gold_answer=row["answer"],
            answer_type="numeric",
        )


def _mcq_examples(rows, dataset: str, domain: str, q_field: str, ans_field: str) -> Iterator[Example]:
    for i, row in enumerate(rows):
        choices_raw = row.get("choices", {})
        choices = {
            "label": list(choices_raw.get("label", [])),
            "text": list(choices_raw.get("text", [])),
        } if choices_raw else None
        yield Example(
            id=f"{dataset}:{i}",
            dataset=dataset,
            domain=domain,
            question=row[q_field],
            gold_answer=str(row[ans_field]).strip().upper(),
            answer_type="mcq",
            choices=choices,
        )


def _hotpotqa_examples(rows, domain: str) -> Iterator[Example]:
    for i, row in enumerate(rows):
        yield Example(
            id=f"hotpot_qa:{i}",
            dataset="hotpot_qa",
            domain=domain,
            question=row["question"],
            gold_answer=row["answer"],
            answer_type="span",
        )


_LOADERS: dict[str, tuple] = {
    "gsm8k": ("openai/gsm8k", "main", _gsm8k_examples),
    "commonsense_qa": ("tau/commonsense_qa", None, None),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", None),
    "hotpot_qa": ("hotpotqa/hotpot_qa", "fullwiki", None),
    "openbookqa": ("allenai/openbookqa", "main", None),
    "mmlu": ("cais/mmlu", "all", None),
    "strategyqa": ("ChilleD/StrategyQA", None, None),
    "aqua_rat": ("deepmind/aqua_rat", "raw", None),
}


def _list_choices(raw: list) -> dict[str, list[str]]:
    """List-style choices (MMLU / AQuA-RAT) -> HF {label, text} dict with letter labels."""
    letters = [chr(ord("A") + j) for j in range(len(raw))]
    return {"label": letters, "text": [str(c) for c in raw]}


def load_examples(
    data_cfg: DictConfig, split: str = "train", limit: int | None = None
) -> Iterator[Example]:
    """Stream Example objects for the configured dataset and split.

    Args:
        data_cfg: Hydra data config group with fields: name, hf_path, hf_name,
                  domain, answer_type, and optionally max_examples.
        split: Logical split — "train" or "eval".
        limit: Hard cap on the number of examples returned (None = no cap).
    """
    from datasets import load_dataset  # type: ignore[import]

    name: str = data_cfg.name
    hf_path: str = getattr(data_cfg, "hf_path", _LOADERS.get(name, (name,))[0])
    hf_name: str | None = getattr(data_cfg, "hf_name", None) or None
    domain: str = getattr(data_cfg, "domain", "general")
    answer_type: str = getattr(data_cfg, "answer_type", "numeric")
    actual_split = _hf_split(name, split)

    log.info("loading %s/%s split=%s", hf_path, hf_name or "", actual_split)
    ds = load_dataset(hf_path, hf_name, split=actual_split, streaming=True, trust_remote_code=False)

    count = 0
    for i, row in enumerate(ds):
        if limit is not None and count >= limit:
            break
        ex = _row_to_example(name, i, row, domain, answer_type)
        if ex is not None:
            yield ex
            count += 1


def _row_to_example(
    name: str, idx: int, row: dict, domain: str, answer_type: str
) -> Example | None:
    """Convert one raw HF row to an Example; return None to skip bad rows."""
    try:
        if name == "gsm8k":
            return Example(
                id=f"gsm8k:{idx}",
                dataset="gsm8k",
                domain=domain,
                question=row["question"],
                gold_answer=row["answer"],
                answer_type="numeric",
            )
        if name in ("commonsense_qa", "arc_challenge", "openbookqa"):
            q_field = "question_stem" if name == "openbookqa" else "question"
            ans_field = "answerKey"
            choices_raw = row.get("choices", {})
            choices = {
                "label": list(choices_raw.get("label", [])),
                "text": list(choices_raw.get("text", [])),
            } if choices_raw else None
            return Example(
                id=f"{name}:{idx}",
                dataset=name,
                domain=domain,
                question=row[q_field],
                gold_answer=str(row[ans_field]).strip().upper(),
                answer_type="mcq",
                choices=choices,
            )
        if name == "hotpot_qa":
            return Example(
                id=f"hotpot_qa:{idx}",
                dataset="hotpot_qa",
                domain=domain,
                question=row["question"],
                gold_answer=row["answer"],
                answer_type="span",
            )
        if name == "mmlu":
            choices = _list_choices(row["choices"])  # list[str]; answer is int 0-3
            return Example(
                id=f"mmlu:{idx}",
                dataset="mmlu",
                domain=domain,
                question=row["question"],
                gold_answer=choices["label"][int(row["answer"])],
                answer_type="mcq",
                choices=choices,
            )
        if name == "aqua_rat":
            choices = _list_choices(row["options"])  # ["A)..", "B).."]; correct is a letter
            return Example(
                id=f"aqua_rat:{idx}",
                dataset="aqua_rat",
                domain=domain,
                question=row["question"],
                gold_answer=str(row["correct"]).strip().upper(),
                answer_type="mcq",
                choices=choices,
            )
        if name == "strategyqa":
            return Example(
                id=f"strategyqa:{idx}",
                dataset="strategyqa",
                domain=domain,
                question=row["question"],
                gold_answer="yes" if row["answer"] else "no",  # bool -> yes/no
                answer_type="boolean",
            )
    except (KeyError, TypeError) as exc:
        log.debug("skipping row %d (%s): %s", idx, name, exc)
        return None
    log.warning("unknown dataset %r — yielding raw fields", name)
    return Example(
        id=f"{name}:{idx}",
        dataset=name,
        domain=domain,
        question=str(row.get("question", "")),
        gold_answer=str(row.get("answer", row.get("answerKey", ""))),
        answer_type=answer_type,
    )


def prefetch(cfg: DictConfig) -> None:
    """Warm the HuggingFace dataset cache for the configured data group."""
    log.info("prefetching %s ...", cfg.data.name)
    for split in ("train", "eval"):
        try:
            count = sum(1 for _ in load_examples(cfg.data, split=split))
            log.info("  %s split: %d examples cached", split, count)
        except Exception as exc:  # noqa: BLE001
            log.warning("  %s split not available: %s", split, exc)
