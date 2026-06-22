"""Per-dataset gold extraction and verifiable answer matching.

Every correctness reward and every eval score goes through `grade`. `answer_type`
(`numeric` | `mcq` | `span`) comes from the dataset config.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass

from axiom.common.errors import ConfigError

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_ANSWER_IS = re.compile(r"answer\s*(?:is|:|=)\s*", re.IGNORECASE)
_MCQ_LETTER = re.compile(r"\b([A-E])\b")
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_YESNO = re.compile(r"\b(yes|no|true|false)\b", re.IGNORECASE)


@dataclass(slots=True)
class Verdict:
    """Outcome of grading one model output against gold."""

    correct: bool
    pred: str | None
    gold: str
    f1: float | None = None  # populated for span answers


# --- numeric (GSM8K) -------------------------------------------------------


def _clean_number(s: str) -> str | None:
    m = _NUMBER.search(s.replace(",", ""))
    return m.group(0) if m else None


def _gold_numeric(raw: str) -> str:
    tail = raw.rsplit("####", 1)[-1]  # GSM8K marks the answer after '####'
    num = _clean_number(tail)
    if num is None:
        raise ConfigError(f"no numeric gold in {raw!r}")
    return num


def _pred_numeric(text: str) -> str | None:
    for candidate in (_BOXED.findall(text) or [])[-1:]:
        if (num := _clean_number(candidate)) is not None:
            return num
    if (m := _ANSWER_IS.search(text)) and (num := _clean_number(text[m.end() :])) is not None:
        return num
    nums = _NUMBER.findall(text.replace(",", ""))
    return nums[-1] if nums else None


def _match_numeric(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return pred.strip() == gold.strip()


# --- multiple choice (CommonsenseQA / OpenBookQA / ARC) --------------------


def _choice_labels(choices: dict | None) -> dict[str, str]:
    """Map normalized choice text -> label letter, from HF `choices` dicts."""
    if not choices:
        return {}
    labels, texts = choices.get("label", []), choices.get("text", [])
    return {_squad_norm(t): str(lbl).strip().upper() for lbl, t in zip(labels, texts, strict=False)}


def _pred_mcq(text: str, choices: dict | None) -> str | None:
    valid = [str(lbl).strip().upper() for lbl in (choices or {}).get("label", [])]
    tail = text[m.end() :] if (m := _ANSWER_IS.search(text)) else ""
    # 1) an explicit valid label after "answer is" (covers letters A-E and digits 1-4)
    for lbl in valid:
        if tail and re.search(rf"(?<![A-Za-z0-9]){re.escape(lbl)}(?![A-Za-z0-9])", tail):
            return lbl
    # 2) a bare letter right after "answer is"
    if tail and (lm := _MCQ_LETTER.search(tail)):
        return lm.group(1)
    # 3) the answer stated as choice text
    for norm_text, label in _choice_labels(choices).items():
        if norm_text and norm_text in _squad_norm(text):
            return label
    # 4) last bare letter anywhere
    letters = _MCQ_LETTER.findall(text)
    return letters[-1] if letters else None


# --- span (HotpotQA) -------------------------------------------------------


def _squad_norm(s: str) -> str:
    s = s.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(_ARTICLES.sub(" ", s).split())


def _f1_span(pred: str, gold: str) -> float:
    p, g = _squad_norm(pred).split(), _squad_norm(gold).split()
    if not p or not g:
        return float(p == g)
    overlap = sum((Counter(p) & Counter(g)).values())  # SQuAD-style multiset overlap
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


# --- boolean (StrategyQA) --------------------------------------------------


def _norm_bool(s: str) -> str:
    v = s.strip().lower()
    if v in ("yes", "true", "1"):
        return "yes"
    if v in ("no", "false", "0"):
        return "no"
    return v


def _pred_boolean(text: str) -> str | None:
    tail = text[m.end() :] if (m := _ANSWER_IS.search(text)) else ""
    hits = _YESNO.findall(tail) or _YESNO.findall(text)
    return _norm_bool(hits[-1]) if hits else None


def has_answer(text: str) -> bool:
    """True if the text states a final answer (used for adaptive early-exit)."""
    return bool(_ANSWER_IS.search(text) or _BOXED.search(text))


def extract_prediction(text: str, answer_type: str, choices: dict | None = None) -> str | None:
    """Pull the predicted answer from text without needing a gold (for inference)."""
    if answer_type == "numeric":
        return _pred_numeric(text)
    if answer_type == "mcq":
        return _pred_mcq(text, choices)
    if answer_type == "span":
        return text.strip() or None
    if answer_type == "boolean":
        return _pred_boolean(text)
    raise ConfigError(f"unknown answer_type '{answer_type}'")


def grade(output: str, gold_raw: str, answer_type: str, choices: dict | None = None) -> Verdict:
    """Grade a model output. `gold_raw` is the dataset's raw answer field."""
    if answer_type == "numeric":
        gold, pred = _gold_numeric(gold_raw), _pred_numeric(output)
        return Verdict(_match_numeric(pred, gold), pred, gold)
    if answer_type == "mcq":
        gold = gold_raw.strip().upper()
        pred = _pred_mcq(output, choices)
        return Verdict(pred == gold, pred, gold)
    if answer_type == "span":
        f1 = _f1_span(output, gold_raw)
        return Verdict(_squad_norm(output) == _squad_norm(gold_raw), output.strip(), gold_raw, f1)
    if answer_type == "boolean":
        gold, pred = _norm_bool(gold_raw), _pred_boolean(output)
        return Verdict(pred == gold, pred, gold)
    raise ConfigError(f"unknown answer_type '{answer_type}'")
