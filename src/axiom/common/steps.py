"""THE step-segmentation contract.

This is the only place that defines what "a reasoning step" is. PRM training, PRM
scoring, verifier-guided decoding, and the frontend must all agree, so they all call
`segment`. The function is deterministic and idempotent:

    segment(text) == segment("\\n\\n".join(segment(text)))

`STEP_SENTINEL` is inserted at step boundaries by the PRM as a reserved special token;
it reads each step's representation from the hidden state at the sentinel position.
"""

from __future__ import annotations

import re

STEP_SENTINEL = "<|step|>"

# A line that starts a new step: "Step 3:", "3.", "3)", "(3)", or a bullet.
_ENUM = re.compile(r"^\s*(?:step\s*\d+\s*[:.)]?|\d+\s*[.)]|\(\d+\)|[-*•])\s+", re.IGNORECASE)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_BLANKLINE = re.compile(r"\n\s*\n")
_WS = re.compile(r"[ \t]+")


def normalize_text(text: str) -> str:
    """Collapse intra-line whitespace and runs of blank lines; strip edges."""
    lines = [_WS.sub(" ", ln).rstrip() for ln in text.replace("\r\n", "\n").split("\n")]
    return _BLANKLINE.sub("\n\n", "\n".join(lines)).strip()


def _sentence_split(piece: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(piece) if s.strip()]


def _merge_short(pieces: list[str], min_words: int) -> list[str]:
    """Fold fragments shorter than `min_words` into their neighbour, deterministically."""
    out: list[str] = []
    for piece in pieces:
        if out and len(piece.split()) < min_words:
            out[-1] = f"{out[-1]} {piece}".strip()
        else:
            out.append(piece)
    # A short leading fragment has no previous neighbour; fold it forward.
    if len(out) >= 2 and len(out[0].split()) < min_words:
        out[1] = f"{out[0]} {out[1]}".strip()
        out = out[1:]
    return out


def segment(text: str, *, min_words: int = 3, max_words: int = 80) -> list[str]:
    """Split a reasoning trace into step strings (idempotent, tokenizer-independent)."""
    text = normalize_text(text)
    if not text:
        return []

    pieces: list[str] = []
    for block in _BLANKLINE.split(text):
        cur: list[str] = []
        for line in block.split("\n"):
            if cur and _ENUM.match(line):
                pieces.append(" ".join(cur).strip())
                cur = [line]
            else:
                cur.append(line)
        if cur:
            pieces.append(" ".join(cur).strip())

    expanded: list[str] = []
    for piece in pieces:
        if len(piece.split()) > max_words and not _ENUM.match(piece):
            expanded.extend(_sentence_split(piece))
        else:
            expanded.append(piece)

    return [s for s in _merge_short(expanded, min_words) if s]


def render_with_sentinels(steps: list[str], sentinel: str = STEP_SENTINEL) -> str:
    """Join steps for the PRM: each step is suffixed with the boundary sentinel."""
    return "".join(f"{s} {sentinel}\n" for s in steps)
