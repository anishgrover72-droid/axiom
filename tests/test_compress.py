from __future__ import annotations

from axiom.distill.compress import _answer_preserved


def test_numeric_answer_must_survive():
    # The compressed reasoning still ends on the gold number -> preserved.
    assert _answer_preserved(["Add 8 and 10", "So the total is 18"], "18", "numeric")
    # Compression dropped the step that derives the answer -> not preserved.
    assert not _answer_preserved(["Add 8 and 10", "We carry on"], "18", "numeric")


def test_mcq_answer_preserved():
    assert _answer_preserved(["Plants need light", "The answer is B"], "B", "mcq")
    assert not _answer_preserved(["Plants need light", "The answer is A"], "B", "mcq")


def test_missing_gold_or_type_accepts():
    # Nothing to preserve against -> never block a compression.
    assert _answer_preserved(["anything"], None, "numeric")
    assert _answer_preserved(["anything"], "18", None)
