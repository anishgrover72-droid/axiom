from __future__ import annotations

import pytest

from axiom.common.errors import ConfigError
from axiom.prm.labeling.confidence import confidence_from_answers
from axiom.prm.labeling.nli_consistency import _class_index


def test_confidence_masks_unmeasurable_step():
    # Unanimous -> 1.0; split 2/3 -> ~0.67; no parseable answers -> None (masked, not 0.0).
    out = confidence_from_answers([["A", "A", "A"], ["A", "A", "B"], [None, None]])
    assert out[0] == 1.0
    assert out[1] == pytest.approx(2 / 3)
    assert out[2] is None


def test_class_index_resolves_and_fails_loud():
    idx = {"contradiction": 0, "neutral": 1, "entailment": 2}
    assert _class_index(idx, "contradiction") == 0
    assert _class_index(idx, "entail") == 2
    with pytest.raises(ConfigError):
        _class_index(idx, "nonexistent")
