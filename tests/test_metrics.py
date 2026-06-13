from axiom.eval.metrics import (
    accuracy,
    expected_calibration_error,
    pearson,
    roc_auc,
)


def test_roc_auc_extremes_and_ties():
    assert roc_auc([0.1, 0.2, 0.8, 0.9], [False, False, True, True]) == 1.0
    assert roc_auc([0.9, 0.8, 0.2, 0.1], [False, False, True, True]) == 0.0
    assert roc_auc([0.5, 0.5, 0.5, 0.5], [True, False, True, False]) == 0.5


def test_ece_perfect_is_zero():
    assert expected_calibration_error([0.0, 1.0, 1.0], [False, True, True]) == 0.0


def test_pearson():
    assert pearson([1, 2, 3], [2, 4, 6]) == 1.0
    assert pearson([1, 1, 1], [2, 4, 6]) == 0.0  # constant series -> 0


def test_accuracy():
    assert accuracy([True, True, False, False]) == 0.5
