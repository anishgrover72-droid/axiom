"""Dependency-free metrics, shared by the XD-PRM gate and the benchmark harness."""

from __future__ import annotations

from collections.abc import Sequence


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def accuracy(correct: Sequence[bool]) -> float:
    return mean([float(c) for c in correct])


def roc_auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """ROC-AUC via the Mann-Whitney U statistic (rank-based, ties averaged)."""
    pairs = sorted(zip(scores, labels, strict=True), key=lambda p: p[0])
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # 1-based, ties share mean rank
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    n_pos = sum(1 for _, y in pairs if y)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    rank_sum_pos = sum(r for r, (_, y) in zip(ranks, pairs, strict=True) if y)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def expected_calibration_error(
    conf: Sequence[float], correct: Sequence[bool], n_bins: int = 10
) -> float:
    """Mean |accuracy - confidence| over equal-width confidence bins."""
    total = len(conf)
    if total == 0:
        return 0.0
    ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, c in enumerate(conf) if (lo < c <= hi) or (b == 0 and c == 0.0)]
        if not idx:
            continue
        bin_conf = mean([conf[i] for i in idx])
        bin_acc = mean([float(correct[i]) for i in idx])
        ece += len(idx) / total * abs(bin_acc - bin_conf)
    return ece


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson correlation; 0.0 if either series is constant."""
    n = len(x)
    if n < 2:
        return 0.0
    mx, my = mean(x), mean(y)
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)
    return cov / (vx * vy) ** 0.5 if vx > 0 and vy > 0 else 0.0
