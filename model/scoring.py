"""
NASA C-MAPSS scoring function (Saxena et al., 2008).

Asymmetric around zero error: late predictions (predicted RUL > true RUL,
meaning you thought you had more time than you did) are penalized far more
heavily than early predictions. This encodes a real safety preference -- in
aircraft maintenance, being overly cautious is far cheaper than missing a
failure window.

    d = predicted_RUL - true_RUL

    s = sum( exp(d/10)  - 1 )   for d >= 0   (late prediction -- smaller
                                               denominator, steeper penalty)
    s = sum( exp(-d/13) - 1 )   for d <  0   (early prediction -- larger
                                               denominator, gentler penalty)

Lower score is better. This is NOT symmetric like RMSE -- always report both.

Note: an earlier OCR transcription of the original 2008 paper used in this
project had a1/a2 swapped relative to how the metric is actually implemented
and reported across the RUL literature. This version matches the convention
used consistently in current papers (e.g. arXiv:2410.03134, arXiv:2104.05049,
arXiv:2202.10916): denominator 10 on the late side, 13 on the early side.
Verified against multiple independent sources.
"""

import numpy as np


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = y_pred - y_true
    late = d >= 0
    early = ~late

    score = np.zeros_like(d, dtype=np.float64)
    score[late] = np.exp(d[late] / 10.0) - 1.0
    score[early] = np.exp(-d[early] / 13.0) - 1.0

    return float(score.sum())


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Convenience wrapper: returns both metrics used in every C-MAPSS paper."""
    return {
        "rmse": rmse(y_true, y_pred),
        "nasa_score": nasa_score(y_true, y_pred),
    }


if __name__ == "__main__":
    y_true = np.array([50.0, 50.0, 50.0, 50.0])
    y_pred_late = np.array([52.0, 51.0, 52.0, 51.0])
    y_pred_early = np.array([48.0, 49.0, 48.0, 49.0])

    print("Late-biased predictions:", evaluate(y_true, y_pred_late))
    print("Early-biased predictions:", evaluate(y_true, y_pred_early))
    print("(Late should score worse than early for the same |error|)")
