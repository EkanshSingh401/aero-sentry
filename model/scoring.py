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


def nasa_loss_torch(pred, target, clamp_d=200.0):
    """
    Differentiable PyTorch version of the NASA score, usable directly as a
    training loss. Lets a model's point-estimate head be optimized directly
    against the actual deployment metric, instead of a generic proxy like
    MSE or symmetric pinball loss that has no notion of the late/early
    asymmetry.

    d = pred - target (same convention as nasa_score above)

    Stability note -- this went through two rounds of correction during
    smoke-testing:

    1. First issue: this loss is exponential, so an untrained model with
       large early errors can push exp(d/10) or exp(-d/13) toward
       numerical overflow. Original fix was clamping |d| to 50.

    2. Second issue, found by stress-testing that first fix: torch.clamp
       has ZERO gradient outside its bounds. RUL values here range 0-125,
       so a fresh model predicting near 0 against a true value of 120 has
       d=-120 -- a completely realistic early-training error, not a
       pathological one. A clamp of 50 would silently zero the gradient
       for exactly the predictions the model most needs to learn from,
       creating a dead zone that could stall training entirely.

    Fix: clamp_d=200, well beyond any realistic error on this dataset
    (max possible |d| is roughly 125), so it only engages for truly
    pathological outputs, while leaving gradients intact across the entire
    realistic error range.
    """
    import torch
    d = pred - target
    d_clamped = torch.clamp(d, -clamp_d, clamp_d)
    late_loss = torch.exp(d_clamped / 10.0) - 1.0
    early_loss = torch.exp(-d_clamped / 13.0) - 1.0
    loss = torch.where(d_clamped >= 0, late_loss, early_loss)
    return loss.mean()


if __name__ == "__main__":
    y_true = np.array([50.0, 50.0, 50.0, 50.0])
    y_pred_late = np.array([52.0, 51.0, 52.0, 51.0])
    y_pred_early = np.array([48.0, 49.0, 48.0, 49.0])

    print("Late-biased predictions:", evaluate(y_true, y_pred_late))
    print("Early-biased predictions:", evaluate(y_true, y_pred_early))
    print("(Late should score worse than early for the same |error|)")
