"""
metrics.py — Model evaluation utilities.

Fixes vs v1:
  - compute_ece now correctly operates on the full 2-D probability matrix
    (predictions vs. class indices).  ECE is now in [0, 1].
  - evaluate_model passes the full prob matrix to compute_ece.
"""

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score


# ─────────────────────────────────────────────────────────────────────────────
def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error (ECE).

    Parameters
    ----------
    y_true : (N,) array of true class indices.
    y_prob : (N, C) array of class probabilities  *or*
             (N,)   array of max confidences (legacy; accuracy unavailable).

    Returns
    -------
    float in [0, 1].
    """
    if y_prob.ndim == 2:
        confidence  = np.max(y_prob, axis=1)            # (N,)
        predictions = np.argmax(y_prob, axis=1)          # (N,)
        correct     = (predictions == y_true).astype(float)
    else:
        # Legacy path: only max-confidence scalar per sample; accuracy unknown.
        # Return a simple mean-absolute-deviation as a proxy.
        confidence = y_prob
        correct    = np.zeros_like(confidence)           # can't know accuracy

    n   = len(y_true)
    ece = 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.sum() == 0:
            continue
        avg_conf = confidence[mask].mean()
        avg_acc  = correct[mask].mean()
        ece += mask.sum() * abs(avg_conf - avg_acc)

    return float(ece / max(1, n))


# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(model, device, dataloader, criterion):
    """
    Run one full pass over `dataloader` in eval mode.

    Returns
    -------
    avg_loss, accuracy_pct, f1_macro, precision_macro, recall_macro, ece
    """
    model.eval()
    total_loss  = 0.0
    all_preds   = []
    all_targets = []
    all_probs   = []          # full softmax distributions

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss   = criterion(output, target)
            total_loss += loss.item() * data.size(0)

            probs = torch.softmax(output, dim=1)
            preds = output.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    n           = len(all_targets)
    avg_loss    = total_loss / max(1, n)
    all_preds   = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs   = np.array(all_probs)          # (N, C)

    acc       = float((all_preds == all_targets).mean() * 100.0)
    f1        = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    precision = float(precision_score(all_targets, all_preds, average="macro", zero_division=0))
    recall    = float(recall_score(all_targets, all_preds, average="macro", zero_division=0))
    ece       = compute_ece(all_targets, all_probs)       # pass full (N,C) matrix

    return avg_loss, acc, f1, precision, recall, ece
