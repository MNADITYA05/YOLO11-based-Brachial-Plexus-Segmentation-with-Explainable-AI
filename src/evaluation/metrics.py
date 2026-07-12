"""
Segmentation evaluation metrics.

Single source of truth for all metric computation, confusion matrix
generation, and results persistence used across the project.
"""

import os
import logging
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)


def calculate_metrics(ground_truth: np.ndarray, prediction: np.ndarray) -> dict:
    """
    Compute pixel-level segmentation metrics against a ground truth mask.

    Args:
        ground_truth: 2-D array; non-zero pixels are foreground.
        prediction:   2-D array; non-zero pixels are foreground.

    Returns:
        Dict with keys: accuracy, precision, recall, f1_score, iou, dice_coeff.
        All values are in [0, 1].
    """
    gt   = ground_truth > 0
    pred = prediction   > 0

    TP = np.sum(pred  &  gt)
    FP = np.sum(pred  & ~gt)
    TN = np.sum(~pred & ~gt)
    FN = np.sum(~pred &  gt)

    accuracy   = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    precision  = TP / (TP + FP + 1e-8)
    recall     = TP / (TP + FN + 1e-8)
    f1_score   = 2 * precision * recall / (precision + recall + 1e-8)
    iou        = TP / (TP + FP + FN + 1e-8)
    dice_coeff = 2 * TP / (2 * TP + FP + FN + 1e-8)

    return {
        'accuracy':   float(accuracy),
        'precision':  float(precision),
        'recall':     float(recall),
        'f1_score':   float(f1_score),
        'iou':        float(iou),
        'dice_coeff': float(dice_coeff),
    }


def generate_confusion_matrix(
    all_ground_truth: list[np.ndarray],
    all_predictions:  list[np.ndarray],
) -> np.ndarray:
    """
    Build a normalised pixel-level confusion matrix across all images.

    Args:
        all_ground_truth: List of 2-D GT mask arrays.
        all_predictions:  List of 2-D predicted mask arrays.

    Returns:
        2x2 numpy array, row-normalised (true-label normalisation).
    """
    gt_flat   = np.concatenate([m.flatten() for m in all_ground_truth]) > 0
    pred_flat = np.concatenate([m.flatten() for m in all_predictions])  > 0
    return confusion_matrix(gt_flat, pred_flat, normalize='true')


def plot_confusion_matrix(cm: np.ndarray, save_path: str) -> None:
    """
    Render and save a confusion matrix heatmap.

    Args:
        cm:        2x2 normalised confusion matrix from generate_confusion_matrix().
        save_path: Full file path for the output PNG.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm * 100,
        annot=True,
        fmt='.1f',
        cmap='Blues',
        xticklabels=['Background', 'Foreground'],
        yticklabels=['Background', 'Foreground'],
    )
    plt.title('Normalised Segmentation Confusion Matrix (%)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"Confusion matrix saved to {save_path}")


def save_mean_metrics(all_metrics: list[dict], output_dir: str) -> dict:
    """
    Average per-image metrics and write them to mean_metrics.txt.

    Args:
        all_metrics: List of metric dicts from calculate_metrics().
        output_dir:  Directory where mean_metrics.txt is written.

    Returns:
        Dict of mean values, or empty dict if all_metrics is empty.
    """
    if not all_metrics:
        logger.warning("No metrics to average — no images were successfully processed.")
        return {}

    sums = defaultdict(float)
    for m in all_metrics:
        for k, v in m.items():
            sums[k] += v

    n = len(all_metrics)
    mean_metrics = {k: v / n for k, v in sums.items()}

    out_path = os.path.join(output_dir, 'mean_metrics.txt')
    with open(out_path, 'w') as f:
        f.write("Mean Metrics Across All Images\n")
        f.write("-" * 30 + "\n")
        f.write(f"Number of images processed: {n}\n")
        f.write("-" * 30 + "\n\n")
        for k, v in mean_metrics.items():
            f.write(f"{k.replace('_', ' ').title()}: {v:.4f}\n")

    logger.info(f"Mean metrics saved to {out_path}")
    return mean_metrics
