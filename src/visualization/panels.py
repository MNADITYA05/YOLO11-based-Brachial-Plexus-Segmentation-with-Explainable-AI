"""
Visualisation utilities for segmentation output panels.

Produces the 4-panel composite image displayed per prediction:
    Original | Ground Truth overlay | Prediction overlay | EigenCAM heatmap
"""

import cv2
import numpy as np


def create_combined_output(
    original_image: np.ndarray,
    ground_truth:   np.ndarray,
    predicted:      np.ndarray,
    eigencam:       np.ndarray,
    metrics:        dict,
    gap_width:      int   = 20,
    metrics_height: int   = 140,
    bottom_space:   int   = 60,
    blend_alpha:    float = 0.4,
) -> np.ndarray:
    """
    Assemble a 4-panel composite image with a metrics header.

    Layout (left to right):
        Original image | Ground truth overlay | Prediction overlay | EigenCAM heatmap

    Args:
        original_image: HxWxC BGR image (uint8).
        ground_truth:   HxW ground truth mask (uint8, non-zero = foreground).
        predicted:      HxW predicted mask    (uint8, non-zero = foreground).
        eigencam:       HxW heatmap in [0, 1] from EigenCAM.get_eigencam().
        metrics:        Dict from calculate_metrics() — Dice, IoU, Acc, Prec, Rec, F1.
        gap_width:      Width (px) of white vertical separator between panels.
        metrics_height: Height (px) of the white header panel above the images.
        bottom_space:   Height (px) of the white footer panel below the images.
        blend_alpha:    EigenCAM heatmap opacity (0 = invisible, 1 = fully opaque).

    Returns:
        Composite image as an HxWxC uint8 numpy array.
    """
    h, w = original_image.shape[:2]
    ground_truth = cv2.resize(ground_truth, (w, h))
    predicted    = cv2.resize(predicted,    (w, h))

    # --- EigenCAM overlay ---
    heatmap          = cv2.applyColorMap((eigencam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap          = cv2.resize(heatmap, (w, h))
    eigencam_overlay = cv2.addWeighted(original_image, 1 - blend_alpha, heatmap, blend_alpha, 0)

    # --- Prediction overlay with contours ---
    pred_overlay = np.zeros_like(original_image)
    pred_overlay[predicted > 0] = original_image[predicted > 0]
    pred_contours, _ = cv2.findContours(
        predicted.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(pred_overlay, pred_contours, -1, (0, 255, 0), 1)

    # --- Ground truth overlay with contours ---
    gt_overlay = np.zeros_like(original_image)
    gt_overlay[ground_truth > 0] = original_image[ground_truth > 0]
    gt_contours, _ = cv2.findContours(
        ground_truth.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(gt_overlay, gt_contours, -1, (0, 255, 0), 1)

    # --- Metrics header ---
    total_width = w * 4 + gap_width * 3
    top_bg      = np.ones((metrics_height, total_width, 3), dtype=np.uint8) * 255
    bottom_bg   = np.ones((bottom_space,   total_width, 3), dtype=np.uint8) * 255
    font        = cv2.FONT_HERSHEY_SIMPLEX

    metric_lines = [
        f'Dice: {metrics["dice_coeff"]:.4f}',
        f'IoU: {metrics["iou"]:.4f}',
        f'Accuracy: {metrics["accuracy"]:.4f}',
        f'Precision: {metrics["precision"]:.4f}',
        f'Recall: {metrics["recall"]:.4f}',
        f'F1 Score: {metrics["f1_score"]:.4f}',
    ]
    for i, line in enumerate(metric_lines):
        cv2.putText(top_bg, line, (10, 25 + i * 20), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    panel_labels = ['Original', 'Ground Truth', 'Prediction', 'EigenCAM']
    label_x = [
        w // 2 - 40,
        w + gap_width + w // 2 - 70,
        2 * w + 2 * gap_width + w // 2 - 50,
        3 * w + 3 * gap_width + w // 2 - 50,
    ]
    for label, x in zip(panel_labels, label_x):
        cv2.putText(top_bg, label, (x, metrics_height - 10), font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    # --- Assemble ---
    separator = np.ones((h, gap_width, 3), dtype=np.uint8) * 255
    image_row = np.hstack([
        original_image, separator,
        gt_overlay,     separator,
        pred_overlay,   separator,
        eigencam_overlay,
    ])
    return np.vstack([top_bg, image_row, bottom_bg])
