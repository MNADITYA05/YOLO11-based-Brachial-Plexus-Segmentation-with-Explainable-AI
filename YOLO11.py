"""
YOLO11-based Brachial Plexus Segmentation — inference & evaluation pipeline.

Runs YOLO11 instance segmentation over a directory of medical images,
evaluates predictions against ground truth masks using six metrics,
generates per-image EigenCAM XAI visualisations, and produces a
dataset-level normalised confusion matrix and mean metrics summary.

Usage:
    # Default config
    python YOLO11.py

    # Custom config file
    python YOLO11.py --config config/yolo11.yaml

    # Override individual paths without editing the config
    python YOLO11.py --image-dir IMAGES --mask-dir MASKS --output-dir PREDICTIONS

    # Override model weights
    python YOLO11.py --weights runs/train/yolo11_brachial/weights/best.pt
"""

import argparse
import logging
import os

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from src.evaluation.metrics import (
    calculate_metrics,
    generate_confusion_matrix,
    plot_confusion_matrix,
    save_mean_metrics,
)
from src.explainability.eigencam import EigenCAM
from src.visualization.panels import create_combined_output

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s — %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='YOLO11 Brachial Plexus Segmentation + EigenCAM evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--config',     default='config/yolo11.yaml',
                        help='Path to the project YAML config file.')
    parser.add_argument('--image-dir',  default=None,
                        help='Override config: directory of input images.')
    parser.add_argument('--mask-dir',   default=None,
                        help='Override config: directory of ground truth masks.')
    parser.add_argument('--output-dir', default=None,
                        help='Override config: directory for all outputs.')
    parser.add_argument('--weights',    default=None,
                        help='Override config: YOLO model weights path.')
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Image loading  (no mask blending — raw image goes to the model)
# ---------------------------------------------------------------------------

def preprocess_image(image_path: str, mask_path: str) -> tuple:
    """
    Load an image and its corresponding ground truth mask.

    The mask is resized to match the image if dimensions differ.
    The raw image is returned as-is — no annotation is blended onto it
    before inference, ensuring the model cannot observe the ground truth.

    Args:
        image_path: Path to the input image (.png).
        mask_path:  Path to the binary ground truth mask (.png).

    Returns:
        (image, mask) — both as numpy arrays (BGR and grayscale respectively).
    """
    image = cv2.imread(image_path)
    mask  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image.shape[:2] != mask.shape:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    return image, mask


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_images(cfg: dict) -> None:
    image_dir  = cfg['data']['image_dir']
    mask_dir   = cfg['data']['mask_dir']
    output_dir = cfg['data']['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Loading model: {cfg['model']['weights']}")
    model    = YOLO(cfg['model']['weights'])
    eigencam = EigenCAM(
        model,
        target_layer_index=cfg['model']['target_layer_index'],
        top_k=cfg['eigencam']['top_k'],
    )

    all_gt, all_pred, all_metrics = [], [], []

    filenames = sorted(
        f for f in os.listdir(image_dir)
        if f.endswith('.png') and not f.endswith('_mask.png')
    )
    logger.info(f"Found {len(filenames)} images in {image_dir!r}")

    for filename in filenames:
        image_path = os.path.join(image_dir, filename)
        mask_path  = os.path.join(mask_dir, f"{os.path.splitext(filename)[0]}_mask.png")

        if not os.path.exists(mask_path):
            logger.warning(f"Mask not found for {filename} — skipping.")
            continue

        try:
            image, mask = preprocess_image(image_path, mask_path)

            # EigenCAM and inference both receive the raw image only
            eigen_cam = eigencam.get_eigencam(image)
            results   = model(image)

            if results[0].masks is None:
                logger.warning(f"No mask detected for {filename} — skipping.")
                continue

            pred_mask = results[0].masks.data.cpu().numpy()[0]
            pred_mask = cv2.resize(pred_mask, (image.shape[1], image.shape[0]))
            pred_mask = (pred_mask * 255).astype(np.uint8)

            metrics = calculate_metrics(mask, pred_mask)
            all_gt.append(mask)
            all_pred.append(pred_mask)
            all_metrics.append(metrics)

            panel = create_combined_output(
                image, mask, pred_mask, eigen_cam, metrics,
                gap_width=cfg['visualization']['panel_gap_width'],
                metrics_height=cfg['visualization']['metrics_header_height'],
                bottom_space=cfg['visualization']['bottom_space'],
                blend_alpha=cfg['eigencam']['blend_alpha'],
            )
            out_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_combined_output.png")
            cv2.imwrite(out_path, panel)
            logger.info(
                f"Processed {filename} — "
                + ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            )

        except Exception as exc:
            logger.error(f"Error processing {filename}: {exc}")

    # Dataset-level outputs
    if all_gt and all_pred:
        cm = generate_confusion_matrix(all_gt, all_pred)
        plot_confusion_matrix(cm, os.path.join(output_dir, 'confusion_matrix.png'))

    mean_metrics = save_mean_metrics(all_metrics, output_dir)
    if mean_metrics:
        logger.info("Mean metrics summary:")
        for k, v in mean_metrics.items():
            logger.info(f"  {k}: {v:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)

    # CLI arguments override config file values
    if args.image_dir:  cfg['data']['image_dir']  = args.image_dir
    if args.mask_dir:   cfg['data']['mask_dir']   = args.mask_dir
    if args.output_dir: cfg['data']['output_dir'] = args.output_dir
    if args.weights:    cfg['model']['weights']   = args.weights

    process_images(cfg)
    logger.info("Segmentation analysis complete.")
    logger.info("Outputs saved to: %s", cfg['data']['output_dir'])


if __name__ == '__main__':
    main()
