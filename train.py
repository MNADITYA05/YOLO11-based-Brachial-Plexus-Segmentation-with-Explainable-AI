"""
Fine-tune YOLO11 on a brachial plexus segmentation dataset.

The dataset must follow the Ultralytics YOLO segmentation format:

    dataset/
        images/
            train/  *.png
            val/    *.png
        labels/
            train/  *.txt   (polygon coordinates, one file per image)
            val/    *.txt
        data.yaml           (nc, names, train/val paths)

Sample data.yaml:
    path: /absolute/path/to/dataset
    train: images/train
    val:   images/val
    nc: 1
    names: ['brachial_plexus']

Usage:
    # Minimal
    python train.py --data dataset/data.yaml

    # Full options
    python train.py --data dataset/data.yaml --epochs 150 --imgsz 640 --batch 16 --device 0
"""

import argparse
import logging

import yaml
from ultralytics import YOLO

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
        description='Fine-tune YOLO11 for brachial plexus segmentation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--config',  default='config/yolo11.yaml',
                        help='Project config YAML (provides base weights path).')
    parser.add_argument('--data',    required=True,
                        help='Ultralytics dataset YAML (train/val paths + class names).')
    parser.add_argument('--epochs',  type=int,   default=100,
                        help='Number of training epochs.')
    parser.add_argument('--imgsz',   type=int,   default=640,
                        help='Input image size (square).')
    parser.add_argument('--batch',   type=int,   default=16,
                        help='Batch size (-1 for AutoBatch).')
    parser.add_argument('--device',  default='',
                        help='Training device: 0, 0,1, cpu, or empty for auto.')
    parser.add_argument('--workers', type=int,   default=8,
                        help='Number of DataLoader worker processes.')
    parser.add_argument('--project', default='runs/train',
                        help='Root directory for training run outputs.')
    parser.add_argument('--name',    default='yolo11_brachial',
                        help='Name of this training run (subdirectory under --project).')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    weights = cfg['model']['weights']
    logger.info(f"Loading base model from: {weights}")
    model = YOLO(weights)

    logger.info(
        f"Starting training — epochs={args.epochs}, imgsz={args.imgsz}, "
        f"batch={args.batch}, device={args.device or 'auto'}"
    )
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        task='segment',
    )

    best_weights = f"{args.project}/{args.name}/weights/best.pt"
    logger.info(f"Training complete. Best weights saved to: {best_weights}")
    logger.info(f"To run inference with fine-tuned weights:")
    logger.info(f"  python YOLO11.py --weights {best_weights}")


if __name__ == '__main__':
    main()
