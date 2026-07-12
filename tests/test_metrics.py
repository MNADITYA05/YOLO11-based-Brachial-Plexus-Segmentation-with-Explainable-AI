"""
Unit tests for src/evaluation/metrics.py

Covers: perfect overlap, zero overlap, empty prediction,
        full-image prediction, output key completeness, value ranges.
"""

import numpy as np
import pytest

from src.evaluation.metrics import calculate_metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mask(h: int = 100, w: int = 100, fill: bool = False,
              region: tuple = None) -> np.ndarray:
    """Return a uint8 mask of shape (h, w). Optionally fill all or a region."""
    m = np.zeros((h, w), dtype=np.uint8)
    if fill:
        m[:] = 255
    elif region is not None:
        r1, r2, c1, c2 = region
        m[r1:r2, c1:c2] = 255
    return m


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCalculateMetrics:

    def test_perfect_overlap(self):
        mask = make_mask(region=(20, 80, 20, 80))
        m = calculate_metrics(mask, mask)
        assert m['dice_coeff'] == pytest.approx(1.0, abs=1e-4)
        assert m['iou']        == pytest.approx(1.0, abs=1e-4)
        assert m['accuracy']   == pytest.approx(1.0, abs=1e-4)
        assert m['precision']  == pytest.approx(1.0, abs=1e-4)
        assert m['recall']     == pytest.approx(1.0, abs=1e-4)
        assert m['f1_score']   == pytest.approx(1.0, abs=1e-4)

    def test_zero_overlap(self):
        gt   = make_mask(region=(0,  50, 0,  50))
        pred = make_mask(region=(50, 100, 50, 100))
        m = calculate_metrics(gt, pred)
        assert m['iou']        == pytest.approx(0.0, abs=1e-4)
        assert m['dice_coeff'] == pytest.approx(0.0, abs=1e-4)
        assert m['precision']  == pytest.approx(0.0, abs=1e-4)
        assert m['recall']     == pytest.approx(0.0, abs=1e-4)

    def test_empty_prediction(self):
        """Model predicts nothing — recall should be 0, precision numerically 0."""
        gt   = make_mask(region=(20, 80, 20, 80))
        pred = make_mask()   # all zeros
        m = calculate_metrics(gt, pred)
        assert m['recall']     == pytest.approx(0.0, abs=1e-4)
        assert m['dice_coeff'] == pytest.approx(0.0, abs=1e-4)

    def test_full_image_prediction(self):
        """Model predicts entire image as foreground — recall = 1, precision < 1."""
        gt   = make_mask(region=(20, 80, 20, 80))
        pred = make_mask(fill=True)
        m = calculate_metrics(gt, pred)
        assert m['recall']    == pytest.approx(1.0, abs=1e-4)
        assert m['precision'] <  1.0

    def test_output_keys(self):
        m = calculate_metrics(make_mask(fill=True), make_mask(fill=True))
        assert set(m.keys()) == {'accuracy', 'precision', 'recall', 'f1_score', 'iou', 'dice_coeff'}

    def test_all_values_in_unit_range(self):
        gt   = make_mask(region=(10, 90, 10, 90))
        pred = make_mask(region=(15, 85, 15, 85))
        m = calculate_metrics(gt, pred)
        for key, val in m.items():
            assert 0.0 <= val <= 1.0, f"{key} = {val} is outside [0, 1]"

    def test_partial_overlap_dice_between_0_and_1(self):
        gt   = make_mask(region=(10, 60, 10, 60))
        pred = make_mask(region=(30, 80, 30, 80))
        m = calculate_metrics(gt, pred)
        assert 0.0 < m['dice_coeff'] < 1.0
        assert 0.0 < m['iou']        < 1.0

    def test_f1_equals_dice(self):
        """F1 and Dice are mathematically identical for binary segmentation."""
        gt   = make_mask(region=(10, 70, 10, 70))
        pred = make_mask(region=(20, 80, 20, 80))
        m = calculate_metrics(gt, pred)
        assert m['f1_score'] == pytest.approx(m['dice_coeff'], abs=1e-6)
