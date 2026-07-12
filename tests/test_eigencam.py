"""
Unit tests for src/explainability/eigencam.py

Uses mock objects to avoid loading a real YOLO model.
Covers: output shape, output range, inversion, error handling.
"""

import numpy as np
import torch
import torch.nn as nn
import pytest
from unittest.mock import MagicMock, patch

from src.explainability.eigencam import EigenCAM


# ---------------------------------------------------------------------------
# Minimal model stub
# ---------------------------------------------------------------------------

class _PassthroughLayer(nn.Module):
    """A no-op layer used only to support hook registration."""
    def forward(self, x):
        return x


class _MockYOLO:
    """
    Minimal stub that mirrors the model.model.model[...] structure
    expected by EigenCAM without loading real weights.
    """
    def __init__(self, num_layers: int = 10):
        inner = MagicMock()
        inner.model = [_PassthroughLayer() for _ in range(num_layers)]
        self.model = inner

    def predict(self, source, **kwargs):
        """No-op — activations are injected directly in tests."""
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_cam(h_img: int = 240, w_img: int = 320,
               act_shape: tuple = (1, 32, 20, 20),
               top_k: int = 3) -> tuple:
    """
    Create an EigenCAM instance with a mock model and pre-injected activations.
    Returns (cam, img, mock_handle).
    """
    model = _MockYOLO()
    cam   = EigenCAM(model, target_layer_index=-3, top_k=top_k)
    cam.activations = torch.rand(*act_shape)
    img = np.random.randint(0, 255, (h_img, w_img, 3), dtype=np.uint8)
    return cam, img


def _run(cam: EigenCAM, img: np.ndarray) -> np.ndarray:
    """Run get_eigencam with the hook patched out (activations pre-set)."""
    mock_handle = MagicMock()
    with patch.object(cam.target_layer, 'register_forward_hook', return_value=mock_handle):
        return cam.get_eigencam(img)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEigenCAM:

    def test_output_spatial_dimensions_match_input(self):
        h, w = 240, 320
        cam, img = _build_cam(h, w)
        result = _run(cam, img)
        assert result.shape == (h, w), f"Expected ({h}, {w}), got {result.shape}"

    def test_output_range_is_unit_interval(self):
        cam, img = _build_cam()
        result = _run(cam, img)
        assert result.min() >= 0.0, f"Min value {result.min()} < 0"
        assert result.max() <= 1.0, f"Max value {result.max()} > 1"

    def test_output_is_2d(self):
        cam, img = _build_cam(h_img=100, w_img=150)
        result = _run(cam, img)
        assert result.ndim == 2

    def test_uniform_activations_produce_zero_heatmap(self):
        """
        Uniform activations → uniform pre-inversion heatmap (all 1s after norm)
        → inverted output is all 0s.
        """
        cam, img = _build_cam(act_shape=(1, 4, 5, 5))
        cam.activations = torch.ones(1, 4, 5, 5)
        result = _run(cam, img)
        assert np.allclose(result, 0.0, atol=0.05), \
            f"Expected near-zero output for uniform activations, got max={result.max():.4f}"

    def test_raises_when_no_activations(self):
        """If the hook never fires and activations stay None, raise ValueError."""
        model = _MockYOLO()
        cam   = EigenCAM(model, target_layer_index=-3)
        cam.activations = None
        img   = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="No activations captured"):
            _run(cam, img)

    def test_raises_on_wrong_activation_ndim(self):
        """A 3-D activation tensor should raise ValueError."""
        cam, img = _build_cam()
        cam.activations = torch.rand(32, 20, 20)  # missing batch dim
        with pytest.raises(ValueError, match="Expected 4-D"):
            _run(cam, img)

    def test_different_image_sizes(self):
        """Output shape must match input image regardless of activation spatial size."""
        for h, w in [(128, 128), (480, 640), (64, 96)]:
            cam, img = _build_cam(h_img=h, w_img=w, act_shape=(1, 16, 10, 10))
            result = _run(cam, img)
            assert result.shape == (h, w)

    def test_top_k_1_runs_without_error(self):
        cam, img = _build_cam(top_k=1)
        result = _run(cam, img)
        assert result.shape == (240, 320)
