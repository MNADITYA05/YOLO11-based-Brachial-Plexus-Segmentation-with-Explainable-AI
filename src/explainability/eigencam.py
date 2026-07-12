"""
EigenCAM — gradient-free explainability for YOLO11 segmentation.

Projects the principal components of feature map activations back into
spatial dimensions to produce a per-prediction attention heatmap.

Reference:
    Muhammad et al., "EigenCAM: Class Activation Map using Principal
    Components", IJCNN 2020.
"""

import logging

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class EigenCAM:
    """
    Gradient-free explainability via eigenvector projection of activations.

    Hooks into a configurable layer of the YOLO11 backbone, decomposes the
    covariance of its activations, and projects via the top-k eigenvectors
    to produce a spatial heatmap. The map is inverted so high-attention
    regions appear bright.

    Args:
        model:               Loaded Ultralytics YOLO model.
        target_layer_index:  Index into model.model.model[...] to hook.
                             Default -3 (3rd-to-last layer, detection head features).
        top_k:               Number of dominant eigenvectors to use.
    """

    def __init__(self, model, target_layer_index: int = -3, top_k: int = 3):
        self.model        = model
        self.activations  = None
        self.top_k        = top_k
        self.target_layer = self.model.model.model[target_layer_index]

    # ------------------------------------------------------------------
    # Internal hook
    # ------------------------------------------------------------------

    def _hook(self, module, input, output):
        """Forward hook that captures layer activations."""
        self.activations = output[0] if isinstance(output, tuple) else output

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_eigencam(self, img: np.ndarray) -> np.ndarray:
        """
        Compute the EigenCAM heatmap for a single image.

        Args:
            img: Input image as an HxWxC uint8 numpy array (BGR).

        Returns:
            Normalised heatmap in [0, 1] with shape (H, W).
            High values indicate regions the model attended to.

        Raises:
            ValueError: If no activations were captured or the activation
                        tensor has an unexpected number of dimensions.
        """
        self.activations = None
        handle = self.target_layer.register_forward_hook(self._hook)

        with torch.no_grad():
            self.model.predict(source=img, save=False, save_txt=False)

        handle.remove()

        if self.activations is None:
            raise ValueError(
                "No activations captured. Check that target_layer_index "
                "refers to a valid layer in model.model.model."
            )

        acts = self.activations.cpu()
        if acts.ndim != 4:
            raise ValueError(f"Expected 4-D activation tensor, got shape {acts.shape}.")

        b, c, h, w = acts.shape
        acts_flat = acts.reshape(b, c, h * w)               # (B, C, H*W)
        cov       = torch.bmm(acts_flat, acts_flat.transpose(1, 2))  # (B, C, C)

        cam = self._project(acts_flat, cov, b, c, h, w)

        # Resize to input image dimensions
        cam = F.interpolate(cam, size=(img.shape[0], img.shape[1]),
                            mode='bilinear', align_corners=False)
        cam = cam.squeeze().numpy()

        # Normalise to [0, 1] and invert
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return 1.0 - cam

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _project(self, acts_flat, cov, b, c, h, w) -> torch.Tensor:
        """
        Project activations onto top-k eigenvectors.

        Falls back to mean-channel activation if eigendecomposition fails.
        """
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)

            # Sort descending by eigenvalue magnitude
            idx          = torch.argsort(eigenvalues, dim=1, descending=True)
            eigenvalues  = torch.gather(eigenvalues,  1, idx)
            eigenvectors = torch.gather(eigenvectors, 2,
                                        idx.unsqueeze(1).expand(-1, c, -1))

            weighted = eigenvectors * eigenvalues.unsqueeze(1)   # (B, C, C)
            k        = min(self.top_k, eigenvalues.shape[1])

            cam = torch.zeros((b, 1, h * w), dtype=torch.float32)
            for i in range(k):
                v   = weighted[:, :, i].unsqueeze(2)             # (B, C, 1)
                cam += torch.abs(torch.bmm(v.transpose(1, 2), acts_flat))

            return cam.reshape(b, 1, h, w)

        except RuntimeError as exc:
            logger.warning(
                f"Eigendecomposition failed ({exc}). "
                "Falling back to mean-channel activation."
            )
            return torch.mean(self.activations.cpu(), dim=1, keepdim=True)
