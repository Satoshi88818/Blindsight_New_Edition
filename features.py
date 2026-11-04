"""Image -> feature maps (edges / brightness), grid-level filtering, chroma and pseudo-depth."""

import cv2
import numpy as np


def extract_features(gray, extractor):
    """uint8 grayscale square -> float32 feature map in [0, 1]."""
    if extractor == 'canny':
        f = cv2.Canny(gray, 50, 150)
        return cv2.GaussianBlur(f, (5, 5), 1).astype(np.float32) / 255.0
    if extractor == 'luma':                      # filled brightness, local-contrast normalised
        eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
        return cv2.GaussianBlur(eq, (0, 0), 1.5).astype(np.float32) / 255.0
    g = gray.astype(np.float32) / 255.0          # float: no uint8 wrap-around
    if extractor == 'dog':
        f = np.clip(cv2.GaussianBlur(g, (5, 5), 1.0) - cv2.GaussianBlur(g, (15, 15), 5.0), 0, None)
        return np.clip(f / max(float(np.percentile(f, 99.5)), 0.03), 0, 1)
    if extractor == 'highpass':
        f = np.abs(cv2.Laplacian(g, cv2.CV_32F))
        return np.clip(f / max(float(np.percentile(f, 99.5)), 0.3), 0, 1)
    raise ValueError(f"unknown extractor: {extractor}")


def center_surround(s, cfg):
    """Difference-of-Gaussians on the electrode grid."""
    center = cv2.GaussianBlur(s, (0, 0), cfg.center_surround_sigma_center)
    surround = cv2.GaussianBlur(s, (0, 0), cfg.center_surround_sigma_surround)
    return np.clip(center - cfg.center_surround_weight * surround, 0, 1)


def chroma(sq_bgr, sampler):
    """Per-electrode RGB hue/saturation (max channel = 1), sampled like the brightness."""
    rgb = np.stack([sampler(sq_bgr[..., i]) for i in (2, 1, 0)], -1) / 255.0
    col = rgb / np.maximum(rgb.max(-1, keepdims=True), 0.05)
    mean = col.mean(-1, keepdims=True)
    return np.clip(mean + 1.3 * (col - mean), 0, 1).astype(np.float32)   # slight saturation boost


class PseudoDepth:
    """Experimental: motion magnitude as a depth proxy (1 = far/static). Not true depth."""

    def __init__(self, sampler):
        self.sampler = sampler
        self.prev = None

    def __call__(self, gray):
        small = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
        depth = np.full((128, 128), 0.5, np.float32)
        if self.prev is not None:
            flow = cv2.calcOpticalFlowFarneback(self.prev, small, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.hypot(flow[..., 0], flow[..., 1])
            depth = cv2.GaussianBlur((1.0 / (1.0 + mag)).astype(np.float32), (5, 5), 2)
        self.prev = small
        return np.clip(self.sampler(depth), 0, 1)
