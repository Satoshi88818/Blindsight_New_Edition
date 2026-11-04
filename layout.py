"""Electrode layout, retinotopic mapping and sampling of images at electrode locations."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Layout:
    ex: np.ndarray          # visual-field x of each electrode, normalised to [-1, 1]
    ey: np.ndarray          # visual-field y (rows grow downward)
    ecc: np.ndarray         # eccentricity in degrees
    angle_deg: np.ndarray   # polar angle of the electrode's visual-field position
    map_x: np.ndarray       # cv2.remap sampling maps (pixels of the sample_res image)
    map_y: np.ndarray
    px: np.ndarray          # drawing position of each phosphene in the output (flattened)
    py: np.ndarray
    sample_res: int


def build_layout(cfg, rng):
    """Regular G x G electrode grid -> visual-field positions and output drawing positions."""
    G = cfg.grid_size
    lin = np.linspace(-1, 1, G)
    cx, cy = np.meshgrid(lin, lin)
    if cfg.implant_type == 'retinal':
        ex, ey = cx.copy(), cy.copy()
    else:
        # Inverse of ecc_cort = a*ln(ecc/e0 + 1). A Chebyshev radius keeps the square grid inside
        # the max_ecc disc; the scale `a` cancels once normalised, so it is not a parameter.
        e0 = 0.3
        m = np.maximum(np.abs(cx), np.abs(cy))
        ecc = e0 * ((cfg.max_ecc / e0 + 1.0) ** m - 1.0)
        ang = np.arctan2(cy, cx)
        frac = ecc / cfg.max_ecc
        ang = ang + rng.normal(0, 0.07 * (1 + 0.1 * frac))
        ecc = np.clip(ecc + rng.normal(0, 0.2 * (1 + 0.1 * frac)), 0, cfg.max_ecc)
        ex, ey = ecc * np.cos(ang) / cfg.max_ecc, ecc * np.sin(ang) / cfg.max_ecc

    ex32, ey32 = ex.astype(np.float32), ey.astype(np.float32)
    S = G * 3
    map_x = ((ex32 + 1) / 2 * (S - 1)).astype(np.float32)
    map_y = ((ey32 + 1) / 2 * (S - 1)).astype(np.float32)

    W, sp = G * cfg.spacing, cfg.spacing
    if cfg.layout == 'percept':          # drawn where the phosphene appears in the visual field
        px = (ex32 * 0.5 + 0.5) * (W - sp) + sp / 2
        py = (ey32 * 0.5 + 0.5) * (W - sp) + sp / 2
    else:                                # drawn on the regular electrode grid
        px = (cx * 0.5 + 0.5) * (W - sp) + sp / 2
        py = (cy * 0.5 + 0.5) * (W - sp) + sp / 2

    return Layout(ex=ex32, ey=ey32, ecc=np.hypot(ex, ey) * cfg.max_ecc,
                  angle_deg=np.degrees(np.arctan2(ey, ex)), map_x=map_x, map_y=map_y,
                  px=px.ravel(), py=py.ravel(), sample_res=S)


class Sampler:
    """Samples a square image at every electrode's visual-field location -> (G, G)."""

    def __init__(self, cfg, layout):
        self.G = cfg.grid_size
        self.retinal = cfg.implant_type == 'retinal'
        self.layout = layout

    def __call__(self, img):
        img = img.astype(np.float32)
        if self.retinal:
            return cv2.resize(img, (self.G, self.G), interpolation=cv2.INTER_AREA)
        S = self.layout.sample_res
        small = cv2.resize(img, (S, S), interpolation=cv2.INTER_AREA)
        return cv2.remap(small, self.layout.map_x, self.layout.map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
