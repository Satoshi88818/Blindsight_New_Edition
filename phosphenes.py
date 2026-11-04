"""Phosphene model: current -> brightness, per-electrode shapes and Gaussian splatting."""

import math
from functools import lru_cache

import cv2
import numpy as np

NEIGH4 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.float32)
SIGMA_STEP = 1.15          # kernel cache quantisation (15 % steps)
LOG_STEP = math.log(SIGMA_STEP)
ANGLE_BIN_DEG = 10
N_ANGLE_BINS = 180 // ANGLE_BIN_DEG


@lru_cache(maxsize=1024)
def gaussian_kernel(qx, qy, qa):
    """Peak-normalised rotated Gaussian from quantised (sigma_x, sigma_y, angle)."""
    sx, sy = SIGMA_STEP ** qx, SIGMA_STEP ** qy
    r = max(2, int(math.ceil(3 * max(sx, sy))))
    ax = np.arange(-r, r + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    th = math.radians(qa * ANGLE_BIN_DEG)
    c, s = math.cos(th), math.sin(th)
    xr = xx * c + yy * s
    yr = -xx * s + yy * c
    return np.exp(-(xr ** 2 / (2 * sx ** 2) + yr ** 2 / (2 * sy ** 2))).astype(np.float32)


def phosphene_colors(vals):
    """Legacy brightness-coded hues (RGB): grey -> blue -> yellow."""
    v = np.clip(vals, 0, 1)
    col = np.zeros((v.size, 3), np.float32)
    lo, mid, hi = v < 0.3, (v >= 0.3) & (v < 0.7), v >= 0.7
    col[lo] = np.stack([v[lo]] * 3, -1)
    col[mid, 2] = v[mid]
    col[hi, 0] = v[hi]
    col[hi, 1] = v[hi]
    return col


class PhospheneRenderer:
    def __init__(self, cfg, layout, rng):
        self.cfg, self.layout, self.rng = cfg, layout, rng
        self.G = cfg.grid_size
        self.out_size = self.G * cfg.spacing
        self.use_color = bool(cfg.color_enabled or cfg.image_color)
        self._build_static()

    def _build_static(self):
        c, G, rng, L = self.cfg, self.G, self.rng, self.layout
        base = c.base_phosphene_sigma
        sx = base * rng.uniform(0.8, 1.2, (G, G))
        sy = base * rng.uniform(0.8, 1.2, (G, G))
        ang = np.zeros((G, G))
        if c.implant_type == 'cortical' and c.radial_elongation > 1.0:
            sr = np.maximum(sx, sy) * c.radial_elongation
            st = np.minimum(sx, sy)
            sx, sy, ang = sr, st, L.angle_deg.copy()
        shape = (rng.choice([0, 1, 2], size=(G, G), p=[0.5, 0.3, 0.2])
                 if c.varied_shapes else np.zeros((G, G), int))
        u = rng.uniform(-c.phosphene_shape_var, c.phosphene_shape_var, (G, G))
        sx = np.where(shape == 1, sx * (2.0 + u), sx)
        sy = np.where(shape == 2, sy * (1.5 + u), sy)
        ang = ang + np.where(shape == 2, 90.0, 0.0)

        frac = np.clip(L.ecc / c.max_ecc, 0, 1)
        crowd = 1 + (c.crowding_factor - 1) * frac
        self.sx0 = (sx * crowd).ravel()
        self.sy0 = (sy * crowd).ravel()
        self.qa = (np.rint(np.mod(ang, 180) / ANGLE_BIN_DEG).astype(int) % N_ANGLE_BINS).ravel()

        size0 = c.ecc_base_size
        size_max = c.ecc_base_size + c.ecc_linear_factor * c.max_ecc
        size = c.ecc_base_size + c.ecc_linear_factor * L.ecc
        self.size_term = (0.4 * np.clip((size - size0) / max(size_max - size0, 1e-8), 0, 1)).astype(np.float32)

    def brightness(self, s_eff):
        """Effective signal -> perceived brightness via current, threshold and saturation."""
        c = self.cfg
        current = s_eff * c.max_current
        act = np.clip((current - c.threshold_current)
                      / max(c.saturation_current - c.threshold_current, 1e-6), 0, 1)
        return (act ** c.brightness_gamma).astype(np.float32)

    def render(self, b, depth=None, chroma=None):
        """Brightness grid (G,G) -> float32 image (W,W) or (W,W,3) in [0,1]."""
        c, G, rng, L = self.cfg, self.G, self.rng, self.layout
        nb = cv2.filter2D(b, -1, NEIGH4, borderType=cv2.BORDER_CONSTANT)   # sum of 4 neighbours
        dyn = 1 + (c.max_sigma_scale - 1) * np.clip(0.6 * b + self.size_term, 0, 1)
        b_r = b.copy()

        fill = (b > c.filling_in_bright_thresh) & (nb / 4.0 < c.filling_in_dim_thresh)
        dyn = np.where(fill, dyn * c.filling_in_dilation_factor, dyn)
        b_r = np.where(fill, b_r * 1.1, b_r)
        if depth is not None:
            b_r = b_r * (1 + 0.5 * (1 - depth))
            dyn = dyn * (1 + c.crowding_factor * depth)

        # crosstalk on the electrode grid: mostly suppressive, sometimes facilitatory
        coupling = np.where(rng.random((G, G)) < 0.7, -c.crosstalk_factor, c.facilitation_factor)
        nb2 = cv2.filter2D(b_r.astype(np.float32), -1, NEIGH4, borderType=cv2.BORDER_CONSTANT)
        b_r = np.clip(b_r + coupling * nb2, 0, 1.5).astype(np.float32)

        sx, sy = self.sx0 * dyn.ravel(), self.sy0 * dyn.ravel()
        f = np.minimum(1.0, c.max_sigma_px / np.maximum(np.maximum(sx, sy), 1e-6))
        sx, sy = np.maximum(sx * f, 0.8), np.maximum(sy * f, 0.8)
        qx = np.rint(np.log(sx) / LOG_STEP).astype(int)
        qy = np.rint(np.log(sy) / LOG_STEP).astype(int)

        W = self.out_size
        color = self.use_color
        out = np.zeros((W, W, 3) if color else (W, W), np.float32)
        vals = b_r.ravel()
        if color:
            colors = chroma.reshape(-1, 3) * vals[:, None] if chroma is not None else phosphene_colors(vals)
        jit = rng.normal(0, c.position_jitter_std * c.spacing, (2, vals.size))
        for k in np.flatnonzero(vals > 0.02):
            ker = gaussian_kernel(int(qx[k]), int(qy[k]), int(self.qa[k]))
            r = ker.shape[0] // 2
            cx, cy = int(L.px[k] + jit[0, k]), int(L.py[k] + jit[1, k])
            x0, y0 = cx - r, cy - r
            xa, ya = max(x0, 0), max(y0, 0)
            xb, yb = min(x0 + ker.shape[1], W), min(y0 + ker.shape[0], W)
            if xb <= xa or yb <= ya:
                continue
            sub = ker[ya - y0:yb - y0, xa - x0:xb - x0]
            if color:
                out[ya:yb, xa:xb] += sub[..., None] * colors[k]
            else:
                out[ya:yb, xa:xb] += sub * vals[k]

        # soft saturation (exposure curve): dim phosphenes stay visible, overlaps saturate smoothly
        return (1.0 - np.exp(-2.5 * out)).astype(np.float32)
