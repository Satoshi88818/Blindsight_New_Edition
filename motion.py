"""Simulated head / gaze motion applied to the camera frame."""

import math

import cv2
import numpy as np


def center_square(frame):
    h, w = frame.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return frame[y0:y0 + s, x0:x0 + s]


class GazeModel:
    """Slow head sway + drifting microsaccades + user-triggered saccade offset."""

    def __init__(self, cfg, rng):
        self.cfg, self.rng = cfg, rng
        self.shift = np.zeros(2)       # saccade offset (px), set by the user
        self.micro = np.zeros(2)
        self.next_micro = rng.exponential(1 / cfg.microsaccade_freq) if cfg.microsaccade_freq > 0 else 1e18
        self.prev_total = None
        self.last_shape = (480, 640)

    def saccade(self, magnitude=80):
        self.shift = self.rng.integers(-magnitude, magnitude + 1, 2).astype(float)
        return self.shift

    def reset(self):
        self.shift[:] = 0

    def apply(self, frame, t):
        """Return (shifted frame, delta shift in px, displacement in px since the last frame)."""
        c = self.cfg
        head = np.zeros(2)
        if c.head_movement:
            head[0] = c.head_amplitude * math.sin(2 * math.pi * c.head_freq * t)
            head[1] = c.head_amplitude * math.cos(2 * math.pi * c.head_freq * t * 0.7)
        if t >= self.next_micro:
            self.micro = self.rng.uniform(-10, 10, 2)
            self.next_micro = t + self.rng.exponential(1 / c.microsaccade_freq)
        total = self.shift + head + self.micro
        if self.prev_total is None:
            self.prev_total = total.copy()
        delta = total - self.prev_total
        self.prev_total = total.copy()
        h, w = frame.shape[:2]
        self.last_shape = (h, w)
        M = np.float32([[1, 0, total[0]], [0, 1, total[1]]])
        frame = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return frame, delta, float(np.hypot(*delta))


def motion_blur(out, delta, disp, cfg):
    k = int(cfg.motion_blur_strength * disp / cfg.spacing) | 1
    if k > 1:
        k = min(k, 41)
        out = cv2.blur(out, (k, 1) if abs(delta[0]) > abs(delta[1]) else (1, k))
    return out
