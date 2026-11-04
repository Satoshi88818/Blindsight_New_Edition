"""Simulator: wires gaze, features, temporal dynamics and phosphene rendering together."""

import cv2
import numpy as np

from .config import EXTRACTORS
from .dynamics import TemporalModel
from .features import PseudoDepth, center_surround, chroma, extract_features
from .layout import Sampler, build_layout
from .motion import GazeModel, center_square, motion_blur
from .phosphenes import PhospheneRenderer
from .sonification import Sonifier


class Simulator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        # NOTE: construction order fixes the RNG stream (keeps runs reproducible per seed)
        self.layout = build_layout(cfg, self.rng)
        self.sampler = Sampler(cfg, self.layout)
        self.renderer = PhospheneRenderer(cfg, self.layout, self.rng)
        self.dynamics = TemporalModel(cfg, self.rng, self.layout.ecc)
        self.gaze = GazeModel(cfg, self.rng)
        self.depth = PseudoDepth(self.sampler) if cfg.pseudo_depth else None
        self.sonifier = Sonifier(self.layout.ecc) if cfg.enable_sonification else None
        self.extractor = cfg.feature_extractor

    # ---- convenience for the UI ----
    @property
    def use_color(self):
        return self.renderer.use_color

    @property
    def fatigue(self):
        return self.dynamics.fatigue

    @property
    def dead_count(self):
        return self.dynamics.dead_count

    @property
    def last_frame_shape(self):
        return self.gaze.last_shape

    def cycle_extractor(self, step):
        self.extractor = EXTRACTORS[(EXTRACTORS.index(self.extractor) + step) % len(EXTRACTORS)]
        return self.extractor

    # ---- main entry point ----
    def process(self, frame_bgr, t, dt):
        """BGR camera frame -> (shifted frame, phosphene image float32 in [0,1])."""
        c = self.cfg
        frame, delta, disp = self.gaze.apply(frame_bgr, t)
        sq = center_square(frame)
        gray = cv2.cvtColor(sq, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (c.grid_size * 10, c.grid_size * 10), interpolation=cv2.INTER_AREA)
        depth = self.depth(gray) if self.depth is not None else None

        s = self.sampler(extract_features(gray, self.extractor))
        if self.extractor != 'dog':                  # DoG already is a centre-surround filter
            s = center_surround(s, c)
        b = self.renderer.brightness(self.dynamics.apply(s, dt))
        if self.sonifier is not None:
            self.sonifier.play(b, depth)
        out = self.renderer.render(b, depth, chroma(sq, self.sampler) if c.image_color else None)
        out = motion_blur(out, delta, disp, c)
        self.dynamics.update(dt, disp)
        return frame, out
