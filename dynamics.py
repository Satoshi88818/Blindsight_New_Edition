"""Temporal electrode/percept dynamics: adaptation, persistence, noise, fatigue, failures."""

import numpy as np

from .config import ADAPT_GAIN, FATIGUE_RECOVERY, REF_FPS, per_frame


class TemporalModel:
    def __init__(self, cfg, rng, ecc):
        self.cfg, self.rng = cfg, rng
        G = cfg.grid_size
        self.G = G
        self.adapt = np.zeros((G, G), np.float32)
        self.prev_persist = np.zeros((G, G), np.float32)
        self.failure_mask = np.ones((G, G), np.float32)
        self.fatigue = 0.0
        self.training_progress = 0.0
        # fixed non-responsive electrodes, more likely in the periphery
        frac = np.clip(ecc / cfg.max_ecc, 0, 1)
        p_drop = np.clip(cfg.dropout_prob * (1 + (cfg.peripheral_dropout_scale - 1) * frac), 0, 0.95)
        self.responsive = (rng.random((G, G)) > p_drop).astype(np.float32)

    @property
    def dead_count(self):
        return int((self.failure_mask == 0).sum())

    def apply(self, s, dt):
        """Grid signal s in [0,1] -> effective signal after adaptation/persistence/noise/masks."""
        c = self.cfg
        self.adapt += per_frame(c.adaptation_rate, dt) * (s - self.adapt)
        adapted = s / (1.0 + ADAPT_GAIN * self.adapt) * (1.0 - self.fatigue)
        decay = c.persistence_decay ** (dt * REF_FPS)
        persisted = decay * self.prev_persist + (1 - decay) * adapted
        self.prev_persist = persisted.astype(np.float32)
        noise_scale = 1.0 - 0.3 * self.training_progress if c.training_mode else 1.0
        shape = persisted.shape
        gain = (1 + self.rng.normal(0, c.multiplicative_noise_std, shape)
                + self.rng.normal(0, c.temporal_noise_std * noise_scale, shape))
        noisy = np.clip(persisted * gain, 0, 1.5)
        return (noisy * self.responsive * self.failure_mask).astype(np.float32)

    def update(self, dt, disp):
        """Advance slow state (fatigue, failures, training) once per frame."""
        c = self.cfg
        speed = min(disp / max(dt, 1e-3) / c.spacing, 20.0)       # electrode spacings per second
        gain = (1 - 0.5 * self.training_progress) if c.training_mode else 1.0
        self.fatigue = float(np.clip(
            self.fatigue + dt * (c.fatigue_rate * speed * gain - FATIGUE_RECOVERY * self.fatigue), 0, 0.9))
        if self.rng.random() < 1 - (1 - c.failure_prob) ** dt:
            i, j = self.rng.integers(0, self.G, 2)
            self.failure_mask[i, j] = 0.0
        if c.training_mode:
            self.training_progress = min(1.0, self.training_progress + 0.02 * dt)
