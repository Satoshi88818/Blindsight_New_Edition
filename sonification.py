"""Optional sonification: mixes the K brightest phosphenes into one short buffer."""

import sys

import numpy as np


class Sonifier:
    RATE, DUR, K, BASE, MAXF = 22050, 0.1, 24, 200.0, 800.0

    def __init__(self, ecc):
        try:
            import pygame
        except ImportError:
            print("pygame not installed - sonification disabled", file=sys.stderr)
            self.ok = False
            return
        self.pg = pygame
        pygame.mixer.init(frequency=self.RATE, size=-16, channels=1, buffer=512)
        self.channel = pygame.mixer.Channel(0)
        self.t = np.linspace(0, self.DUR, int(self.RATE * self.DUR), endpoint=False)
        self.ecc = np.asarray(ecc).ravel()
        self.ok = True

    def play(self, b, depth=None):
        if not self.ok or self.channel.get_busy() or b.mean() < 0.01:
            return
        flat = b.ravel()
        idx = np.argsort(flat)[-self.K:]
        idx = idx[flat[idx] > 0.01]
        if idx.size == 0:
            return
        d = np.full(flat.size, 0.5) if depth is None else depth.ravel()
        freqs = self.BASE + flat[idx] * (self.MAXF - self.BASE) + self.ecc[idx] * 10 + (1 - d[idx]) * 100
        wave = (flat[idx, None] * np.sin(2 * np.pi * freqs[:, None] * self.t[None])).sum(0)
        wave /= max(flat[idx].sum(), 1e-6)
        self.channel.play(self.pg.sndarray.make_sound((wave * 0.5 * 32767).astype(np.int16)))
