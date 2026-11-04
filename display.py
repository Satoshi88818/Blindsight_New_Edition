"""Rendering of the side-by-side view, live window loop and offline GIF export."""

import math
import time

import cv2
import numpy as np


def draw_vf_overlay(img, cfg, frame_shape):
    """Eccentricity rings + meridians on the camera panel, matched to the sampled square."""
    H, W = img.shape[:2]
    h, w = frame_shape
    s = min(h, w)
    bw, bh = s / w * W / 2, s / h * H / 2
    cx, cy = W // 2, H // 2
    cv2.rectangle(img, (int(cx - bw), int(cy - bh)), (int(cx + bw), int(cy + bh)), (0, 255, 255), 1)
    for e in (2, 5, 10, 20):
        if e <= cfg.max_ecc:
            ax = (int(e / cfg.max_ecc * bw), int(e / cfg.max_ecc * bh))
            cv2.ellipse(img, (cx, cy), ax, 0, 0, 360, (255, 255, 255), 1)
            cv2.putText(img, f"{e}d", (cx + ax[0] + 4, cy - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1)
    for ang in range(0, 360, 45):
        a = math.radians(ang)
        r = 1.0 / max(abs(math.cos(a)), abs(math.sin(a)))
        cv2.line(img, (cx, cy), (int(cx + bw * math.cos(a) * r), int(cy + bh * math.sin(a) * r)),
                 (0, 200, 255), 1)


def compose(sim, frame, phos, extra=""):
    """Camera panel (with HUD) + phosphene panel -> BGR uint8 image."""
    c = sim.cfg
    left = cv2.resize(frame, (640, 480))
    if c.show_vf_overlay:
        draw_vf_overlay(left, c, sim.last_frame_shape)
    right = cv2.resize(phos, (480, 480), interpolation=cv2.INTER_AREA)
    right = (right * 255).astype(np.uint8)
    right = cv2.cvtColor(right, cv2.COLOR_RGB2BGR if sim.use_color else cv2.COLOR_GRAY2BGR)
    lines = [f"{c.implant_type} | {sim.extractor} | layout {c.layout} | dropout {c.dropout_prob:.2f}",
             f"noise {c.temporal_noise_std:.2f} | gamma {c.brightness_gamma:.1f} | elong {c.radial_elongation:.1f}x",
             f"{c.threshold_current:g}-{c.saturation_current:g}uA (max {c.max_current:g}) | profile {c.patient_profile}",
             f"fatigue {sim.fatigue:.2f} | training {c.training_mode} | dead {sim.dead_count}",
             extra or "s saccade | r reset | q/w extractor | ESC quit"]
    cv2.rectangle(left, (0, 0), (470, 8 + 18 * len(lines)), (0, 0, 0), -1)
    for i, ln in enumerate(lines):
        cv2.putText(left, ln, (6, 18 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return np.hstack([left, right])


def run_gif(cfg, sim, src):
    """Offline render with a deterministic clock (frame index / fps)."""
    from PIL import Image
    n = max(1, int(cfg.gif_duration * cfg.gif_fps))
    dt = 1.0 / cfg.gif_fps
    frames = []
    print(f"Rendering {n} frames...")
    for k in range(n):
        frame = src.read()
        if frame is None:
            break
        f, ph = sim.process(frame, k * dt, dt)
        img = cv2.cvtColor(compose(sim, f, ph, "offline render"), cv2.COLOR_BGR2RGB)
        if cfg.gif_scale != 1.0:
            img = cv2.resize(img, None, fx=cfg.gif_scale, fy=cfg.gif_scale, interpolation=cv2.INTER_AREA)
        frames.append(Image.fromarray(img))
        if (k + 1) % 20 == 0:
            print(f"  {k + 1}/{n}")
    if not frames:
        raise RuntimeError("No frames captured")
    frames[0].save(cfg.gif_path, save_all=True, append_images=frames[1:],
                   duration=int(1000 / cfg.gif_fps), loop=0, optimize=False)
    print(f"GIF saved: {cfg.gif_path}")


def run_live(cfg, sim, src):
    win = "Blindsight (ESC to quit)"
    try:
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    except cv2.error:
        raise SystemExit("No display available - use --save_gif for headless rendering.")
    start = last = time.time()
    interval = 0.05
    while True:
        frame = src.read()
        if frame is None:
            break
        now = time.time()
        dt = float(np.clip(now - last, 0.01, 0.2))
        last = now
        f, ph = sim.process(frame, now - start, dt)
        cv2.imshow(win, compose(sim, f, ph))
        wait = max(1, int((interval - (time.time() - now)) * 1000))
        key = cv2.waitKey(wait) & 0xFF
        if key == 27 or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord('s'):
            sh = sim.gaze.saccade()
            print(f"Saccade: shift ({sh[0]:+.0f}, {sh[1]:+.0f})")
        elif key == ord('r'):
            sim.gaze.reset()
            print("Shift reset")
        elif key in (ord('q'), ord('w')):
            print(f"Feature extractor: {sim.cycle_extractor(-1 if key == ord('q') else 1)}")
    cv2.destroyAllWindows()
