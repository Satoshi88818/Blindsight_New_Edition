"""Frame sources: webcam or a static image."""

import cv2


class Source:
    def __init__(self, cfg):
        self.cap = None
        self.static = None
        if cfg.use_webcam:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("Could not open webcam (use --no-webcam --image_path IMG)")
        else:
            self.static = cv2.imread(cfg.image_path)
            if self.static is None:
                raise ValueError(f"Could not load image: {cfg.image_path}")

    def read(self):
        """Return a BGR frame, or None when the source is exhausted."""
        if self.static is not None:
            return self.static.copy()
        ok, frame = self.cap.read()
        return cv2.flip(frame, 1) if ok else None

    def release(self):
        if self.cap is not None:
            self.cap.release()
