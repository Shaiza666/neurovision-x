"""Draw detection overlays."""

from __future__ import annotations

import cv2
import numpy as np

from neurovision_x.detectors.base import Detection

_PALETTE = (
    (62, 224, 176),
    (255, 176, 72),
    (120, 176, 255),
    (255, 112, 148),
    (186, 154, 255),
    (255, 214, 92),
)


def draw_detections(image_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
    canvas = image_bgr.copy()
    for detection in detections:
        color = _PALETTE[hash(detection.label) % len(_PALETTE)]
        x1, y1, x2, y2 = detection.box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        caption = f"{detection.label} {detection.score:.2f}"
        scale = 0.5
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        top = max(0, y1 - text_h - baseline - 6)
        cv2.rectangle(canvas, (x1, top), (x1 + text_w + 8, top + text_h + baseline + 6), color, -1)
        cv2.putText(
            canvas,
            caption,
            (x1 + 4, top + text_h + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (14, 17, 22),
            thickness,
            cv2.LINE_AA,
        )
    return canvas
