"""Image and video inference pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from neurovision_x.detectors.base import Detection
from neurovision_x.visualize import draw_detections


@dataclass
class FrameResult:
    image_bgr: np.ndarray
    detections: list[Detection]
    elapsed_ms: float

    @property
    def rows(self) -> list[dict[str, float | str | int]]:
        return [detection.as_row() for detection in self.detections]


@dataclass
class VideoResult:
    output_path: Path
    frames_processed: int
    detections: list[Detection] = field(default_factory=list)
    elapsed_ms: float = 0.0
    preview_frames: list[np.ndarray] = field(default_factory=list)


class VisionPipeline:
    def __init__(self, detector) -> None:
        self.detector = detector

    def run_image(
        self,
        image_bgr: np.ndarray,
        confidence: float = 0.25,
        iou: float = 0.45,
    ) -> FrameResult:
        started = time.perf_counter()
        detections = self.detector.detect(image_bgr, confidence=confidence, iou=iou)
        annotated = draw_detections(image_bgr, detections)
        elapsed_ms = (time.perf_counter() - started) * 1000
        return FrameResult(image_bgr=annotated, detections=detections, elapsed_ms=elapsed_ms)

    def run_video(
        self,
        source: Path,
        destination: Path,
        confidence: float = 0.25,
        iou: float = 0.45,
        stride: int = 1,
        max_frames: int | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> VideoResult:
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {source}")

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        destination.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(destination),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps / max(stride, 1),
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"Could not write video: {destination}")

        started = time.perf_counter()
        processed = 0
        index = 0
        collected: list[Detection] = []
        previews: list[np.ndarray] = []
        planned = frame_count
        if max_frames is not None and frame_count:
            planned = min(frame_count, max_frames * max(stride, 1))

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if max_frames is not None and processed >= max_frames:
                    break
                if index % max(stride, 1) == 0:
                    result = self.run_image(frame, confidence=confidence, iou=iou)
                    writer.write(result.image_bgr)
                    collected.extend(result.detections)
                    if len(previews) < 8:
                        previews.append(result.image_bgr)
                    processed += 1
                    if progress is not None:
                        progress(processed, planned or processed)
                index += 1
        finally:
            capture.release()
            writer.release()

        return VideoResult(
            output_path=destination,
            frames_processed=processed,
            detections=collected,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            preview_frames=previews,
        )
