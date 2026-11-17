"""Pipeline tests that do not download model weights."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neurovision_x.detectors.base import Detection
from neurovision_x.detectors.yolov8_onnx import letterbox, nms
from neurovision_x.pipeline import VisionPipeline


class FakeDetector:
    def detect(self, image_bgr, confidence=0.25, iou=0.45):
        return [Detection(label="bus", score=0.91, box=(10, 12, 80, 70))]


class PipelineTests(unittest.TestCase):
    def test_letterbox_is_square(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        padded, ratio, pad = letterbox(image, 640)
        self.assertEqual(padded.shape[:2], (640, 640))
        self.assertAlmostEqual(ratio, 640 / 200)
        self.assertGreater(pad[1], pad[0])

    def test_nms_keeps_higher_score(self):
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [40, 40, 60, 60]], dtype=np.float32)
        scores = np.array([0.4, 0.9, 0.8], dtype=np.float32)
        kept = nms(boxes, scores, 0.5)
        self.assertEqual(kept, [1, 2])

    def test_pipeline_draws_detector_output(self):
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        result = VisionPipeline(FakeDetector()).run_image(image)
        self.assertEqual(len(result.detections), 1)
        self.assertEqual(result.rows[0]["label"], "bus")
        self.assertFalse(np.array_equal(result.image_bgr, image))
        self.assertGreater(result.elapsed_ms, 0)


if __name__ == "__main__":
    unittest.main()
