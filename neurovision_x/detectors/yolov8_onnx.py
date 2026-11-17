"""YOLOv8n object detector backed by ONNX Runtime."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from neurovision_x.detectors.base import Detection
from neurovision_x.labels import COCO_LABELS

MODEL_URL = "https://dl.opencv.org/models/yolov8/yolov8n.onnx"
MODEL_SHA1 = "136807b88d0b02bc226bdeb9741141d857752e10"
INPUT_SIZE = 640


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_model_path() -> Path:
    return project_root() / "models" / "yolov8n.onnx"


def letterbox(image: np.ndarray, new_shape: int = INPUT_SIZE) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize with unchanged aspect ratio and pad to a square."""
    height, width = image.shape[:2]
    ratio = min(new_shape / height, new_shape / width)
    resized_w = int(round(width * ratio))
    resized_h = int(round(height * ratio))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    pad_w = (new_shape - resized_w) / 2
    pad_h = (new_shape - resized_h) / 2
    top = int(round(pad_h - 0.1))
    bottom = int(round(pad_h + 0.1))
    left = int(round(pad_w - 0.1))
    right = int(round(pad_w + 0.1))
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return padded, ratio, (float(left), float(top))


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy non-maximum suppression. Boxes are xyxy."""
    if len(boxes) == 0:
        return []
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[current, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[current, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[current, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[current, 3], boxes[rest, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_current = (boxes[current, 2] - boxes[current, 0]) * (boxes[current, 3] - boxes[current, 1])
        area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        union = area_current + area_rest - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = rest[iou <= iou_threshold]
    return keep


def ensure_model(path: Path | None = None) -> Path:
    """Download the official OpenCV YOLOv8n ONNX weights if they are missing."""
    destination = path or default_model_path()
    if destination.exists() and _sha1(destination) == MODEL_SHA1:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".onnx.part")
    urllib.request.urlretrieve(MODEL_URL, temporary)
    digest = _sha1(temporary)
    if digest != MODEL_SHA1:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Model checksum mismatch: expected {MODEL_SHA1}, got {digest}")
    temporary.replace(destination)
    return destination


def _sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class YoloV8OnnxDetector:
    def __init__(self, model_path: Path | None = None) -> None:
        import onnxruntime as ort

        resolved = ensure_model(model_path)
        self.session = ort.InferenceSession(str(resolved), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def detect(
        self,
        image_bgr: np.ndarray,
        confidence: float = 0.25,
        iou: float = 0.45,
    ) -> list[Detection]:
        blob, ratio, (pad_x, pad_y) = _preprocess(image_bgr)
        outputs = self.session.run(None, {self.input_name: blob})
        return _decode(outputs[0], image_bgr.shape, ratio, pad_x, pad_y, confidence, iou)


def _preprocess(image_bgr: np.ndarray) -> tuple[np.ndarray, float, tuple[float, float]]:
    padded, ratio, pad = letterbox(image_bgr, INPUT_SIZE)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    blob = rgb.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[None]
    return blob, ratio, pad


def _decode(
    raw: np.ndarray,
    image_shape: tuple[int, ...],
    ratio: float,
    pad_x: float,
    pad_y: float,
    confidence: float,
    iou: float,
) -> list[Detection]:
    prediction = np.squeeze(raw)
    if prediction.ndim != 2:
        return []
    if prediction.shape[0] in (84, 85) or prediction.shape[0] < prediction.shape[1]:
        prediction = prediction.T

    height, width = image_shape[:2]
    boxes: list[list[float]] = []
    scores: list[float] = []
    labels: list[str] = []

    for row in prediction:
        if row.shape[0] == 6:
            x1, y1, x2, y2, score, class_id = row.tolist()
            if score < confidence:
                continue
            class_index = int(class_id)
        else:
            class_scores = row[4:]
            class_index = int(np.argmax(class_scores))
            score = float(class_scores[class_index])
            if score < confidence:
                continue
            cx, cy, bw, bh = row[:4].tolist()
            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2

        x1 = (x1 - pad_x) / ratio
        y1 = (y1 - pad_y) / ratio
        x2 = (x2 - pad_x) / ratio
        y2 = (y2 - pad_y) / ratio
        x1 = float(np.clip(x1, 0, width - 1))
        y1 = float(np.clip(y1, 0, height - 1))
        x2 = float(np.clip(x2, 0, width - 1))
        y2 = float(np.clip(y2, 0, height - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append([x1, y1, x2, y2])
        scores.append(score)
        labels.append(COCO_LABELS[class_index] if 0 <= class_index < len(COCO_LABELS) else str(class_index))

    if not boxes:
        return []

    box_array = np.asarray(boxes, dtype=np.float32)
    score_array = np.asarray(scores, dtype=np.float32)
    kept = nms(box_array, score_array, iou)
    return [
        Detection(
            label=labels[index],
            score=float(score_array[index]),
            box=tuple(int(round(value)) for value in box_array[index]),
        )
        for index in kept
    ]
