"""Command line entry for NeuroVision-X."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from neurovision_x.detectors.yolov8_onnx import YoloV8OnnxDetector, ensure_model
from neurovision_x.pipeline import VisionPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NeuroVision-X on an image or video.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    image_parser = subparsers.add_parser("image", help="Detect objects in one image")
    image_parser.add_argument("source", type=Path)
    image_parser.add_argument("-o", "--output", type=Path, default=Path("outputs/image.jpg"))
    _add_thresholds(image_parser)

    video_parser = subparsers.add_parser("video", help="Detect objects in a video")
    video_parser.add_argument("source", type=Path)
    video_parser.add_argument("-o", "--output", type=Path, default=Path("outputs/video.mp4"))
    video_parser.add_argument("--stride", type=int, default=1)
    video_parser.add_argument("--max-frames", type=int, default=None)
    _add_thresholds(video_parser)

    subparsers.add_parser("download", help="Download the YOLOv8n ONNX weights")
    args = parser.parse_args()

    if args.command == "download":
        path = ensure_model()
        print(path)
        return

    pipeline = VisionPipeline(YoloV8OnnxDetector())
    if args.command == "image":
        image = cv2.imread(str(args.source))
        if image is None:
            raise SystemExit(f"Could not read image: {args.source}")
        result = pipeline.run_image(image, confidence=args.confidence, iou=args.iou)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.output), result.image_bgr)
        print(f"{len(result.detections)} detections in {result.elapsed_ms:.0f} ms -> {args.output}")
        for row in result.rows:
            print(f"  {row['label']} {row['score']:.2f} {row['x1']},{row['y1']} {row['x2']},{row['y2']}")
        return

    result = pipeline.run_video(
        args.source,
        args.output,
        confidence=args.confidence,
        iou=args.iou,
        stride=args.stride,
        max_frames=args.max_frames,
    )
    print(f"{result.frames_processed} frames, {len(result.detections)} boxes -> {result.output_path}")


def _add_thresholds(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)


if __name__ == "__main__":
    main()
