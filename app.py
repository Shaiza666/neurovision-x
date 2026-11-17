"""NeuroVision-X image and video inference UI."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from neurovision_x import __version__
from neurovision_x.detectors.yolov8_onnx import YoloV8OnnxDetector
from neurovision_x.pipeline import VisionPipeline

ROOT = Path(__file__).resolve().parent
SAMPLE_URL = "https://ultralytics.com/images/bus.jpg"
SAMPLE_PATH = ROOT / "assets" / "samples" / "bus.jpg"
OUTPUTS = ROOT / "outputs"

st.set_page_config(page_title="NeuroVision-X", page_icon="NV", layout="wide")
st.markdown(
    """
    <style>
      .nv-kicker { letter-spacing: 0.16em; text-transform: uppercase; color: #8b9bab; font-size: 0.75rem; }
      .nv-title { font-size: 2.4rem; font-weight: 650; margin: 0.15rem 0 0.25rem; }
      .nv-title span { color: #3ee0b0; }
      div[data-testid="stMetric"] { background: #12181f; border: 1px solid #243041; padding: 0.6rem 0.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_pipeline() -> VisionPipeline:
    return VisionPipeline(YoloV8OnnxDetector())


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_upload(payload: bytes) -> np.ndarray | None:
    array = np.frombuffer(payload, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def ensure_sample() -> Path:
    if SAMPLE_PATH.exists():
        return SAMPLE_PATH
    import urllib.request

    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(SAMPLE_URL, SAMPLE_PATH)
    return SAMPLE_PATH


def main() -> None:
    st.markdown('<p class="nv-kicker">Local object detection</p>', unsafe_allow_html=True)
    st.markdown('<p class="nv-title">Neuro<span>Vision</span>-X</p>', unsafe_allow_html=True)
    st.caption(f"YOLOv8n on ONNX Runtime · v{__version__} · images and video stay on this machine")

    with st.sidebar:
        st.header("Detector")
        confidence = st.slider("Confidence", 0.05, 0.9, 0.25, 0.05)
        iou = st.slider("Overlap (IoU)", 0.1, 0.9, 0.45, 0.05)
        stride = st.slider("Video frame stride", 1, 8, 2)
        max_frames = st.number_input("Max video frames", min_value=1, max_value=2000, value=120)
        st.caption("Weights download once to models/yolov8n.onnx.")

    image_tab, video_tab = st.tabs(["Image", "Video"])

    with image_tab:
        uploaded = st.file_uploader("Image", type=["jpg", "jpeg", "png", "bmp", "webp"])
        use_sample = st.button("Use the sample street scene")
        image = None
        if uploaded is not None:
            image = read_upload(uploaded.getvalue())
        elif use_sample:
            image = cv2.imread(str(ensure_sample()))

        if image is not None:
            with st.spinner("Running detector"):
                result = load_pipeline().run_image(image, confidence=confidence, iou=iou)
            left, right = st.columns(2)
            left.image(bgr_to_rgb(image), caption="Input")
            right.image(bgr_to_rgb(result.image_bgr), caption="Detections")
            m1, m2, m3 = st.columns(3)
            m1.metric("Objects", len(result.detections))
            m2.metric("Time", f"{result.elapsed_ms:.0f} ms")
            m3.metric("Size", f"{image.shape[1]}×{image.shape[0]}")
            if result.rows:
                st.dataframe(result.rows, width="stretch")
            else:
                st.info("No objects passed the confidence threshold.")
            ok, encoded = cv2.imencode(".jpg", result.image_bgr)
            if ok:
                st.download_button("Download annotated image", encoded.tobytes(), "neurovision-x.jpg", "image/jpeg")

    with video_tab:
        video = st.file_uploader("Video", type=["mp4", "avi", "mov", "mkv"])
        if video is not None and st.button("Run on video", type="primary"):
            OUTPUTS.mkdir(parents=True, exist_ok=True)
            source = OUTPUTS / f"source-{video.name}"
            destination = OUTPUTS / f"annotated-{Path(video.name).stem}.mp4"
            source.write_bytes(video.getvalue())
            progress = st.progress(0.0, text="Starting")

            def on_progress(done: int, total: int) -> None:
                fraction = 1.0 if total <= 0 else min(done / total, 1.0)
                progress.progress(fraction, text=f"Frame {done} of {total or '?'}")

            result = load_pipeline().run_video(
                source,
                destination,
                confidence=confidence,
                iou=iou,
                stride=int(stride),
                max_frames=int(max_frames),
                progress=on_progress,
            )
            progress.progress(1.0, text="Finished")
            c1, c2 = st.columns(2)
            c1.metric("Frames", result.frames_processed)
            c2.metric("Boxes", len(result.detections))
            if result.preview_frames:
                st.image(
                    [bgr_to_rgb(frame) for frame in result.preview_frames],
                    caption=[f"Preview {index + 1}" for index in range(len(result.preview_frames))],
                )
            st.download_button(
                "Download annotated video",
                destination.read_bytes(),
                destination.name,
                "video/mp4",
            )


if __name__ == "__main__":
    main()
