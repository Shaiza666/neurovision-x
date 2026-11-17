# NeuroVision-X

Local computer-vision app for object detection on images and video. Inference runs on this machine with YOLOv8n (ONNX Runtime). The first run downloads the 6 MB weights from OpenCV into `models/yolov8n.onnx`.

## Run the UI

```powershell
cd NeuroVision-X
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m streamlit run app.py
```

Open the local URL Streamlit prints. Use **Image** for a photo (or the built-in street scene) and **Video** for a clip. Confidence, overlap, frame stride, and the frame cap are in the sidebar.

## Command line

```powershell
.\.venv\Scripts\python -m neurovision_x download
.\.venv\Scripts\python -m neurovision_x image photo.jpg -o outputs\photo.jpg
.\.venv\Scripts\python -m neurovision_x video clip.mp4 -o outputs\clip.mp4 --stride 2 --max-frames 120
```

## Tests

```powershell
.\.venv\Scripts\python -m unittest tests.test_pipeline
```

The tests cover letterboxing, non-maximum suppression, and the pipeline with a stand-in detector. They do not download weights.
