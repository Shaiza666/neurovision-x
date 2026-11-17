# Research demonstration (Phase 7)

**NeuroVision-X is an experimental research prototype, not a clinically validated diagnostic system.**

```powershell
.\.venv\Scripts\python -m streamlit run app_research.py
```

The YOLOv8 detector UI remains `streamlit run app.py`.

## What the dashboard does

- Lists **on-disk** checkpoints under `experiments/`.
- Loads weights with the architecture that matches the file (smoke jobs use `training.smoke.features`, not the full YAML widths).
- Accepts co-registered T1, T1ce, T2, FLAIR NIfTI (optional seg). Missing modalities are errors; zeros are not synthesized.
- Runs **that checkpoint** (sliding window if needed). No mock logits.
- Shows orthogonal MRI, prediction, GT comparison, T=1 entropy from the same softmax, and Grad-CAM on small volumes.
- Reports wall-clock ms and parameter count from the pass that was run.
- Downloads a ZIP of the Phase 6-style figures.

## Smoke checkpoint

If only `experiments/smoke/` exists, the UI can run an **interface test** on a random tensor whose shape matches the smoke patch. That is a real forward pass of the smoke weights. It is **not MRI and not BraTS**. A banner says so.

A BraTS patient demo requires licensed NIfTI **and** a non-smoke `best.pt`.
