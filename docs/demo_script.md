# Project demonstration script

**Audience:** research-portfolio or lab presentation, about 12 minutes.  
**Rule:** say the disclaimer first. Never present smoke Dice as BraTS. Never click through empty uploads as if they were a patient.

**NeuroVision-X is an experimental research prototype, not a clinically validated diagnostic system.**

## Setup (before the talk)

```powershell
cd neurovision-x
.\.venv\Scripts\python -m unittest
.\.venv\Scripts\python -m streamlit run app_research.py
```

Optional second terminal for the preserved detector:

```powershell
.\.venv\Scripts\python -m streamlit run app.py
```

If `experiments/smoke/best.pt` is missing:

```powershell
.\.venv\Scripts\python scripts\smoke_train.py
```

That command trains on **random tensors**. Say so when you show it.

## Minute-by-minute

| Time | Action | What to say |
| --- | --- | --- |
| 0:00 | Title slide / README | Portfolio methods study, not a published paper, not a clinical device. |
| 0:45 | `docs/architecture.md` diagram | Four MRI channels → 3D U-Net → Dice+CE → optional boundary / SE / MC dropout → JSON metrics. YOLO is a separate 2D demo. |
| 2:00 | Hypotheses | H1: ET HD95 with Kervadec boundary. H2: patient entropy vs failed Dice. H3 deferred (no GPU transformer run). **Untested on BraTS.** |
| 3:00 | Empty `data/raw/brats/` | “Official volumes are not in git. `train.py` exits with `MissingBraTSDataError`. That is deliberate.” |
| 3:30 | `docs/experimental_results.md` | Point at **not measured** A/B/C cells. One smoke folder, excluded from the table. |
| 4:30 | Research dashboard | Sidebar: `smoke / best [SMOKE synthetic]`. Architecture `unet3d`, epoch 2. |
| 5:00 | Click **Validate and load uploads** with no files | Error: missing modalities; empty channels are not synthesized. |
| 5:30 | **Interface test: smoke checkpoint + matching synthetic volume** | Random tensor shape `(4,16,16,16)`. Banner: not MRI, not a patient. |
| 6:00 | **Run model inference** | Read the real numbers from the UI (example recorded on this machine: 388.6 ms, 85,542 parameters, CPU). Class histogram is the argmax of **that** `state_dict`. |
| 7:00 | Anatomy | Axial / coronal / sagittal of the tensor. Looks like noise because it is noise. |
| 7:30 | Segmentation | Overlay is the checkpoint argmax. GT cube is a fixture, not a radiologist contour. |
| 8:00 | Uncertainty | T=1 entropy from the same softmax. “Not MC dropout, not a clinical confidence score.” |
| 8:30 | Explainability | Grad-CAM layer name and `score_mode` from the JSON. “Localization, not proof of correctness.” |
| 9:00 | Metrics & export | Download ZIP if needed. Sidecar repeats the disclaimer. |
| 9:30 | Optional YOLO tab | `app.py` on a photo. “Preserved 2D detector; not glioma MRI.” |
| 10:30 | Tests / code | `src/neurovisionx/` modules; `configs/ablations.yaml` A0–A5. |
| 11:30 | Close | What exists: software plus a reduced MSD CPU measurement. What does not: official BraTS Dice. |

## If someone asks for “the accuracy”

> We have not trained on official BraTS. The measured A/B/C numbers are a reduced MSD Task01 CPU run (see the README table). They are not a challenge score. Smoke Dice on random tensors is a software check.

## If someone asks to upload a hospital scan

> This is not a diagnostic system. The demo requires co-registered research NIfTI with all four modalities. We will not interpret a clinical study on this UI.

## After the talk

Stop Streamlit. Do not commit NIfTI or `experiments/smoke/*.pt` (gitignored).
