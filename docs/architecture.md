# System architecture

**Research prototype. Not a clinically validated diagnostic system.**

This diagram describes the **implemented** software. Official BraTS volumes are not in the repository. Training on patients has not been run. The smoke path uses random tensors and is excluded from research tables.

## Pipeline

```mermaid
flowchart TB
  subgraph inputs [Inputs]
    NIfTI["Licensed BraTS NIfTI<br/>T1, T1ce, T2, FLAIR, optional seg"]
    YOLOIMG["Natural images / video<br/>YOLOv8n ONNX"]
  end

  subgraph data [Phase 2 data layer]
    INV["inventory / validate"]
    SPLIT["patient-level splits"]
    PRE["RAS, 1 mm, z-score, patches"]
  end

  subgraph models [Phases 3–4 models]
    UNET["3D U-Net<br/>Çiçek 2016"]
    ATT["Attention U-Net<br/>Oktay 2018"]
    SE["optional SE stem<br/>Hu 2018"]
    LOSS["Dice + CE<br/>optional Kervadec boundary"]
  end

  subgraph train [Phase 3 engine]
    LOOP["AMP, clip, checkpoints<br/>best.pt / latest.pt"]
    SMOKE["smoke_train.py<br/>random tensors only"]
  end

  subgraph eval [Phases 5–6]
    MET["Dice, HD95, P/R"]
    CAL["ECE, NLL, Brier"]
    MC["MC dropout T=10"]
    CAM["3D Grad-CAM, overlays"]
  end

  subgraph ui [Phase 7]
    DEMO["app_research.py<br/>checkpoint forward only"]
    YOLO["app.py YOLOv8n"]
  end

  NIfTI --> INV --> SPLIT --> PRE
  PRE --> UNET
  PRE --> ATT
  SE --> UNET
  UNET --> LOSS
  ATT --> LOSS
  LOSS --> LOOP
  SMOKE --> LOOP
  LOOP --> MET
  LOOP --> CAL
  LOOP --> MC
  LOOP --> CAM
  LOOP --> DEMO
  YOLOIMG --> YOLO
```

Missing BraTS folders stop the patient path with `MissingBraTSDataError`. The UI does not invent logits.

## Package map (`src/neurovisionx/`)

| Package | Role |
| --- | --- |
| `data/` | NIfTI discovery, validation, patient splits, z-score, patches, signed distance |
| `models/` | 3D U-Net, Attention U-Net, optional SE wrapper, factory |
| `losses/` | Soft Dice, inverse-frequency CE, Kervadec boundary |
| `training/` | Engine, checkpoints, smoke loaders, curves |
| `evaluation/` | Sliding-window inference, nested regions, metrics, cost |
| `uncertainty/` | MC dropout, calibration tables |
| `explainability/` | Grad-CAM, orthogonal views, overlays, 3D preview |
| `experiments/` | Ablation catalog, result collection from JSON |
| `demo/` | Checkpoint load, upload validation, dashboard inference |
| `utils/` | YAML config, seeds, run metadata |

The YOLOv8 detector lives in `neurovision_x/` (underscore) and `app.py`. It is a preserved local demo, not a glioma segmenter.

## What is not in the architecture

- UNETR / Swin UNETR (H3 deferred; factory raises `NotImplementedError`)
- Automatic BraTS download
- A clinical PACS / DICOM diagnostic workflow
- Fabricated `metrics_test.json`
