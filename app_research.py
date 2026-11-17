"""NeuroVision-X research demonstration (Streamlit).

Experimental prototype — not a clinically validated diagnostic system.
Predictions come only from a loaded checkpoint forward pass.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

from neurovisionx import __version__
from neurovisionx.demo.checkpoints import list_demo_checkpoints, load_demo_model
from neurovisionx.demo.pipeline import axis_length, bundle_zip, run_inference, slice_pack
from neurovisionx.demo.uploads import (
    UploadValidationError,
    load_validated_case,
    validate_modality_paths,
    write_named_bytes,
)
from neurovisionx.explainability.overlays import mri_plane
from neurovisionx.explainability.views import ORIENTATION_ORDER

DISCLAIMER = (
    "NeuroVision-X is an experimental research prototype. "
    "It is not a clinically validated diagnostic system. "
    "Do not use these outputs for diagnosis or treatment decisions."
)

st.set_page_config(
    page_title="NeuroVision-X Research Demo",
    page_icon="NV",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.1rem; max-width: 1400px; }
      .nv-banner {
        background: #3b1d12; border: 1px solid #c45c26; color: #f3d3c2;
        padding: 0.7rem 1rem; border-radius: 6px; font-size: 0.92rem;
      }
      .nv-kicker { letter-spacing: 0.18em; text-transform: uppercase;
        color: #8b9bab; font-size: 0.72rem; margin-bottom: 0; }
      .nv-title { font-size: 2.05rem; font-weight: 650; margin: 0.1rem 0 0.35rem; }
      .nv-title span { color: #3ee0b0; }
      div[data-testid="stMetric"] {
        background: #12181f; border: 1px solid #243041; padding: 0.55rem 0.75rem;
        border-radius: 6px;
      }
      .stTabs [data-baseweb="tab-list"] { gap: 0.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _to_image(plane: np.ndarray) -> np.ndarray:
    array = np.asarray(plane)
    if array.ndim == 2:
        lo, hi = float(np.nanmin(array)), float(np.nanmax(array))
        if hi <= lo:
            hi = lo + 1.0
        norm = np.clip((array - lo) / (hi - lo), 0.0, 1.0)
        array = np.stack([norm, norm, norm], axis=-1)
    array = np.clip(array, 0.0, 1.0)
    return (array * 255.0).astype(np.uint8)


def _overlay_mri_pred(mri: np.ndarray, pred_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    base = _to_image(mri).astype(np.float32) / 255.0
    overlay = np.clip(pred_rgb, 0.0, 1.0)
    mask = overlay.sum(axis=-1, keepdims=True) > 0.05
    blended = np.where(mask, (1.0 - alpha) * base + alpha * overlay, base)
    return (np.clip(blended, 0.0, 1.0) * 255.0).astype(np.uint8)


@st.cache_resource
def _cached_model(path_str: str):
    return load_demo_model(Path(path_str), map_location="cpu")


def _header() -> None:
    st.markdown('<p class="nv-kicker">Phase 7 · 3D glioma segmentation research dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="nv-title">Neuro<span>Vision</span>-X</p>', unsafe_allow_html=True)
    st.markdown(f'<div class="nv-banner">{DISCLAIMER}</div>', unsafe_allow_html=True)
    st.caption(f"Package v{__version__} · YOLOv8 detector remains at app.py · no mock logits")


def _sidebar_checkpoint():
    st.sidebar.header("Checkpoint")
    checkpoints = list_demo_checkpoints()
    if not checkpoints:
        st.sidebar.error(
            "No `.pt` files under experiments/. Train a model (or run "
            "`scripts/smoke_train.py`) before inference. The UI will not invent predictions."
        )
        return None
    labels = [row["label"] for row in checkpoints]
    choice = st.sidebar.selectbox("Available weights (on disk)", labels)
    selected = next(row for row in checkpoints if row["label"] == choice)
    st.sidebar.write(f"Status: `{selected.get('status')}`")
    if selected["smoke"]:
        st.sidebar.warning("This is a **synthetic smoke** checkpoint (random tensors). Not BraTS.")
    else:
        st.sidebar.info("Non-smoke checkpoint. Still not a clinical device.")
    st.sidebar.header("Inference")
    class_index = st.sidebar.selectbox(
        "Grad-CAM class",
        options=[1, 2, 3],
        index=2,
        format_func=lambda i: {1: "1 NCR", 2: "2 ED", 3: "3 ET"}[i],
    )
    compute_cam = st.sidebar.checkbox("Compute 3D Grad-CAM (small volumes)", value=True)
    st.sidebar.caption("Heatmaps do not prove clinical reasoning. See docs/explainability.md.")
    return {**selected, "class_index": int(class_index), "compute_cam": bool(compute_cam)}


def _load_from_uploads(uploads: dict) -> dict | None:
    if not any(uploads.values()):
        return None
    tmp = Path(tempfile.mkdtemp(prefix="nvx_upload_"))
    paths: dict[str, Path | None] = {}
    try:
        for key, file in uploads.items():
            if file is None:
                paths[key] = None
                continue
            paths[key] = write_named_bytes(tmp, file.name, file.getvalue())
        resolved = validate_modality_paths(paths)
        return load_validated_case(resolved)
    except UploadValidationError as exc:
        st.error(str(exc))
        return None
    except Exception as exc:
        st.error(f"Failed to read NIfTI: {exc}")
        return None


def _synthetic_volume(patch: tuple[int, int, int]) -> dict:
    rng = np.random.default_rng(2026)
    image = rng.standard_normal((4, *patch), dtype=np.float32)
    image = image - image.mean(axis=(1, 2, 3), keepdims=True)
    std = image.std(axis=(1, 2, 3), keepdims=True)
    image = image / np.clip(std, 1e-3, None)
    d, h, w = patch
    seg = np.zeros(patch, dtype=np.int64)
    seg[d // 4 : d // 2, h // 4 : h // 2, w // 4 : w // 2] = 1
    seg[d // 3 : d // 2, h // 3 : h // 2, w // 3 : w // 2] = 3
    return {"image": image, "seg": seg, "synthetic": True}


def _show_triplanar(pack: dict, keys: list[tuple[str, str]]) -> None:
    cols = st.columns(3)
    for col, orientation in zip(cols, ORIENTATION_ORDER):
        with col:
            st.markdown(f"**{orientation.capitalize()}**")
            for key, caption in keys:
                if key == "overlay":
                    if "mri" not in pack[orientation] or "pred" not in pack[orientation]:
                        st.info(f"{caption}: not available")
                        continue
                    st.image(
                        _overlay_mri_pred(pack[orientation]["mri"], pack[orientation]["pred"]),
                        caption=caption,
                        width="stretch",
                    )
                    continue
                if key not in pack[orientation]:
                    st.info(f"{caption}: not available")
                    continue
                plane = pack[orientation][key]
                st.image(_to_image(plane), caption=caption, width="stretch")


def main() -> None:
    _header()
    selected = _sidebar_checkpoint()
    page = st.sidebar.radio(
        "Navigate",
        (
            "1. Data",
            "2. Anatomy",
            "3. Segmentation",
            "4. Uncertainty",
            "5. Explainability",
            "6. Metrics & export",
        ),
    )

    if selected is None:
        st.stop()

    try:
        model, payload = _cached_model(str(selected["path"]))
    except Exception as exc:
        st.error(f"Could not load checkpoint {selected['path']}: {exc}")
        st.stop()

    model_name = (payload.get("config") or {}).get("model", {}).get("name", "unknown")
    st.sidebar.markdown("#### Loaded weights")
    st.sidebar.write(f"Architecture: `{model_name}`")
    st.sidebar.write(f"Epoch stored: `{payload.get('epoch', '?')}`")
    st.sidebar.caption("Predictions come from this file's `state_dict`. The UI never invents logits.")

    if "case" not in st.session_state:
        st.session_state.case = None
    if "result" not in st.session_state:
        st.session_state.result = None
    if "bundle_bytes" not in st.session_state:
        st.session_state.bundle_bytes = None
    ckpt_key = str(selected["path"])
    if st.session_state.get("ckpt_path") != ckpt_key:
        st.session_state.ckpt_path = ckpt_key
        st.session_state.result = None
        st.session_state.bundle_bytes = None
        if st.session_state.case and st.session_state.case.get("synthetic"):
            st.session_state.case = None

    if page.startswith("1"):
        st.subheader("Input data")
        st.write(
            "Upload **co-registered** T1, T1ce, T2, and FLAIR NIfTI. "
            "Segmentation is optional (enables GT comparison). "
            "Missing modalities are rejected; they are not filled with zeros."
        )
        c1, c2, c3, c4, c5 = st.columns(5)
        nifti_types = ["nii", "nii.gz", "gz"]
        uploads = {
            "t1": c1.file_uploader("T1", type=nifti_types),
            "t1ce": c2.file_uploader("T1ce / T1c", type=nifti_types),
            "t2": c3.file_uploader("T2", type=nifti_types),
            "flair": c4.file_uploader("FLAIR / T2f", type=nifti_types),
            "seg": c5.file_uploader("Seg (optional)", type=nifti_types),
        }
        load_u = st.button("Validate and load uploads", type="primary")
        smoke_btn = st.button("Interface test: smoke checkpoint + matching synthetic volume")
        if load_u:
            if not any(uploads.values()):
                st.error("Upload T1, T1ce, T2, and FLAIR NIfTI files first. Empty channels are not synthesized.")
            else:
                case = _load_from_uploads(uploads)
                if case is not None:
                    case["synthetic"] = False
                    st.session_state.case = case
                    st.session_state.result = None
                    st.session_state.bundle_bytes = None
                    st.success(f"Loaded volume shape {tuple(case['image'].shape)} (C,D,H,W).")
        if smoke_btn:
            if not selected["smoke"]:
                st.error(
                    "The selected checkpoint is not a smoke run. "
                    "Synthetic volumes are only paired with the smoke weights "
                    "so the UI is not shown as a BraTS case."
                )
            else:
                from neurovisionx.demo.checkpoints import patch_size_from_payload

                patch = patch_size_from_payload(payload)
                st.session_state.case = _synthetic_volume(patch)
                st.session_state.result = None
                st.session_state.bundle_bytes = None
                st.warning(
                    f"Loaded a **random tensor** of shape (4,{patch[0]},{patch[1]},{patch[2]}) "
                    "for an interface test. This is not MRI."
                )

        case = st.session_state.case
        if case is not None:
            st.markdown("#### Volume summary")
            m1, m2, m3 = st.columns(3)
            m1.metric("Modalities", "4 (T1, T1ce, T2, FLAIR)")
            m2.metric("Spatial shape", str(tuple(int(v) for v in case["image"].shape[1:])))
            m3.metric("Ground truth", "yes" if "seg" in case else "not provided")
            if case.get("synthetic"):
                st.warning("SYNTHETIC TENSOR — not MRI, not a patient.")
            run = st.button("Run model inference", type="primary")
            if run:
                with st.spinner("Running the loaded checkpoint (no mock logits)…"):
                    try:
                        result = run_inference(
                            model,
                            payload,
                            case["image"],
                            target=case.get("seg"),
                            class_index=int(selected["class_index"]),
                            compute_cam=bool(selected["compute_cam"]),
                        )
                    except Exception as exc:
                        st.session_state.result = None
                        st.exception(exc)
                    else:
                        st.session_state.result = result
                        st.session_state.bundle_bytes = None
                        st.success(
                            f"Finished in {result['elapsed_ms']:.1f} ms on {result['device']} "
                            f"({result['n_params']} parameters)."
                        )
                        st.info(result["status_banner"])
                        st.caption("Open Anatomy, Segmentation, Uncertainty, Explainability, and Metrics next.")
        return

    case = st.session_state.case
    result = st.session_state.result
    if case is None:
        st.info("Load data on **1. Data** first.")
        st.stop()
    if result is None:
        st.info("Run inference on **1. Data** first. Views stay empty until a real forward pass.")
        st.stop()

    st.info(result["status_banner"])
    if case.get("synthetic"):
        st.warning("SYNTHETIC TENSOR — not MRI, not a patient. Ground-truth cubes are also synthetic.")
    lengths = {name: axis_length(result["prediction"], name) for name in ORIENTATION_ORDER}
    sliders = st.columns(3)
    indices = {}
    for col, name in zip(sliders, ORIENTATION_ORDER):
        with col:
            indices[name] = st.slider(
                f"{name} slice",
                0,
                max(lengths[name] - 1, 0),
                lengths[name] // 2,
                key=f"sl_{name}",
            )
    pack = slice_pack(
        case["image"],
        result["prediction"],
        target=case.get("seg"),
        entropy=result.get("entropy"),
        cam=result.get("cam"),
        indices=indices,
    )

    if page.startswith("2"):
        st.subheader("Original MRI (orthogonal)")
        st.caption("FLAIR display window 1–99th percentile. Axes are tensor axes after load, not DICOM numbers.")
        _show_triplanar(pack, [("mri", "FLAIR")])
        names = ("T1", "T1ce", "T2", "FLAIR")
        st.markdown("#### All modalities (selected axial)")
        cols = st.columns(4)
        for i, (col, name) in enumerate(zip(cols, names)):
            with col:
                channel = case["image"][i : i + 1]
                st.image(
                    _to_image(mri_plane(channel, "axial", indices["axial"])),
                    caption=name,
                    width="stretch",
                )

    elif page.startswith("3"):
        st.subheader("Predicted segmentation")
        st.caption("NCR blue · ED orange · ET red. Overlay is the actual argmax of the checkpoint.")
        _show_triplanar(pack, [("mri", "MRI"), ("overlay", "MRI + prediction"), ("pred", "Labels")])
        if "gt" in pack["axial"]:
            gt_title = "Synthetic label cube" if case.get("synthetic") else "Ground truth"
            st.markdown(f"#### {gt_title} vs prediction")
            if case.get("synthetic"):
                st.caption("The cube is a test fixture, not a radiologist contour.")
            _show_triplanar(pack, [("gt", gt_title), ("pred", "Prediction"), ("error", "WT error TP/FP/FN")])
        else:
            st.info("No segmentation NIfTI was uploaded, so GT comparison is skipped.")

    elif page.startswith("4"):
        st.subheader("Uncertainty")
        st.caption(
            "Predictive entropy from the **same softmax** as the displayed labels (T=1). "
            "This is not voxel-wise error and not a clinical confidence score."
        )
        if "entropy" not in pack["axial"]:
            st.error("Entropy was not computed.")
        else:
            _show_triplanar(pack, [("mri", "MRI"), ("entropy", "Entropy (nats)")])
            flat = result["entropy"].ravel()
            st.bar_chart({"entropy histogram": np.histogram(flat, bins=20)[0]})

    elif page.startswith("5"):
        st.subheader("Explainability (Grad-CAM)")
        st.caption(result.get("cam_note", ""))
        st.caption("Grad-CAM is not a proof of correctness. See docs/explainability.md.")
        if result.get("cam") is None:
            st.warning(result.get("cam_note", "Grad-CAM not available for this volume."))
            _show_triplanar(pack, [("mri", "MRI"), ("pred", "Prediction")])
        else:
            _show_triplanar(pack, [("mri", "MRI"), ("cam", "Grad-CAM"), ("pred", "Prediction")])

    else:
        st.subheader("Run information")
        a, b, c, d = st.columns(4)
        a.metric("Inference", f"{result['elapsed_ms']:.1f} ms")
        b.metric("Parameters", f"{result['n_params']:,}")
        c.metric("Device", result["device"])
        d.metric("Patch", str(tuple(result["patch"])))
        st.json(
            {
                "checkpoint": str(selected["path"]),
                "smoke": result["smoke"],
                "class_hist": result["class_hist"],
                "cam_note": result["cam_note"],
                "disclaimer": DISCLAIMER,
            }
        )
        if result.get("scores"):
            st.markdown("#### Overlap vs uploaded GT (this volume only)")
            st.dataframe({k: [v] for k, v in result["scores"].items()})
            st.caption("Single-case scores are not a challenge leaderboard.")
        hist = result["class_hist"]
        st.markdown("#### Predicted voxel counts")
        st.bar_chart({str(k): v for k, v in sorted(hist.items())})
        st.markdown("#### Download research bundle")
        if st.session_state.bundle_bytes is None:
            tmp = Path(tempfile.mkdtemp(prefix="nvx_bundle_"))
            try:
                st.session_state.bundle_bytes = bundle_zip(
                    case["image"], result, target=case.get("seg"), out_dir=tmp
                )
            except Exception as exc:
                st.error(f"Could not build the ZIP: {exc}")
        if st.session_state.bundle_bytes is not None:
            st.download_button(
                "Download PNG + JSON ZIP",
                data=st.session_state.bundle_bytes,
                file_name="neurovisionx_research_viz.zip",
                mime="application/zip",
            )
        st.caption(DISCLAIMER)


if __name__ == "__main__":
    main()
