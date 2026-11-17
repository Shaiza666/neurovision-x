"""Render GitHub README figures from the implemented stack.

Does not load BraTS NIfTI, does not invent predictions, and does not plot
research metrics. Output lives under assets/ (not assets/samples/, which is
gitignored).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets"

NAVY = "#0B1F33"
TEAL = "#0F766E"
TEAL_FILL = "#E6F4F1"
SLATE = "#334155"
MUTED = "#64748B"
LINE = "#94A3B8"
PAPER = "#F8FAFC"
WHITE = "#FFFFFF"
AMBER = "#92400E"
AMBER_FILL = "#FEF3C7"
BLUE_FILL = "#E8F1F8"
VIOLET_FILL = "#F3E8FF"
ROSE_FILL = "#FFE4E6"


def _setup(ax, xlim, ylim) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(WHITE)


def _round(ax, x, y, w, h, fc, ec, lw=1.2, r=0.18, z=2):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={r}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def _text(ax, x, y, s, **kwargs):
    defaults = dict(ha="center", va="center", color=NAVY, zorder=4)
    defaults.update(kwargs)
    ax.text(x, y, s, **defaults)


def _arrow(ax, x1, y1, x2, y2, color=TEAL):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.6,
            color=color,
            zorder=3,
        )
    )


def render_architecture(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16.4, 7.4), dpi=160)
    fig.patch.set_facecolor(WHITE)
    _setup(ax, (0, 16.4), (0, 7.4))

    ax.add_patch(Rectangle((0, 0), 16.4, 7.4, facecolor=WHITE, edgecolor="none"))
    _round(ax, 0.25, 6.55, 15.9, 0.6, NAVY, NAVY, r=0.12)
    _text(
        ax,
        8.2,
        6.85,
        "NeuroVision-X  ·  implemented research pipeline",
        color=WHITE,
        fontsize=13,
        fontweight="bold",
        fontfamily="DejaVu Sans",
    )

    stages = [
        (
            0.35,
            "1  Input",
            "Licensed BraTS NIfTI\nT1  ·  T1ce  ·  T2  ·  FLAIR\noptional segmentation",
            "Not in this repository",
            AMBER_FILL,
            AMBER,
        ),
        (
            3.05,
            "2  Data layer",
            "inventory  ·  validate\npatient-level splits\nRAS, 1 mm, z-score, patches",
            "MissingBraTSDataError if empty",
            TEAL_FILL,
            TEAL,
        ),
        (
            5.75,
            "3  Models",
            "3D U-Net (Cicek 2016)\nAttention U-Net (Oktay 2018)\noptional SE stem (Hu 2018)",
            "Established methods",
            BLUE_FILL,
            "#1D4ED8",
        ),
        (
            8.45,
            "4  Training",
            "Dice + inverse-freq. CE\noptional ET boundary loss\nAMP, clip, best.pt / latest.pt",
            "Patient training not run",
            VIOLET_FILL,
            "#6D28D9",
        ),
        (
            11.15,
            "5  Evaluation",
            "Dice  ·  HD95  ·  P/R\nECE  ·  NLL  ·  Brier\nMC dropout  ·  3D Grad-CAM",
            "BraTS metrics not measured",
            ROSE_FILL,
            "#BE123C",
        ),
        (
            13.85,
            "6  Dashboard",
            "app_research.py\ncheckpoint forward only\nno mock logits or fake metrics",
            "Experimental 3D U-Net demo",
            TEAL_FILL,
            TEAL,
        ),
    ]

    y, w, h = 2.55, 2.5, 3.55
    for x, title, body, foot, fc, ec in stages:
        _round(ax, x, y, w, h, fc, ec, lw=1.4, r=0.16)
        _round(ax, x + 0.12, y + h - 0.62, w - 0.24, 0.46, ec, ec, r=0.08)
        _text(
            ax,
            x + w / 2,
            y + h - 0.39,
            title,
            color=WHITE,
            fontsize=10,
            fontweight="bold",
            fontfamily="DejaVu Sans",
        )
        _text(
            ax,
            x + w / 2,
            y + 1.85,
            body,
            fontsize=8.2,
            color=SLATE,
            linespacing=1.45,
            fontfamily="DejaVu Sans",
        )
        _round(ax, x + 0.14, y + 0.16, w - 0.28, 0.62, WHITE, ec, lw=0.9, r=0.08)
        _text(
            ax,
            x + w / 2,
            y + 0.47,
            foot,
            fontsize=7.2,
            color=ec,
            fontfamily="DejaVu Sans",
        )

    for i in range(len(stages) - 1):
        x1 = stages[i][0] + w
        x2 = stages[i + 1][0]
        _arrow(ax, x1 + 0.02, y + h / 2, x2 - 0.04, y + h / 2)

    _round(ax, 0.35, 0.28, 15.7, 1.95, PAPER, LINE, lw=1.0, r=0.12)
    _text(
        ax,
        8.2,
        1.72,
        "What this diagram is — and is not",
        fontsize=10,
        fontweight="bold",
        fontfamily="DejaVu Sans",
    )
    _text(
        ax,
        8.2,
        0.95,
        "Software that exists in src/neurovisionx/. Official BraTS volumes are absent.\n"
        "No patient-level 3D U-Net training has been run. Smoke weights use random tensors and are not research.\n"
        "YOLOv8n (app.py) is a preserved 2D detector, not part of the glioma segmentation path shown above.",
        fontsize=8.4,
        color=SLATE,
        linespacing=1.5,
        fontfamily="DejaVu Sans",
    )

    fig.savefig(path, bbox_inches="tight", pad_inches=0.12, facecolor=WHITE)
    plt.close(fig)


def render_mri_layout(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14.2, 4.6), dpi=160)
    fig.patch.set_facecolor(WHITE)
    _setup(ax, (0, 14.2), (0, 4.6))

    _round(ax, 0.25, 3.85, 13.7, 0.52, AMBER_FILL, AMBER, r=0.1)
    _text(
        ax,
        7.1,
        4.11,
        "Expected MRI inputs  ·  official BraTS NIfTI are not stored in this repository",
        fontsize=11,
        color=AMBER,
        fontweight="bold",
        fontfamily="DejaVu Sans",
    )

    cards = [
        ("T1", "native T1-weighted"),
        ("T1ce", "contrast-enhanced T1"),
        ("T2", "T2-weighted"),
        ("FLAIR", "T2-FLAIR"),
    ]
    w, h = 2.9, 2.35
    gap = 0.28
    x0 = 0.55
    for i, (name, desc) in enumerate(cards):
        x = x0 + i * (w + gap)
        _round(ax, x, 1.15, w, h, PAPER, LINE, lw=1.3, r=0.14)
        ax.add_patch(
            FancyBboxPatch(
                (x + 0.22, 1.85),
                w - 0.44,
                1.35,
                boxstyle="round,pad=0.02,rounding_size=0.1",
                facecolor=WHITE,
                edgecolor=LINE,
                linewidth=1.0,
                linestyle=(0, (4, 3)),
                zorder=3,
            )
        )
        _text(
            ax,
            x + w / 2,
            2.55,
            "no volume\nin git",
            fontsize=8.5,
            color=MUTED,
            fontfamily="DejaVu Sans",
            linespacing=1.35,
        )
        _text(
            ax,
            x + w / 2,
            1.55,
            name,
            fontsize=12,
            fontweight="bold",
            fontfamily="DejaVu Sans",
        )
        _text(
            ax,
            x + w / 2,
            1.32,
            desc,
            fontsize=7.6,
            color=MUTED,
            fontfamily="DejaVu Sans",
        )

    _text(
        ax,
        7.1,
        0.55,
        "These panels are empty placeholders for co-registered channels. They are not MRI scans and not model output.",
        fontsize=8.6,
        color=SLATE,
        fontfamily="DejaVu Sans",
    )
    fig.savefig(path, bbox_inches="tight", pad_inches=0.12, facecolor=WHITE)
    plt.close(fig)


def render_unet(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(15.6, 7.8), dpi=160)
    fig.patch.set_facecolor(WHITE)
    _setup(ax, (0, 15.6), (0, 7.8))

    _round(ax, 0.25, 7.05, 15.1, 0.52, NAVY, NAVY, r=0.1)
    _text(
        ax,
        7.8,
        7.31,
        "Implemented 3D U-Net  ·  configs/model.yaml  ·  src/neurovisionx/models/unet3d.py",
        color=WHITE,
        fontsize=11.5,
        fontweight="bold",
        fontfamily="DejaVu Sans",
    )

    # Encoder (left, top → bottom), bottleneck, decoder (right, bottom → top)
    enc = [
        (2.15, 5.55, "Enc 16", "ConvBlock\n4 → 16"),
        (2.15, 4.15, "Enc 32", "ConvBlock\n16 → 32"),
        (2.15, 2.75, "Enc 64", "ConvBlock\n32 → 64"),
        (2.15, 1.35, "Enc 128", "ConvBlock\n64 → 128"),
    ]
    dec = [
        (10.55, 5.55, "Dec 16", "up + skip concat\n→ ConvBlock 16"),
        (10.55, 4.15, "Dec 32", "up + skip concat\n→ ConvBlock 32"),
        (10.55, 2.75, "Dec 64", "up + skip concat\n→ ConvBlock 64"),
        (10.55, 1.35, "Dec 128", "up + skip concat\n→ ConvBlock 128"),
    ]

    _round(ax, 0.4, 5.55, 1.45, 1.15, AMBER_FILL, AMBER, r=0.1)
    _text(ax, 1.125, 6.28, "Input", fontsize=9, fontweight="bold", fontfamily="DejaVu Sans")
    _text(
        ax,
        1.125,
        5.88,
        "4 channels\nT1 T1ce\nT2 FLAIR",
        fontsize=7.4,
        color=SLATE,
        fontfamily="DejaVu Sans",
        linespacing=1.25,
    )
    _arrow(ax, 1.88, 6.12, 2.12, 6.12)

    for x, y, title, body in enc:
        _round(ax, x, y, 2.35, 1.15, TEAL_FILL, TEAL, r=0.1)
        _text(ax, x + 1.175, y + 0.82, title, fontsize=10, fontweight="bold", fontfamily="DejaVu Sans")
        _text(ax, x + 1.175, y + 0.38, body, fontsize=7.4, color=SLATE, fontfamily="DejaVu Sans", linespacing=1.25)

    for (x1, y1, *_), (x2, y2, *_) in zip(enc, enc[1:]):
        _arrow(ax, x1 + 1.175, y1, x2 + 1.175, y2 + 1.15)
        _text(
            ax,
            x1 + 0.08,
            (y1 + y2 + 1.15) / 2,
            "pool 2",
            fontsize=6.5,
            color=MUTED,
            ha="right",
            fontfamily="DejaVu Sans",
        )

    _round(ax, 6.35, 0.22, 2.9, 1.0, VIOLET_FILL, "#6D28D9", r=0.1)
    _text(ax, 7.8, 0.88, "Bottleneck 256", fontsize=10, fontweight="bold", fontfamily="DejaVu Sans")
    _text(ax, 7.8, 0.48, "ConvBlock  128 → 256", fontsize=7.5, color=SLATE, fontfamily="DejaVu Sans")
    _arrow(ax, 4.5, 1.45, 6.32, 0.95)
    _arrow(ax, 9.28, 0.95, 10.55, 1.45)

    for x, y, title, body in dec:
        _round(ax, x, y, 2.55, 1.15, BLUE_FILL, "#1D4ED8", r=0.1)
        _text(ax, x + 1.275, y + 0.82, title, fontsize=10, fontweight="bold", fontfamily="DejaVu Sans")
        _text(ax, x + 1.275, y + 0.38, body, fontsize=7.3, color=SLATE, fontfamily="DejaVu Sans", linespacing=1.25)

    for lower, upper in zip(dec[1:], dec):
        _arrow(ax, lower[0] + 1.275, lower[1] + 1.15, upper[0] + 1.275, upper[1])

    # Skip connections
    for i, (e, d) in enumerate(zip(enc, dec)):
        ax.annotate(
            "",
            xy=(d[0], e[1] + 0.58),
            xytext=(e[0] + 2.35, e[1] + 0.58),
            arrowprops=dict(arrowstyle="-|>", color=LINE, lw=1.2, ls=(0, (5, 3))),
            zorder=1,
        )
    _text(ax, 7.8, 6.82, "skip concat", fontsize=7.2, color=MUTED, fontfamily="DejaVu Sans")

    _round(ax, 13.35, 5.55, 1.85, 1.15, ROSE_FILL, "#BE123C", r=0.1)
    _text(ax, 14.275, 6.28, "Head", fontsize=9, fontweight="bold", fontfamily="DejaVu Sans")
    _text(
        ax,
        14.275,
        5.88,
        "Conv3d 1×1×1\n16 → 4 classes\nBG NCR ED ET",
        fontsize=7.2,
        color=SLATE,
        fontfamily="DejaVu Sans",
        linespacing=1.25,
    )
    _arrow(ax, 13.1, 6.12, 13.32, 6.12)

    _round(ax, 0.4, 0.22, 5.6, 1.0, PAPER, LINE, r=0.1)
    _text(
        ax,
        3.2,
        0.72,
        "ConvBlock = Conv3d 3×3×3 + InstanceNorm3d + PReLU (×2),\n"
        "Dropout3d 0.1. YAML features [16, 32, 64, 128].\n"
        "Smoke weights use [8, 16] and are not this figure.",
        fontsize=7.2,
        color=SLATE,
        fontfamily="DejaVu Sans",
        linespacing=1.35,
    )

    fig.savefig(path, bbox_inches="tight", pad_inches=0.12, facecolor=WHITE)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    architecture = OUT / "architecture.png"
    mri_layout = OUT / "mri_input_layout.png"
    unet = OUT / "model_architecture.png"
    render_architecture(architecture)
    render_mri_layout(mri_layout)
    render_unet(unet)
    for p in (architecture, mri_layout, unet):
        print(f"wrote {p} ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
