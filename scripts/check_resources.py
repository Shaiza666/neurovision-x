"""Print local compute limits. Does not start training or download BraTS."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _ram_gb() -> float | None:
    try:
        import ctypes

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MemoryStatusEx()
        stat.dwLength = ctypes.sizeof(MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return float(stat.ullTotalPhys) / (1024**3)
    except Exception:
        pass
    if hasattr(os, "sysconf"):
        try:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
        except (ValueError, OSError):
            return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Hardware check. Does not train.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        import torch
    except ImportError:
        torch = None  # type: ignore[assignment]

    ram = _ram_gb()
    disk = shutil.disk_usage(str(ROOT))
    cuda = bool(torch is not None and torch.cuda.is_available())
    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "ram_gb": None if ram is None else round(ram, 2),
        "disk_free_gb": round(disk.free / (1024**3), 1),
        "torch": None if torch is None else torch.__version__,
        "cuda_available": cuda,
        "cuda_built": bool(torch is not None and torch.backends.cuda.is_built()),
        "official_96_cubed_recommended": bool(cuda and ram is not None and ram >= 16),
        "smoke_16_cubed_ok": True,
        "h3_unetr": "deferred — no matched GPU comparison on this machine"
        if not cuda
        else "optional; factory still raises until protocol approval",
        "note": (
            "50-epoch 96³ BraTS training is not started by this script. "
            "Approve docs/experimental_protocol.md and place licensed NIfTI first."
        ),
        "clinical_disclaimer": "Research prototype. Not a clinically validated diagnostic tool.",
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return
    print("NeuroVision-X resource check (no training started)")
    for key, value in payload.items():
        print(f"  {key}: {value}")
    if not cuda:
        print("  Official A0–A5 96³ jobs should use a CUDA GPU. This host is CPU-only.")
    if ram is not None and ram < 16:
        print("  RAM is below 16 GB. Do not launch the 50-epoch 96³ schedule on this machine.")


if __name__ == "__main__":
    main()
