#!/usr/bin/env python3
"""
check.py  —  Stage 3 pre-flight check for the SD 1.5 texture fine-tuning pipeline.

Verifies processed_dataset/ integrity, GPU readiness, and DataLoader throughput
before committing to a long training run.

Usage
-----
  python check.py
"""

from __future__ import annotations

import csv
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

PROCESSED_DIR = Path("./processed_dataset")
MANIFEST_FILE = PROCESSED_DIR / "manifest.csv"

PASS_MARK = "✓"   # ✓
FAIL_MARK = "✗"   # ✗

# ---------------------------------------------------------------------------
# VRAM estimates for SD v1.5  (fp16 weights, fp32 optimizer states)
# batch=2, gradient-accumulation=4, anchor UNet enabled
# ---------------------------------------------------------------------------
_VRAM_TABLE: list[tuple[str, float]] = [
    ("Anchor UNet   (fp16, frozen)",        1.72),
    ("Training UNet (fp16, fwd+bwd)",       1.72),
    ("Activations   (batch=2, est.)",       2.00),
    ("Optimizer     (AdamW fp32, full FT)", 10.20),
    ("VAE encoder   (fp16)",                0.30),
    ("Text encoder  (fp16, frozen)",        0.25),
]
_VRAM_FULL_FT_GB  = sum(gb for _, gb in _VRAM_TABLE)   # ~16.2 GB
_VRAM_LORA_GB     = 7.0    # rough LoRA lower bound (most optimizer state gone)
_VRAM_MINIMUM_GB  = 6.0    # below this even LoRA is risky

# ---------------------------------------------------------------------------
# Module-level Dataset so DataLoader workers can pickle it on Windows
# (workers spawn a fresh interpreter and must be able to import the class)
# ---------------------------------------------------------------------------
_TORCH_OK = False
try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset

    class _ThroughputDataset(_TorchDataset):  # type: ignore[misc]
        def __init__(self, stems: list[str], root: Path) -> None:
            self.stems = stems
            self.root  = root

        def __len__(self) -> int:
            return len(self.stems)

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            stem = self.stems[idx]
            img  = np.array(
                Image.open(self.root / f"{stem}.png").convert("RGB"),
                dtype=np.float32,
            ) / 255.0
            dct  = np.load(self.root / f"{stem}_dct.npy")
            return (
                torch.from_numpy(img.transpose(2, 0, 1)),  # (3, 256, 256)
                torch.from_numpy(dct),                     # (3, 32, 32, 8, 8)
            )

    _TORCH_OK = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Reporting helper
# ---------------------------------------------------------------------------

def _r(ok: bool, label: str, detail: str = "") -> bool:
    mark = PASS_MARK if ok else FAIL_MARK
    line = f"  {mark} {label}"
    if detail:
        line += f"  — {detail}"
    print(line)
    return ok


# ---------------------------------------------------------------------------
# Check 1: dataset structure
# ---------------------------------------------------------------------------

def check_structure() -> tuple[bool, list[str]]:
    print("\n[1] Dataset structure")

    if not PROCESSED_DIR.exists():
        _r(False, "processed_dataset/ exists")
        return False, []
    _r(True, "processed_dataset/ exists")

    if not MANIFEST_FILE.exists():
        _r(False, "manifest.csv exists")
        return False, []
    _r(True, "manifest.csv exists")

    stems: list[str] = []
    with open(MANIFEST_FILE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            stems.append(row["filename_stem"])

    if not stems:
        _r(False, "manifest.csv non-empty", "0 rows")
        return False, []
    _r(True, "manifest.csv non-empty", f"{len(stems)} entries")

    # Verify every manifest entry has all three output files.
    miss_png = miss_npy = miss_txt = 0
    for stem in stems:
        if not (PROCESSED_DIR / f"{stem}.png").exists():
            miss_png += 1
        if not (PROCESSED_DIR / f"{stem}_dct.npy").exists():
            miss_npy += 1
        if not (PROCESSED_DIR / f"{stem}.txt").exists():
            miss_txt += 1

    triplets_ok = miss_png == 0 and miss_npy == 0 and miss_txt == 0
    detail = (
        f"all {len(stems)} triplets complete"
        if triplets_ok
        else f"missing: {miss_png} png, {miss_npy} npy, {miss_txt} txt"
    )
    _r(triplets_ok, "all (png + npy + txt) triplets complete", detail)

    # Check for PNG files on disk that are not in the manifest.
    manifest_set = set(stems)
    orphans = [p.stem for p in PROCESSED_DIR.glob("img_*.png") if p.stem not in manifest_set]
    if orphans:
        _r(False, "no orphan PNGs", f"{len(orphans)} PNG(s) absent from manifest")
    else:
        _r(True, "no orphan PNGs")

    return triplets_ok, stems


# ---------------------------------------------------------------------------
# Check 2: spot-check 10 random samples
# ---------------------------------------------------------------------------

def check_spot(stems: list[str]) -> bool:
    print("\n[2] Spot-check (10 random samples)")
    sample = random.sample(stems, min(10, len(stems)))
    all_ok = True

    for stem in sample:
        ok    = True
        notes: list[str] = []

        # PNG — must be 256x256 RGB
        try:
            img = Image.open(PROCESSED_DIR / f"{stem}.png").convert("RGB")
            if img.size != (256, 256):
                ok = False
                notes.append(f"wrong size {img.size}")
        except Exception as exc:
            ok = False
            notes.append(f"PNG error: {exc}")

        # DCT array — shape (3,32,32,8,8), finite values
        try:
            dct = np.load(PROCESSED_DIR / f"{stem}_dct.npy")
            if dct.shape != (3, 32, 32, 8, 8):
                ok = False
                notes.append(f"dct shape {dct.shape}")
            elif not np.isfinite(dct).all():
                ok = False
                notes.append("dct contains NaN/Inf")
        except Exception as exc:
            ok = False
            notes.append(f"NPY error: {exc}")

        # Caption — file must exist; empty is a warning only
        txt_path = PROCESSED_DIR / f"{stem}.txt"
        if not txt_path.exists():
            ok = False
            notes.append("txt missing")
        elif not txt_path.read_text(encoding="utf-8").strip():
            notes.append("caption empty (warning)")

        detail = ", ".join(notes) if notes else "256x256 RGB | dct (3,32,32,8,8) finite | caption ok"
        _r(ok, stem, detail)
        if not ok:
            all_ok = False

    return all_ok


# ---------------------------------------------------------------------------
# Check 3: CUDA / GPU
# ---------------------------------------------------------------------------

def check_cuda() -> tuple[bool, Optional[float]]:
    print("\n[3] CUDA / GPU")

    if not _TORCH_OK:
        _r(False, "torch importable", "not installed — run: pip install torch")
        return False, None
    _r(True, "torch importable", f"v{torch.__version__}")  # type: ignore[name-defined]

    if not torch.cuda.is_available():  # type: ignore[name-defined]
        _r(False, "CUDA available", "CPU-only build or no CUDA GPU found")
        return False, None
    _r(True, "CUDA available")

    name        = torch.cuda.get_device_name(0)      # type: ignore[name-defined]
    free_b, total_b = torch.cuda.mem_get_info(0)     # type: ignore[name-defined]
    free_gb  = free_b  / 1024 ** 3
    total_gb = total_b / 1024 ** 3
    _r(True, "GPU detected", f"{name}  |  {free_gb:.1f} GB free / {total_gb:.1f} GB total")
    return True, free_gb


# ---------------------------------------------------------------------------
# Check 4: VRAM estimate
# ---------------------------------------------------------------------------

def check_vram(free_gb: Optional[float]) -> bool:
    print("\n[4] VRAM estimate  (SD v1.5, batch=2, grad_accum=4, anchor enabled)")
    print()
    for label, gb in _VRAM_TABLE:
        print(f"       {gb:5.2f} GB   {label}")
    print(f"       {'':5}  -----")
    print(f"       {_VRAM_FULL_FT_GB:5.2f} GB   Total  (full fine-tune)")
    print(f"       {_VRAM_LORA_GB:5.2f} GB   Approx. with LoRA  (for reference)")
    print()

    if free_gb is None:
        return _r(False, "GPU available for VRAM check")

    if free_gb >= _VRAM_FULL_FT_GB:
        return _r(True,  "VRAM sufficient for full fine-tune",
                  f"{free_gb:.1f} GB free >= {_VRAM_FULL_FT_GB:.1f} GB needed")
    elif free_gb >= _VRAM_LORA_GB:
        _r(False, "VRAM sufficient for full fine-tune",
           f"only {free_gb:.1f} GB free (need ~{_VRAM_FULL_FT_GB:.1f} GB) — use LoRA")
        return _r(True, "VRAM sufficient for LoRA fine-tune",
                  f"{free_gb:.1f} GB free >= ~{_VRAM_LORA_GB:.1f} GB needed")
    else:
        _r(False, "VRAM sufficient for full fine-tune",
           f"only {free_gb:.1f} GB free")
        return _r(False, "VRAM sufficient for LoRA fine-tune",
                  f"{free_gb:.1f} GB < {_VRAM_MINIMUM_GB:.1f} GB minimum — upgrade GPU or reduce batch")


# ---------------------------------------------------------------------------
# Check 5: DataLoader throughput
# ---------------------------------------------------------------------------

def check_throughput(stems: list[str]) -> bool:
    print("\n[5] DataLoader throughput  (256 random samples, 4 workers)")

    if not _TORCH_OK:
        _r(False, "torch available for DataLoader test")
        return False

    from torch.utils.data import DataLoader  # type: ignore[import]

    N      = min(256, len(stems))
    sample = random.sample(stems, N)

    loader = DataLoader(
        _ThroughputDataset(sample, PROCESSED_DIR),
        batch_size=8,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),  # type: ignore[name-defined]
        persistent_workers=False,
    )

    t0    = time.perf_counter()
    count = 0
    try:
        for imgs, _dct in loader:
            count += imgs.shape[0]
    except Exception as exc:
        _r(False, "DataLoader loads without error", str(exc))
        return False

    elapsed = time.perf_counter() - t0
    ips = count / elapsed if elapsed > 0 else 0.0
    mins_per_epoch = (len(stems) / ips / 60) if ips > 0 else float("inf")

    _r(True, "DataLoader loads without error")
    ok = _r(
        ips >= 5,
        "Throughput acceptable (>= 5 img/s)",
        f"{ips:.0f} img/s  →  est. {mins_per_epoch:.1f} min/epoch "
        f"over {len(stems)} images",
    )
    return ok


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    random.seed(42)

    results: dict[str, bool] = {}

    structure_ok, stems = check_structure()
    results["Dataset structure"] = structure_ok

    if stems:
        results["Spot-check (10 samples)"] = check_spot(stems)
    else:
        results["Spot-check (10 samples)"] = False
        print("\n[2] Spot-check  — skipped (no stems)")

    cuda_ok, free_gb = check_cuda()
    results["CUDA available"] = cuda_ok

    vram_ok = check_vram(free_gb)
    results["VRAM sufficient"] = vram_ok

    if stems:
        results["DataLoader throughput"] = check_throughput(stems)
    else:
        results["DataLoader throughput"] = False
        print("\n[5] DataLoader throughput  — skipped (no stems)")

    # --- Summary ---
    print()
    print("─" * 52)
    print("SUMMARY")
    print("─" * 52)
    all_pass = True
    for name, ok in results.items():
        _r(ok, name)
        if not ok:
            all_pass = False
    print("─" * 52)
    if all_pass:
        print(f"  {PASS_MARK} All checks passed — ready to run finetune.py")
    else:
        print(f"  {FAIL_MARK} Some checks failed — fix issues above before training")
    print()

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
