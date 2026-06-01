#!/usr/bin/env python3
"""
preprocess.py  —  Stage 2 of the SD 1.5 texture fine-tuning pipeline.

Converts filtered_dataset/ images into training-ready 256x256 PNGs with
pre-computed block-DCT arrays.

Usage
-----
  python preprocess.py                    # process all images
  python preprocess.py --max-images 100   # smoke-test on first 100
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from scipy.fft import dctn
from tqdm import tqdm

FILTERED_DIR  = Path("./filtered_dataset")
PROCESSED_DIR = Path("./processed_dataset")
MANIFEST_FILE = PROCESSED_DIR / "manifest.csv"

MIN_SIDE    = 250   # discard images with shorter side below this
TARGET_SIZE = 256   # final square resolution


# ---------------------------------------------------------------------------
# Image processing helpers
# ---------------------------------------------------------------------------

def _centre_crop_resize(img: Image.Image) -> Optional[Image.Image]:
    """Square-crop to shorter side, Lanczos-resize to TARGET_SIZE. None if too small."""
    w, h = img.size
    short = min(w, h)
    if short < MIN_SIDE:
        return None
    left = (w - short) // 2
    top  = (h - short) // 2
    img  = img.crop((left, top, left + short, top + short))
    return img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)


def _compute_dct(img: Image.Image) -> np.ndarray:
    """
    Block-wise 8x8 DCT-II on the image normalised to [-1, 1].

    Output shape: (3, 32, 32, 8, 8)  —  [channel, block_row, block_col, pr, pc]

    Reshape trick: in C-order, reshaping (3, 256, 256) -> (3, 32, 8, 32, 8)
    maps element [c, row, col] to [c, row//8, row%8, col//8, col%8].
    A single transpose then puts block and intra-block axes in the right order.
    """
    arr  = np.array(img, dtype=np.float32) / 127.5 - 1.0    # (256, 256, 3), [-1, 1]
    chw  = arr.transpose(2, 0, 1)                             # (3, 256, 256)
    blks = chw.reshape(3, 32, 8, 32, 8).transpose(0, 1, 3, 2, 4)  # (3, 32, 32, 8, 8)
    return dctn(blks, axes=(-2, -1), norm="ortho").astype(np.float32)


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def _stems_in_manifest() -> set[str]:
    if not MANIFEST_FILE.exists():
        return set()
    with open(MANIFEST_FILE, newline="", encoding="utf-8") as f:
        return {row["filename_stem"] for row in csv.DictReader(f)}


def _is_complete(stem: str) -> bool:
    return (
        (PROCESSED_DIR / f"{stem}.png").exists()
        and (PROCESSED_DIR / f"{stem}_dct.npy").exists()
        and (PROCESSED_DIR / f"{stem}.txt").exists()
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess filtered images into 256x256 PNGs + DCT arrays.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--max-images", type=int, default=None, metavar="N",
        help="Process only the first N stems (sorted); for smoke-testing.",
    )
    args = parser.parse_args()

    if not FILTERED_DIR.exists():
        print(
            f"ERROR: {FILTERED_DIR} not found. Run filter_dataset.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    PROCESSED_DIR.mkdir(exist_ok=True)

    # Collect stems in deterministic sorted order.
    all_stems = sorted(p.stem for p in FILTERED_DIR.glob("img_*.jpg"))
    if not all_stems:
        print("No img_*.jpg files found in filtered_dataset/. Nothing to do.")
        return

    if args.max_images is not None:
        all_stems = all_stems[: args.max_images]
        print(f"--max-images {args.max_images}: capped at {len(all_stems)} stem(s).")

    # Determine what still needs processing.
    already_complete = {s for s in all_stems if _is_complete(s)}
    to_process       = [s for s in all_stems if s not in already_complete]
    in_manifest      = _stems_in_manifest()

    print(
        f"Stems total: {len(all_stems)} | "
        f"complete: {len(already_complete)} | "
        f"to process: {len(to_process)}"
    )

    # Open manifest in append mode; write header only when creating fresh.
    write_header = not MANIFEST_FILE.exists() or MANIFEST_FILE.stat().st_size == 0
    manifest_f = open(MANIFEST_FILE, "a", newline="", encoding="utf-8")
    manifest_w = csv.writer(manifest_f)
    if write_header:
        manifest_w.writerow(["filename_stem", "caption"])

    # Backfill manifest for stems that are complete on disk but missing from
    # the CSV (happens when a previous run crashed after writing files but
    # before flushing the manifest row).
    backfilled = 0
    for stem in sorted(already_complete):
        if stem not in in_manifest:
            txt = PROCESSED_DIR / f"{stem}.txt"
            caption = txt.read_text(encoding="utf-8").strip() if txt.exists() else ""
            manifest_w.writerow([stem, caption])
            backfilled += 1
    if backfilled:
        manifest_f.flush()
        print(f"Backfilled {backfilled} manifest entries for already-complete stems.")

    if not to_process:
        print("All images already processed.")
        manifest_f.close()
        return

    discarded = 0
    processed = 0
    t0 = time.time()

    for stem in tqdm(to_process, unit="img", desc="Preprocessing"):
        jpg_src = FILTERED_DIR / f"{stem}.jpg"
        txt_src = FILTERED_DIR / f"{stem}.txt"

        try:
            img = Image.open(jpg_src).convert("RGB")
        except Exception as exc:
            tqdm.write(f"  SKIP {stem}: cannot open ({exc})")
            discarded += 1
            continue

        img = _centre_crop_resize(img)
        if img is None:
            discarded += 1
            continue

        dct = _compute_dct(img)

        img.save(PROCESSED_DIR / f"{stem}.png")
        np.save(PROCESSED_DIR / f"{stem}_dct.npy", dct)

        if txt_src.exists():
            shutil.copy2(txt_src, PROCESSED_DIR / f"{stem}.txt")
            caption = txt_src.read_text(encoding="utf-8").strip()
        else:
            (PROCESSED_DIR / f"{stem}.txt").write_text("", encoding="utf-8")
            caption = ""

        manifest_w.writerow([stem, caption])
        manifest_f.flush()
        processed += 1

    manifest_f.close()
    elapsed = time.time() - t0

    # Disk usage estimate.
    # NPY size is exact: 3 * 32 * 32 * 8 * 8 * 4 bytes = 196 608 bytes
    # PNG is ~200 KB for lossless 256x256 RGB (varies by image content).
    total_done     = len(already_complete) + processed
    npy_kb         = 3 * 32 * 32 * 8 * 8 * 4 / 1024          # 192 KB
    per_img_kb     = 200 + npy_kb
    disk_gb        = total_done * per_img_kb / (1024 ** 2)

    print(
        f"\nDone in {elapsed:.1f}s | "
        f"processed: {processed} | discarded: {discarded} | "
        f"total in dataset: {total_done}"
    )
    print(
        f"Disk usage estimate: {disk_gb:.2f} GB "
        f"(~{per_img_kb:.0f} KB/image: ~200 KB PNG + {npy_kb:.0f} KB NPY)"
    )


if __name__ == "__main__":
    main()
