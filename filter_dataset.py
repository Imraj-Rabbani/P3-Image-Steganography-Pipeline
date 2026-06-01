#!/usr/bin/env python3
"""
filter_dataset.py  —  Stage 1 of the SD 1.5 texture fine-tuning pipeline.

Scans gmongaras/Imagenet21K_Recaption on Hugging Face Hub and keeps images
whose Y-channel mid-band DCT content exceeds a quality threshold suitable
for steganographic training data.

Usage
-----
  python filter_dataset.py                 # full run (50 000 images)
  python filter_dataset.py --max-shards 1  # single-shard smoke test
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files

# ---------------------------------------------------------------------------
# Constants (module-level so worker subprocesses can read them)
# ---------------------------------------------------------------------------

JPEG_Q: np.ndarray = np.array(
    [
        [ 3,  2,  2,  3,  4,  6,  8, 10],
        [ 2,  2,  3,  4,  5,  9, 10,  9],
        [ 3,  3,  4,  5,  6,  9, 11,  9],
        [ 3,  4,  5,  6,  8, 14, 13, 10],
        [ 4,  5,  7,  9, 11, 17, 16, 12],
        [ 5,  7,  9, 10, 13, 17, 18, 15],
        [10, 13, 12, 14, 16, 19, 19, 17],
        [14, 17, 18, 18, 19, 18, 19, 17],
    ],
    dtype=np.float32,
)

EMBED_POS = [
    (0,1),(1,0),(2,0),(1,1),(0,2),(0,3),(1,2),(2,1),(3,0),(4,0),
    (3,1),(2,2),(1,3),(0,4),(0,5),(1,4),(2,3),(3,2),(4,1),(5,0),
]
# Pre-split into row/col arrays for fast numpy fancy-indexing inside the worker.
_ER: np.ndarray = np.array([r for r, _ in EMBED_POS], dtype=np.intp)
_EC: np.ndarray = np.array([c for _, c in EMBED_POS], dtype=np.intp)

MIN_GOOD_BLOCK_RATIO = 0.30
TARGET_ACCEPTED      = 50_000
CHECKPOINT_INTERVAL  = 1_000    # write checkpoint every N accepted images
JPEG_QUALITY         = 95

REPO_ID         = "gmongaras/Imagenet21K_Recaption"
OUTPUT_DIR      = Path("filtered_dataset")
CHECKPOINT_FILE = Path("filter_checkpoint.json")
INDEX_FILE      = OUTPUT_DIR / "index.csv"

# Known column name candidates, tried in priority order.
_IMAGE_COL_CANDIDATES   = ("image", "jpg", "jpeg", "png", "img", "pixel_values")
_CAPTION_COL_CANDIDATES = ("recaption_short", "recaption","caption", "text", "description")


# ---------------------------------------------------------------------------
# Worker  (must be a module-level function so multiprocessing can pickle it)
# ---------------------------------------------------------------------------

def _analyze(
    payload: tuple[bytes, str],
) -> tuple[bool, float, Optional[bytes], str]:
    """
    Compute Y-channel DCT embeddability score for one image.

    Returns (accepted, good_block_ratio, image_bytes_or_None, caption).
    image_bytes is non-None only when accepted — avoids shipping rejected
    payload back across the process boundary.
    """
    image_bytes, caption = payload
    try:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return False, 0.0, None, caption

        Y = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)

        bh, bw = Y.shape[0] // 8, Y.shape[1] // 8
        total = bh * bw
        if total == 0:
            return False, 0.0, None, caption

        good = 0
        for by in range(bh):
            for bx in range(bw):
                block = Y[by*8 : by*8+8, bx*8 : bx*8+8]
                q = np.round(cv2.dct(block) / JPEG_Q).astype(np.int32)
                # Block is "good" only when ALL 20 mid-band positions are embeddable.
                if np.all(np.abs(q[_ER, _EC]) >= 1):
                    good += 1

        ratio = good / total
        if ratio > MIN_GOOD_BLOCK_RATIO:
            return True, ratio, image_bytes, caption
        return False, ratio, None, caption

    except Exception:
        return False, 0.0, None, caption


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _detect_image_field(df: pd.DataFrame) -> Optional[str]:
    """Return the column that holds raw image bytes."""
    for name in _IMAGE_COL_CANDIDATES:
        if name in df.columns:
            return name
    # Fallback: first column that stores bytes or a dict with a 'bytes' key.
    for col in df.columns:
        sample = df[col].dropna()
        if sample.empty:
            continue
        v = sample.iloc[0]
        if isinstance(v, (bytes, bytearray)):
            return col
        if isinstance(v, dict) and "bytes" in v:
            return col
    return None


def _detect_caption_field(df: pd.DataFrame) -> Optional[str]:
    """Return the column that holds the image caption / description."""
    for name in _CAPTION_COL_CANDIDATES:
        if name in df.columns:
            return name
    # Fallback: first string-typed column.
    for col in df.columns:
        sample = df[col].dropna()
        if not sample.empty and isinstance(sample.iloc[0], str):
            return col
    return None


def _get_bytes(value) -> Optional[bytes]:
    """Extract raw image bytes from the various formats HF datasets use."""
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, dict):
        b = value.get("bytes")
        if b is not None:
            return bytes(b)
    return None


def _clean_caption(value) -> str:
    if value is None:
        return ""
    return str(value).strip().replace("\n", " ").replace("\t", " ")[:220]


def _row_iter(
    df: pd.DataFrame,
    img_col: str,
    cap_col: Optional[str],
) -> Iterator[tuple[bytes, str]]:
    """Yield (image_bytes, caption) for every valid row in a shard DataFrame."""
    for row in df.to_dict("records"):
        img_bytes = _get_bytes(row.get(img_col))
        if img_bytes is None:
            continue
        cap = _clean_caption(row.get(cap_col) if cap_col else None)
        yield img_bytes, cap


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE) as f:
                data = json.load(f)
            if {"last_shard", "total_accepted"} <= data.keys():
                return data
        except Exception:
            pass
    return {"last_shard": -1, "total_accepted": 0}


def _save_checkpoint(last_shard: int, total_accepted: int) -> None:
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"last_shard": last_shard, "total_accepted": total_accepted}, f)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter ImageNet-21K-Recaption for DCT-rich training images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Set HF_TOKEN env var if the dataset requires authentication.",
    )
    parser.add_argument(
        "--max-shards", type=int, default=None, metavar="N",
        help="Stop after N shards (smoke-test mode).",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    ckpt      = _load_checkpoint()
    last_done: int = ckpt["last_shard"]
    total_acc: int = ckpt["total_accepted"]
    last_ckpt_acc  = total_acc

    print(f"Checkpoint  : last_shard={last_done}, total_accepted={total_acc}")

    if total_acc >= TARGET_ACCEPTED:
        print(f"Already reached target of {TARGET_ACCEPTED}. Nothing to do.")
        return

    # Open index.csv; write the header only on a genuinely fresh run.
    write_header = not INDEX_FILE.exists() or INDEX_FILE.stat().st_size == 0
    idx_file = open(INDEX_FILE, "a", newline="", encoding="utf-8")
    idx_writer = csv.writer(idx_file)
    if write_header:
        idx_writer.writerow(["filename", "caption", "good_block_ratio"])
        idx_file.flush()

    # Discover all parquet shards in the repository.
    print("Listing repository shards…")
    all_shards = sorted(
        p for p in list_repo_files(REPO_ID, repo_type="dataset")
        if p.endswith(".parquet")
    )
    if not all_shards:
        print("ERROR: no parquet shards found in repository.", file=sys.stderr)
        idx_file.close()
        sys.exit(1)

    shards = all_shards if args.max_shards is None else all_shards[: args.max_shards]
    print(f"Shards      : {len(shards)} (of {len(all_shards)} total)")

    n_workers = max(1, mp.cpu_count() - 2)
    print(f"Workers     : {n_workers}")
    print()

    img_field: Optional[str] = None
    cap_field: Optional[str] = None

    wall_t0         = time.time()
    rows_since_start = 0
    acc_since_start  = 0

    with mp.Pool(processes=n_workers) as pool:
        for shard_idx, shard_path in enumerate(shards):
            if shard_idx <= last_done:
                continue  # already processed in a previous run

            shard_t0 = time.time()
            print(f"[shard {shard_idx:>4d}] Downloading {shard_path} …", end=" ", flush=True)

            local = hf_hub_download(
                repo_id=REPO_ID,
                filename=shard_path,
                repo_type="dataset",
            )
            df = pd.read_parquet(local)
            print(f"{len(df)} rows")

            # Detect field names from the first usable shard.
            if img_field is None:
                img_field = _detect_image_field(df)
                if img_field is None:
                    print(
                        f"ERROR: cannot identify an image column.\n"
                        f"Available columns: {list(df.columns)}",
                        file=sys.stderr,
                    )
                    idx_file.close()
                    sys.exit(1)
                print(f"           image field   → {img_field!r}")
            if cap_field is None:
                cap_field = _detect_caption_field(df)
                print(f"           caption field → {cap_field!r}")

            acc_this_shard = 0
            reached_target = False

            for accepted, ratio, img_bytes, caption in pool.imap(
                _analyze,
                _row_iter(df, img_field, cap_field),
                chunksize=32,
            ):
                rows_since_start += 1
                if not accepted:
                    continue

                # --- save image ---
                idx   = total_acc
                fname = f"img_{idx:06d}.jpg"
                img_out = OUTPUT_DIR / fname

                if img_bytes and img_bytes[:2] == b"\xff\xd8":
                    # Original bytes are already JPEG — store them verbatim.
                    img_out.write_bytes(img_bytes)
                else:
                    arr = np.frombuffer(img_bytes, dtype=np.uint8)
                    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    cv2.imwrite(str(img_out), bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

                # --- save caption ---
                (OUTPUT_DIR / f"img_{idx:06d}.txt").write_text(caption, encoding="utf-8")

                # --- update index ---
                idx_writer.writerow([fname, caption, f"{ratio:.4f}"])
                idx_file.flush()

                total_acc       += 1
                acc_this_shard  += 1
                acc_since_start += 1

                # Periodic checkpoint (every CHECKPOINT_INTERVAL accepted images).
                if total_acc - last_ckpt_acc >= CHECKPOINT_INTERVAL:
                    _save_checkpoint(last_done, total_acc)
                    last_ckpt_acc = total_acc

                if total_acc >= TARGET_ACCEPTED:
                    reached_target = True
                    break  # stop consuming this shard's imap stream

            # --- end of shard ---
            shard_elapsed = time.time() - shard_t0
            elapsed       = time.time() - wall_t0

            # Rough ETA based on per-row acceptance rate since this run started.
            eta_str = "—"
            if acc_since_start > 0 and rows_since_start > 0 and elapsed > 0:
                accept_rate  = acc_since_start / rows_since_start
                rows_per_sec = rows_since_start / elapsed
                if accept_rate > 0 and rows_per_sec > 0:
                    rows_needed = (TARGET_ACCEPTED - total_acc) / accept_rate
                    eta_str = f"{rows_needed / rows_per_sec / 3600:.1f}h"

            print(
                f"[shard {shard_idx:>4d}] accepted={acc_this_shard:>5d} | "
                f"total={total_acc:>6d}/{TARGET_ACCEPTED} | "
                f"shard={shard_elapsed:.0f}s | "
                f"elapsed={elapsed/60:.1f}m | ETA={eta_str}"
            )

            # Only advance last_done when the shard was fully processed.
            if not reached_target:
                last_done = shard_idx

            # Always write checkpoint at shard boundary (and before stopping).
            _save_checkpoint(last_done, total_acc)
            last_ckpt_acc = total_acc

            if reached_target:
                print(f"Target of {TARGET_ACCEPTED} reached. Stopping.")
                break

    idx_file.close()
    total_elapsed = time.time() - wall_t0
    print(f"\nFinished. accepted={total_acc}  |  wall time={total_elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
