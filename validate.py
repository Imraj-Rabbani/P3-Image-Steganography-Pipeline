#!/usr/bin/env python3
"""
validate.py  —  Stage 5 of the SD 1.5 texture fine-tuning pipeline.

Head-to-head comparison of baseline vs fine-tuned SD 1.5 using DSTG
steganography as the downstream task. Generates a fixed prompt x seed grid,
embeds and extracts a 512-bit secret per image, and reports per-image,
per-prompt, and aggregate metrics.

Usage
-----
  python validate.py
  python validate.py --finetuned_dir D:\\image_stego_runs\\finetune_output\\final
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import io
import string
import sys
import textwrap
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.fft import dctn, idctn
from skimage.metrics import peak_signal_noise_ratio as _sk_psnr
from skimage.metrics import structural_similarity as _sk_ssim
from tqdm.auto import tqdm

from diffusers import DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from peft import LoraConfig, get_peft_model

# DCT-domain stego engine (embed / extract / capacity / metrics) — same folder.
import dct


# ===========================================================================
# Constants
# ===========================================================================
SD15_MODEL_ID  = "runwayml/stable-diffusion-v1-5"
IMAGE_SIZE     = 256
DDIM_STEPS     = 50
SECRET_BITS    = 512
CRC_POLY       = 0x202D     # CRC-14 polynomial
MAGIC          = 0x4447     # 16-bit magic marker
VERSION        = 1
HEADER_BITS    = 64         # 32 in block (0,0) + 32 in block (0,1)

# LoRA config — must match finetune.py exactly to load the state dict.
LORA_RANK      = 4
LORA_ALPHA     = 8
LORA_DROPOUT   = 0.05
LORA_TARGETS   = ["to_q", "to_k", "to_v"]

# Fixed evaluation prompts (no CLI override — this IS the experiment).
PROMPTS: list[str] = [
    "a close-up photograph of tabby cat fur, sharp detail",
    "a macro photograph of green moss on a forest floor",
    "detailed bark texture of an old oak tree",
    "woven linen fabric, natural lighting, close-up",
    "rough granite stone surface, high detail",
    "shallow water with sand ripples, overhead view",
    "bird feathers in soft daylight, macro",
    "wool sweater texture, close detail",
    "weathered red brick wall, even lighting",
    "dense ivy leaves filling the frame",
]

# JPEG luminance Q-table (same as filter_dataset.py / preprocess.py / finetune.py)
Q_TABLE = np.array(
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

# Standard JPEG 8x8 zigzag scan order (64 positions).
ZIGZAG_ORDER: list[tuple[int, int]] = [
    (0,0),(0,1),(1,0),(2,0),(1,1),(0,2),(0,3),(1,2),
    (2,1),(3,0),(4,0),(3,1),(2,2),(1,3),(0,4),(0,5),
    (1,4),(2,3),(3,2),(4,1),(5,0),(6,0),(5,1),(4,2),
    (3,3),(2,4),(1,5),(0,6),(0,7),(1,6),(2,5),(3,4),
    (4,3),(5,2),(6,1),(7,0),(7,1),(6,2),(5,3),(4,4),
    (3,5),(2,6),(1,7),(2,7),(3,6),(4,5),(5,4),(6,3),
    (7,2),(7,3),(6,4),(5,5),(4,6),(3,7),(4,7),(5,6),
    (6,5),(7,4),(7,5),(6,6),(5,7),(6,7),(7,6),(7,7),
]

# Stable-mid scan order: zigzag, restricted to (non-DC, Q >= 8) — 39 positions.
STABLE_MID_ORDER: list[tuple[int, int]] = [
    (u, v) for (u, v) in ZIGZAG_ORDER
    if (u, v) != (0, 0) and Q_TABLE[u, v] >= 8.0
]
assert len(STABLE_MID_ORDER) == 39, f"expected 39 stable mid positions, got {len(STABLE_MID_ORDER)}"

# PASS/FAIL thresholds
THRESH_K_DELTA       = 0.0      # K_fine must EXCEED K_base
THRESH_PSNR_DB       = 38.0
THRESH_SSIM          = 0.95
THRESH_BIT_ACCURACY  = 0.99
THRESH_EXACT_REC     = 0.90


# ===========================================================================
# Block DCT helpers (full-image -> blocks -> DCT -> blocks -> full-image)
# ===========================================================================
def _block_dct(Y: np.ndarray) -> np.ndarray:
    """Y: (H, W) float32. Returns (bh, bw, 8, 8) orthonormal block DCT-II."""
    H, W = Y.shape
    bh, bw = H // 8, W // 8
    blocks = Y.reshape(bh, 8, bw, 8).transpose(0, 2, 1, 3).copy()  # (bh, bw, 8, 8)
    return dctn(blocks, axes=(-2, -1), norm="ortho")


def _block_idct(coeffs: np.ndarray) -> np.ndarray:
    """coeffs: (bh, bw, 8, 8). Returns Y of shape (bh*8, bw*8)."""
    bh, bw, _, _ = coeffs.shape
    blocks = idctn(coeffs, axes=(-2, -1), norm="ortho")
    return blocks.transpose(0, 2, 1, 3).reshape(bh * 8, bw * 8)


# ===========================================================================
# CRC-14, xorshift PRNG, bit helpers
# ===========================================================================
def _crc14(bits: Iterable[int], poly: int = CRC_POLY) -> int:
    crc = 0
    for bit in bits:
        if ((crc >> 13) & 1) ^ (bit & 1):
            crc = ((crc << 1) ^ poly) & 0x3FFF
        else:
            crc = (crc << 1) & 0x3FFF
    return crc


def _make_xorshift(seed: int) -> Callable[[], int]:
    """32-bit xorshift PRNG. Used only for distortion-tie-breaking."""
    state = [seed & 0xFFFFFFFF or 1]   # must be non-zero

    def _next() -> int:
        x = state[0]
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17)
        x ^= (x << 5)  & 0xFFFFFFFF
        state[0] = x
        return x

    return _next


def _int_to_bits(value: int, n_bits: int) -> list[int]:
    return [(value >> (n_bits - 1 - i)) & 1 for i in range(n_bits)]


def _bits_to_int(bits: Iterable[int]) -> int:
    out = 0
    for b in bits:
        out = (out << 1) | (b & 1)
    return out


def _bits_to_str(bits: list[int]) -> str:
    chars: list[str] = []
    for i in range(0, len(bits) - 7, 8):
        chars.append(chr(_bits_to_int(bits[i : i + 8])))
    return "".join(chars)


def _str_to_bits(s: str) -> list[int]:
    bits: list[int] = []
    for c in s:
        v = ord(c)
        for i in range(8):
            bits.append((v >> (7 - i)) & 1)
    return bits


# ===========================================================================
# Secret message
# ===========================================================================
def make_secret(num_bits: int = SECRET_BITS, seed: int = 0) -> tuple[str, list[int]]:
    """Deterministic 64-char (512-bit) printable ASCII secret."""
    assert num_bits % 8 == 0, "secret length must be a multiple of 8 bits"
    rng = np.random.RandomState(seed)
    alphabet = string.ascii_letters + string.digits     # 62 printable chars
    idx = rng.randint(0, len(alphabet), num_bits // 8)
    message = "".join(alphabet[i] for i in idx)
    return message, _str_to_bits(message)


# ===========================================================================
# DSTG header pack / unpack
# ===========================================================================
def _pack_header(payload_len: int, K: int, crc: int) -> list[int]:
    """64-bit header  =  magic(16) | version(4) | payload_len(24) | K(6) | crc(14)."""
    assert 0 <= payload_len < (1 << 24), "payload_len out of range"
    assert 0 <= K < (1 << 6),            "K out of range"
    assert 0 <= crc < (1 << 14),         "CRC out of range"
    bits  = _int_to_bits(MAGIC,       16)
    bits += _int_to_bits(VERSION,      4)
    bits += _int_to_bits(payload_len, 24)
    bits += _int_to_bits(K,            6)
    bits += _int_to_bits(crc,         14)
    assert len(bits) == HEADER_BITS
    return bits


def _unpack_header(bits: list[int]) -> tuple[int, int, int, int, int]:
    """Returns (magic, version, payload_len, K, crc)."""
    assert len(bits) == HEADER_BITS, f"header is {len(bits)} bits, expected {HEADER_BITS}"
    magic       = _bits_to_int(bits[ 0:16])
    version     = _bits_to_int(bits[16:20])
    payload_len = _bits_to_int(bits[20:44])
    K           = _bits_to_int(bits[44:50])
    crc         = _bits_to_int(bits[50:64])
    return magic, version, payload_len, K, crc


# ===========================================================================
# LSB-match write — min-distortion + never-below-2
# ===========================================================================
def _lsb_match(q: int, target: int, coeff: float, Q_uv: float, prng) -> int:
    """
    Adjust q so that (q & 1) == target_bit.

    Rules:
    - If LSB already matches, return q unchanged.
    - Otherwise flip by +1 or -1, choosing whichever minimises rounding error.
    - Never-below-2: if |q| >= 2 (cover-eligible), forbid flips that would
      cause |q_new| < 2 (which would change the cover eligibility set and
      break the extractor).
    """
    if (q & 1) == (target & 1):
        return q

    q_plus,  q_minus  = q + 1, q - 1
    d_plus           = abs(q_plus  * Q_uv - coeff)
    d_minus          = abs(q_minus * Q_uv - coeff)

    if abs(q) >= 2:
        plus_ok  = abs(q_plus)  >= 2
        minus_ok = abs(q_minus) >= 2
    else:
        plus_ok = minus_ok = True

    if plus_ok and minus_ok:
        if d_plus < d_minus:
            return q_plus
        if d_minus < d_plus:
            return q_minus
        return q_plus if (prng() & 1) else q_minus
    if plus_ok:
        return q_plus
    if minus_ok:
        return q_minus
    return q   # both directions would un-eligibilise — should be impossible with +/-1


# ===========================================================================
# DSTG embed / extract / capacity
# ===========================================================================
def _adaptive_K(q: np.ndarray) -> int:
    """K = clip(mean_eligible_per_block + mean_eligible // 2, 4, 39)."""
    stable_mask = np.zeros((8, 8), dtype=bool)
    for u, v in STABLE_MID_ORDER:
        stable_mask[u, v] = True
    eligible = (np.abs(q) >= 2) & stable_mask                   # (bh, bw, 8, 8)
    per_block = eligible.sum(axis=(-2, -1))                      # (bh, bw)
    mean_e = float(per_block.mean())
    return int(np.clip(mean_e + mean_e // 2, 4, len(STABLE_MID_ORDER)))


def _payload_block_iter(bh: int, bw: int) -> Iterable[tuple[int, int]]:
    """Raster order over all blocks except the two header blocks (0,0), (0,1)."""
    for bi in range(bh):
        for bj in range(bw):
            if (bi, bj) in ((0, 0), (0, 1)):
                continue
            yield bi, bj


@contextlib.contextmanager
def _quiet():
    """Silence dct.py's progress prints so the validation loop stays readable."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def dstg_embed(
    cover_bgr: np.ndarray,
    message: str,
) -> tuple[np.ndarray, dict]:
    """Embed `message` into cover via the DCT-domain engine in dct.py.

    dct.embed embeds in the Y channel and repairs the stego so the payload
    survives the BGR<->YCrCb round-trip — extraction is lossless. Returns
    (stego_bgr, info), keeping the info dict shape the rest of validate uses.
    """
    H, W = cover_bgr.shape[:2]
    assert H % 8 == 0 and W % 8 == 0, "cover dims must be multiples of 8"

    with _quiet():
        stego_bgr = dct.embed(cover_bgr, message)

    # K reported as the experiment's *texture-quality* metric: clip(mean + mean//2,
    # 4, 39) over eligible stable-mid coefficients. This is intentionally NOT
    # dct.get_capacity()'s K — that one is a capacity estimate that falls back to
    # FALLBACK_K (24) on near-flat covers and has no floor, which spikes/distorts
    # the baseline-vs-finetuned K comparison. Keep the original monotonic metric.
    Y_centred = cv2.cvtColor(cover_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32) - 128.0
    q         = np.round(_block_dct(Y_centred) / Q_TABLE).astype(np.int32)
    K_metric  = _adaptive_K(q)

    payload_len = len(message.encode("utf-8")) * 8
    info = dict(
        K              = K_metric,
        payload_len    = payload_len,
        embedded_bits  = payload_len,
        magic          = dct.MAGIC_SHORT,
        version        = dct.VERSION,
    )
    return stego_bgr, info


def dstg_extract(stego_bgr: np.ndarray) -> tuple[str, list[int]]:
    """Returns (recovered_message, recovered_bits). Empty on any failure.

    Delegates to dct.extract, which reads the CRC-checked header and payload
    straight from the stego's Y channel. dct.extract raises on a bad header or
    UTF-8 decode error; we map those to the empty result the harness expects.
    """
    try:
        with _quiet():
            message = dct.extract(stego_bgr)
    except (ValueError, RuntimeError):
        return "", []
    return message, _str_to_bits(message)


def dstg_get_capacity(cover_bgr: np.ndarray) -> int:
    """Total payload bits embeddable under dct.py's adaptive-K rule."""
    with _quiet():
        return int(dct.get_capacity(cover_bgr)["payload_bits"])


# ===========================================================================
# Metric helpers
# ===========================================================================
def compute_mid_ratio(bgr_img: np.ndarray) -> float:
    """Mid-band energy ratio: sum |coeff|^2 where u+v in [3, 10] over total."""
    Y = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
    bh, bw = Y.shape[0] // 8, Y.shape[1] // 8
    blocks = Y.reshape(bh, 8, bw, 8).transpose(0, 2, 1, 3).copy()
    coeffs = dctn(blocks, axes=(-2, -1), norm="ortho")
    uv     = np.add.outer(np.arange(8), np.arange(8))              # (8, 8)
    mask   = (uv >= 3) & (uv <= 10)
    energy_total = float((coeffs ** 2).sum())
    energy_mid   = float(((coeffs ** 2) * mask).sum())
    return energy_mid / max(energy_total, 1e-10)


def bit_accuracy(orig_bits: list[int], rec_bits: list[int]) -> float:
    if not orig_bits:
        return 0.0
    n = min(len(orig_bits), len(rec_bits))
    if n == 0:
        return 0.0
    matches = sum(1 for a, b in zip(orig_bits[:n], rec_bits[:n]) if a == b)
    return matches / len(orig_bits)


# ===========================================================================
# SD 1.5 pipeline loading
# ===========================================================================
def _build_pipeline(device: str, dtype: torch.dtype) -> StableDiffusionPipeline:
    pipeline = StableDiffusionPipeline.from_pretrained(
        SD15_MODEL_ID,
        torch_dtype           = dtype,
        safety_checker        = None,
        requires_safety_checker = False,
    )
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def load_baseline_pipeline(device: str, dtype: torch.dtype) -> StableDiffusionPipeline:
    return _build_pipeline(device, dtype)


def load_finetuned_pipeline(
    device:        str,
    dtype:         torch.dtype,
    finetuned_dir: Path,
    lora_rank:     int = LORA_RANK,
) -> StableDiffusionPipeline:
    """Build pipeline, wrap UNet in LoRA, load checkpoint, merge LoRA."""
    pipeline = _build_pipeline(device, dtype)

    unet_pt = finetuned_dir / "unet.pt"
    if not unet_pt.exists():
        raise FileNotFoundError(f"Missing unet checkpoint: {unet_pt}")

    lora_cfg = LoraConfig(
        r              = lora_rank,
        lora_alpha     = LORA_ALPHA,
        target_modules = LORA_TARGETS,
        lora_dropout   = LORA_DROPOUT,
        bias           = "none",
    )
    unet = get_peft_model(pipeline.unet, lora_cfg)

    state = torch.load(unet_pt, map_location=device)
    missing, unexpected = unet.load_state_dict(state, strict=False)
    print(f"  load_state_dict(unet.pt) — missing={len(missing)}, unexpected={len(unexpected)}")

    pipeline.unet = unet.merge_and_unload()
    pipeline.unet.to(device, dtype=dtype)

    vae_pt = finetuned_dir / "vae.pt"
    if vae_pt.exists():
        pipeline.vae.load_state_dict(torch.load(vae_pt, map_location=device))
        pipeline.vae.to(device, dtype=dtype)
        print(f"  loaded vae.pt from {vae_pt}")

    return pipeline


def free_pipeline(pipeline: Optional[StableDiffusionPipeline]) -> None:
    if pipeline is not None:
        del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_pipeline(
    device:        str,
    dtype:         torch.dtype,
    finetuned_dir: Optional[Path] = None,
    lora_rank:     int = LORA_RANK,
) -> StableDiffusionPipeline:
    """Unified loader. None for baseline; a path for a fine-tuned checkpoint."""
    if finetuned_dir is None:
        return load_baseline_pipeline(device, dtype)
    return load_finetuned_pipeline(device, dtype, Path(finetuned_dir), lora_rank=lora_rank)


# ===========================================================================
# Image generation
# ===========================================================================
def generate_cover(
    pipeline: StableDiffusionPipeline,
    prompt:   str,
    seed:     int,
    device:   str,
    size:     int = IMAGE_SIZE,
    steps:    int = DDIM_STEPS,
) -> np.ndarray:
    """Returns BGR uint8 (size, size, 3)."""
    generator = torch.Generator(device=device).manual_seed(int(seed))
    pil = pipeline(
        prompt              = prompt,
        height              = size,
        width               = size,
        num_inference_steps = steps,
        generator           = generator,
    ).images[0]
    rgb = np.array(pil)                             # (H, W, 3) RGB uint8
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)     # BGR uint8


def generate_images(
    pipeline: StableDiffusionPipeline,
    plan:     list[tuple[int, int]],
    save_dir: Path,
    device:   str,
    size:     int = IMAGE_SIZE,
    steps:    int = DDIM_STEPS,
) -> dict[tuple[int, int], np.ndarray]:
    """Generate every (prompt_idx, seed) in `plan`. Saves PNGs to save_dir."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    covers: dict[tuple[int, int], np.ndarray] = {}
    for pi, seed in plan:
        img = generate_cover(pipeline, PROMPTS[pi], seed, device, size, steps)
        covers[(pi, seed)] = img
        cv2.imwrite(str(save_dir / f"cover_p{pi}_s{seed}.png"), img)
    return covers


# ===========================================================================
# Plan: (prompt_idx, seed) pairs
# ===========================================================================
def build_plan(num_images: int, seed_base: int) -> list[tuple[int, int]]:
    """5 prompts x 2 seeds = 10 covers by default."""
    NUM_SEEDS = 2
    num_pairs = max(1, min(len(PROMPTS), num_images // NUM_SEEDS))
    pairs: list[tuple[int, int]] = []
    for i in range(num_pairs):
        for j in range(NUM_SEEDS):
            pairs.append((i, seed_base + 2 * i + j))
    return pairs


# ===========================================================================
# Run DSTG + metrics for one cover
# ===========================================================================
def evaluate_one(
    cover_bgr:    np.ndarray,
    stego_save_path: Path,
    secret_text:  str,
    secret_bits:  list[int],
) -> dict:
    H, W = cover_bgr.shape[:2]

    # Embed -> save as PNG -> reload -> extract  (PNG round-trip is the realism)
    stego_bgr, info = dstg_embed(cover_bgr, secret_text)
    cv2.imwrite(str(stego_save_path), stego_bgr)
    stego_loaded = cv2.imread(str(stego_save_path), cv2.IMREAD_COLOR)

    recovered_msg, recovered_bits = dstg_extract(stego_loaded)

    K              = info["K"]
    mid_ratio      = compute_mid_ratio(cover_bgr)
    qm             = dct.quality_metrics(cover_bgr, stego_loaded, SECRET_BITS)
    psnr_db        = float(qm["psnr"])     # luma (Y) PSNR — fair for Y embedding
    ssim_val       = float(qm["ssim"])     # luma (Y) SSIM
    bit_acc        = bit_accuracy(secret_bits, recovered_bits)
    exact_rec      = 1.0 if recovered_msg == secret_text else 0.0
    capacity_bits  = dstg_get_capacity(cover_bgr)
    bpp            = SECRET_BITS    / float(H * W)
    capacity_bpp   = capacity_bits  / float(H * W)

    return dict(
        K              = K,
        ratio          = mid_ratio,
        psnr_db        = psnr_db,
        ssim           = ssim_val,
        bit_accuracy   = bit_acc,
        exact_recovery = exact_rec,
        capacity_bits  = capacity_bits,
        bpp            = bpp,
        capacity_bpp   = capacity_bpp,
    )


def evaluate_images(
    covers:      dict[tuple[int, int], np.ndarray],
    plan:        list[tuple[int, int]],
    save_dir:    Path,
    secret_text: str,
    secret_bits: list[int],
) -> list[dict]:
    """Run DSTG + metrics for every cover in `plan`. Saves stegos to save_dir."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for pi, seed in plan:
        rows.append(evaluate_one(
            covers[(pi, seed)],
            save_dir / f"stego_p{pi}_s{seed}.png",
            secret_text, secret_bits,
        ))
    return rows


# ===========================================================================
# Aggregation helpers
# ===========================================================================
METRIC_KEYS = (
    "K", "ratio", "psnr_db", "ssim",
    "bit_accuracy", "exact_recovery",
    "capacity_bits", "bpp", "capacity_bpp",
)


def average(rows: list[dict]) -> dict:
    out: dict[str, float] = {}
    if not rows:
        return {k: float("nan") for k in METRIC_KEYS}
    for k in METRIC_KEYS:
        out[k] = float(np.mean([r[k] for r in rows]))
    return out


def fmt(v: float, decimals: int = 4) -> str:
    if not np.isfinite(v):
        return "  nan "
    return f"{v:.{decimals}f}"


# ===========================================================================
# Output: summary, per-prompt table, grid, CSV, PASS/FAIL
# ===========================================================================
def write_summary(
    base_rows: list[dict],
    fine_rows: list[dict],
    out_path:  Path,
) -> tuple[str, dict, dict]:
    base_avg = average(base_rows)
    fine_avg = average(fine_rows)

    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("AGGREGATE METRICS  (averaged over all images per model)")
    lines.append("=" * 78)
    lines.append(f"{'Metric':<18} {'Baseline':>12} {'Fine-tuned':>12} {'Delta':>12}")
    lines.append("-" * 78)
    for k in METRIC_KEYS:
        b, f = base_avg[k], fine_avg[k]
        d    = f - b
        decimals = 4 if k in ("ratio", "ssim", "bit_accuracy", "exact_recovery", "bpp", "capacity_bpp") else 2
        lines.append(f"{k:<18} {fmt(b, decimals):>12} {fmt(f, decimals):>12} {fmt(d, decimals):>12}")
    lines.append("=" * 78)

    text = "\n".join(lines)
    out_path.write_text(text + "\n", encoding="utf-8")
    return text, base_avg, fine_avg


def write_per_prompt_table(
    base_rows: list[dict],
    fine_rows: list[dict],
    plan:      list[tuple[int, int]],
    out_path:  Path,
) -> str:
    by_prompt_base: dict[int, list[dict]] = {}
    by_prompt_fine: dict[int, list[dict]] = {}
    for idx, (pi, _) in enumerate(plan):
        by_prompt_base.setdefault(pi, []).append(base_rows[idx])
        by_prompt_fine.setdefault(pi, []).append(fine_rows[idx])

    lines: list[str] = []
    lines.append("\n" + "=" * 110)
    lines.append("PER-PROMPT BREAKDOWN  (averaged over seeds within each prompt)")
    lines.append("=" * 110)
    header = (
        f"{'#':<3}{'Prompt':<34}"
        f"{'B_K':>8}{'F_K':>8}{'dK':>8}"
        f"{'B_acc':>8}{'F_acc':>8}"
        f"{'B_ratio':>10}{'F_ratio':>10}"
    )
    lines.append(header)
    lines.append("-" * 110)
    for pi in sorted(by_prompt_base.keys()):
        b_avg = average(by_prompt_base[pi])
        f_avg = average(by_prompt_fine[pi])
        prompt_short = PROMPTS[pi][:32]
        lines.append(
            f"{pi:<3}{prompt_short:<34}"
            f"{b_avg['K']:>8.2f}{f_avg['K']:>8.2f}{(f_avg['K']-b_avg['K']):>+8.2f}"
            f"{b_avg['bit_accuracy']:>8.3f}{f_avg['bit_accuracy']:>8.3f}"
            f"{b_avg['ratio']:>10.4f}{f_avg['ratio']:>10.4f}"
        )
    lines.append("=" * 110)
    text = "\n".join(lines)

    with open(out_path, "a", encoding="utf-8") as f:
        f.write(text + "\n")
    return text


def write_grid(
    baseline_covers: dict[tuple[int, int], np.ndarray],
    finetune_covers: dict[tuple[int, int], np.ndarray],
    plan:            list[tuple[int, int]],
    out_path:        Path,
) -> None:
    # Use first seed of each prompt only (even seeds at default seed_base=42).
    seen: set[int] = set()
    rows: list[tuple[int, int]] = []
    for pi, seed in plan:
        if pi in seen:
            continue
        seen.add(pi)
        rows.append((pi, seed))
    n_rows = len(rows)
    if n_rows == 0:
        return

    fig, axes = plt.subplots(n_rows, 3, figsize=(13, 3.4 * n_rows), dpi=120)
    if n_rows == 1:
        axes = np.array([axes])

    for r, (pi, seed) in enumerate(rows):
        wrapped = textwrap.fill(PROMPTS[pi], width=22)
        axes[r, 0].text(0.5, 0.5, wrapped, ha="center", va="center", fontsize=11,
                        transform=axes[r, 0].transAxes)
        axes[r, 0].axis("off")

        b = baseline_covers[(pi, seed)]
        axes[r, 1].imshow(cv2.cvtColor(b, cv2.COLOR_BGR2RGB))
        axes[r, 1].set_title(f"Baseline  ·  seed {seed}", fontsize=9)
        axes[r, 1].axis("off")

        f = finetune_covers[(pi, seed)]
        axes[r, 2].imshow(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        axes[r, 2].set_title(f"Fine-tuned  ·  seed {seed}", fontsize=9)
        axes[r, 2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def write_csv(
    base_rows: list[dict],
    fine_rows: list[dict],
    plan:      list[tuple[int, int]],
    out_path:  Path,
) -> None:
    cols = ["model", "prompt_idx", "prompt_text", "seed"] + list(METRIC_KEYS)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for idx, (pi, seed) in enumerate(plan):
            for model, rows in (("baseline", base_rows), ("finetuned", fine_rows)):
                r = rows[idx]
                row = [model, pi, PROMPTS[pi], seed] + [r[k] for k in METRIC_KEYS]
                writer.writerow(row)


def pass_fail(base_avg: dict, fine_avg: dict) -> tuple[bool, list[tuple[str, bool, str]]]:
    checks = []

    dK_ok = fine_avg["K"] - base_avg["K"] > THRESH_K_DELTA
    checks.append((
        f"K_fine > K_base  (delta = {fine_avg['K'] - base_avg['K']:+.2f})",
        dK_ok,
        "raise --lambda_mid or train more steps if delta is <= 0" if not dK_ok else "",
    ))

    psnr_ok = fine_avg["psnr_db"] > THRESH_PSNR_DB
    checks.append((
        f"PSNR > {THRESH_PSNR_DB:.0f} dB  ({fine_avg['psnr_db']:.2f} dB)",
        psnr_ok,
        "embedding strength too high — lower K cap or shrink payload" if not psnr_ok else "",
    ))

    ssim_ok = fine_avg["ssim"] > THRESH_SSIM
    checks.append((
        f"SSIM > {THRESH_SSIM:.2f}  ({fine_avg['ssim']:.4f})",
        ssim_ok,
        "embedding strength too high — lower K cap" if not ssim_ok else "",
    ))

    bacc_ok = fine_avg["bit_accuracy"] > THRESH_BIT_ACCURACY
    checks.append((
        f"bit_accuracy > {THRESH_BIT_ACCURACY:.2f}  ({fine_avg['bit_accuracy']:.4f})",
        bacc_ok,
        "PNG round-trip noise too large — raise eligibility threshold above |q|>=2" if not bacc_ok else "",
    ))

    exact_ok = fine_avg["exact_recovery"] >= THRESH_EXACT_REC
    checks.append((
        f"exact_recovery >= {THRESH_EXACT_REC:.2f}  ({fine_avg['exact_recovery']:.4f})",
        exact_ok,
        "CRC failing — investigate header read errors" if not exact_ok else "",
    ))

    return all(ok for _, ok, _ in checks), checks


# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num_images",    type=int, default=10,
                        help="Total covers per model. Mapped to 5 prompts x (num_images//2) seeds.")
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--finetuned_dir", type=str, default="./finetune_output/final")
    parser.add_argument("--output_dir",    type=str, default="./validation_output")
    parser.add_argument("--lora_rank",     type=int, default=LORA_RANK)
    args = parser.parse_args()

    device       = "cuda" if torch.cuda.is_available() else "cpu"
    dtype        = torch.bfloat16 if device == "cuda" else torch.float32
    out_dir      = Path(args.output_dir)
    base_dir     = out_dir / "baseline"
    fine_dir     = out_dir / "finetuned"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_dir.mkdir(parents=True, exist_ok=True)
    fine_dir.mkdir(parents=True, exist_ok=True)

    finetuned_dir = Path(args.finetuned_dir)
    if not finetuned_dir.exists():
        print(f"ERROR: --finetuned_dir not found: {finetuned_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Device:        {device}")
    print(f"Output dir:    {out_dir}")
    print(f"Finetuned dir: {finetuned_dir}")

    plan = build_plan(args.num_images, args.seed)
    print(f"Plan: {len(plan)} (prompt, seed) covers per model")
    for pi, seed in plan:
        print(f"  prompt={pi}  seed={seed}  | {PROMPTS[pi]!r}")

    secret_text, secret_bits = make_secret(SECRET_BITS, seed=0)
    print(f"Secret: {SECRET_BITS} bits  ({len(secret_text)} chars)")

    # ---------------------------------------------------------------------
    # Phase 1: generate baseline covers
    # ---------------------------------------------------------------------
    print("\n[1/3] Generating BASELINE covers ...")
    baseline_covers: dict[tuple[int, int], np.ndarray] = {}
    pipeline = load_baseline_pipeline(device, dtype)
    t0 = time.time()
    for pi, seed in tqdm(plan, desc="baseline gen"):
        img = generate_cover(pipeline, PROMPTS[pi], seed, device)
        baseline_covers[(pi, seed)] = img
        cv2.imwrite(str(base_dir / f"cover_p{pi}_s{seed}.png"), img)
    print(f"  baseline generation: {time.time() - t0:.1f}s")
    free_pipeline(pipeline)

    # ---------------------------------------------------------------------
    # Phase 2: generate fine-tuned covers
    # ---------------------------------------------------------------------
    print("\n[2/3] Generating FINE-TUNED covers ...")
    finetune_covers: dict[tuple[int, int], np.ndarray] = {}
    pipeline = load_finetuned_pipeline(device, dtype, finetuned_dir, lora_rank=args.lora_rank)
    t0 = time.time()
    for pi, seed in tqdm(plan, desc="fine-tuned gen"):
        img = generate_cover(pipeline, PROMPTS[pi], seed, device)
        finetune_covers[(pi, seed)] = img
        cv2.imwrite(str(fine_dir / f"cover_p{pi}_s{seed}.png"), img)
    print(f"  fine-tuned generation: {time.time() - t0:.1f}s")
    free_pipeline(pipeline)

    # ---------------------------------------------------------------------
    # Phase 3: DSTG + metrics
    # ---------------------------------------------------------------------
    print("\n[3/3] Running DSTG embed/extract + metrics ...")
    base_rows: list[dict] = []
    fine_rows: list[dict] = []
    for pi, seed in tqdm(plan, desc="DSTG"):
        b_img = baseline_covers[(pi, seed)]
        b_row = evaluate_one(
            b_img,
            base_dir / f"stego_p{pi}_s{seed}.png",
            secret_text, secret_bits,
        )
        base_rows.append(b_row)

        f_img = finetune_covers[(pi, seed)]
        f_row = evaluate_one(
            f_img,
            fine_dir / f"stego_p{pi}_s{seed}.png",
            secret_text, secret_bits,
        )
        fine_rows.append(f_row)

    # ---------------------------------------------------------------------
    # Outputs
    # ---------------------------------------------------------------------
    metrics_txt = out_dir / "metrics.txt"
    metrics_csv = out_dir / "metrics.csv"
    grid_png    = out_dir / "comparison_grid.png"

    summary_text, base_avg, fine_avg = write_summary(base_rows, fine_rows, metrics_txt)
    print()
    print(summary_text)

    per_prompt_text = write_per_prompt_table(base_rows, fine_rows, plan, metrics_txt)
    print(per_prompt_text)

    write_csv(base_rows, fine_rows, plan, metrics_csv)
    print(f"\nWrote CSV : {metrics_csv}")

    write_grid(baseline_covers, finetune_covers, plan, grid_png)
    print(f"Wrote grid: {grid_png}")

    # ---------------------------------------------------------------------
    # PASS / FAIL
    # ---------------------------------------------------------------------
    all_ok, checks = pass_fail(base_avg, fine_avg)
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    with open(metrics_txt, "a", encoding="utf-8") as f:
        f.write("\nVERDICT\n" + "=" * 78 + "\n")
        for label, ok, hint in checks:
            mark = "PASS" if ok else "FAIL"
            line = f"  [{mark}]  {label}"
            print(line)
            f.write(line + "\n")
            if not ok and hint:
                hint_line = f"          hint: {hint}"
                print(hint_line)
                f.write(hint_line + "\n")
        verdict = "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"
        print("=" * 78)
        print(f"  {verdict}")
        f.write("=" * 78 + "\n" + f"  {verdict}\n")
    print()

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
