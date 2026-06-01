#!/usr/bin/env python3
"""
sweep_checkpoints.py  —  evaluate metric trends across training checkpoints.

Loads each selected checkpoint, generates a small image batch with the same
prompts/seeds validate.py uses, runs the same DSTG embed/extract/metrics
pipeline, then plots how the model evolves over training. Picks the best
checkpoint by reliability (exact_recovery > bit_accuracy > K_mean).

Usage
-----
  python sweep_checkpoints.py --checkpoint_root D:\\image_stego_runs\\finetune_output --stride 1
  python sweep_checkpoints.py --steps 2000,10000,80000
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import validate as v   # all DSTG, generation, evaluation primitives live here


# ===========================================================================
# Constants
# ===========================================================================
PSNR_TARGET_DB = 38.0   # horizontal-line marker in the PSNR panel
KEY_DESC = "exact_recovery > bit_accuracy > K_mean"


# ===========================================================================
# Checkpoint discovery + selection
# ===========================================================================
def discover_checkpoints(root: Path) -> list[tuple[int, Path, str]]:
    """
    Find checkpoint_stepNNNNNNN/ dirs (and optionally final/) under `root`.
    Returns [(step, path, label), ...] sorted by step ascending.
    final/ is placed last, at the highest seen step.
    """
    entries: list[tuple[int, Path, str]] = []
    seen_steps: set[int] = set()

    if not root.exists():
        return entries

    for d in root.iterdir():
        if not d.is_dir():
            continue
        if d.name.startswith("checkpoint_step"):
            try:
                step = int(d.name[len("checkpoint_step"):])
            except ValueError:
                continue
            entries.append((step, d, f"step{step}"))
            seen_steps.add(step)

    final = root / "final"
    if final.exists() and final.is_dir():
        step = max(seen_steps) if seen_steps else 0
        entries.append((step, final, "final"))

    # final sorts after the numbered checkpoint at the same step.
    entries.sort(key=lambda e: (e[0], 0 if e[2] != "final" else 1))
    return entries


def select_checkpoints(
    all_ckpts:      list[tuple[int, Path, str]],
    stride:         int,
    explicit_steps: Optional[list[int]],
) -> list[tuple[int, Path, str]]:
    """Apply --stride or --steps. With stride, always include the last entry."""
    if explicit_steps is not None:
        wanted = set(explicit_steps)
        return [e for e in all_ckpts if e[0] in wanted]
    if not all_ckpts:
        return []
    selected = list(all_ckpts[::max(1, stride)])
    if all_ckpts[-1] not in selected:
        selected.append(all_ckpts[-1])
    return selected


# ===========================================================================
# Evaluate one checkpoint (or baseline if ckpt_path is None)
# ===========================================================================
def evaluate_checkpoint(
    label:        str,
    step:         int,
    ckpt_path:    Optional[Path],
    plan:         list[tuple[int, int]],
    secret_text:  str,
    secret_bits:  list[int],
    sweep_out:    Path,
    device:       str,
    dtype:        torch.dtype,
    lora_rank:    int,
) -> dict:
    """Load -> generate -> free GPU -> DSTG-evaluate. Returns aggregated row."""
    work_dir = sweep_out / label
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- {label}  (step {step}) ---")
    print(f"  source: {'<baseline>' if ckpt_path is None else ckpt_path}")

    t0 = time.time()
    pipeline = v.load_pipeline(device, dtype, finetuned_dir=ckpt_path, lora_rank=lora_rank)
    print(f"  pipeline loaded in {time.time() - t0:.1f}s")

    t0 = time.time()
    covers = v.generate_images(pipeline, plan, work_dir, device)
    print(f"  generated {len(covers)} cover(s) in {time.time() - t0:.1f}s")

    # Explicit cleanup so the next checkpoint has the GPU free.
    # del must happen in this scope to drop the *caller's* reference.
    del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    t0 = time.time()
    image_rows = v.evaluate_images(covers, plan, work_dir, secret_text, secret_bits)
    print(f"  DSTG metrics computed in {time.time() - t0:.1f}s")

    avg = v.average(image_rows)
    row = {
        "step":           step,
        "label":          label,
        "K_mean":         avg["K"],
        "ratio":          avg["ratio"],
        "psnr_db":        avg["psnr_db"],
        "ssim":           avg["ssim"],
        "bit_accuracy":   avg["bit_accuracy"],
        "exact_recovery": avg["exact_recovery"],
        "capacity_bits":  avg["capacity_bits"],
        "bpp":            avg["bpp"],
        "capacity_bpp":   avg["capacity_bpp"],
    }
    print(
        f"  -> K_mean={row['K_mean']:.2f}  ratio={row['ratio']:.4f}  "
        f"PSNR={row['psnr_db']:.2f}  SSIM={row['ssim']:.4f}  "
        f"bit_acc={row['bit_accuracy']:.4f}  exact={row['exact_recovery']:.3f}"
    )
    return row


# ===========================================================================
# Outputs: table, CSV, plot
# ===========================================================================
def write_table(rows: list[dict], out_path: Path) -> str:
    lines = []
    lines.append("=" * 110)
    lines.append("SWEEP RESULTS  (one row per checkpoint, metrics averaged over images)")
    lines.append("=" * 110)
    lines.append(
        f"{'step':>7}  {'label':<14}"
        f"{'K_mean':>8}{'ratio':>9}{'PSNR':>8}{'SSIM':>8}"
        f"{'bit_acc':>10}{'exact':>8}{'cap_bits':>10}"
    )
    lines.append("-" * 110)
    for r in rows:
        lines.append(
            f"{r['step']:>7}  {r['label']:<14}"
            f"{r['K_mean']:>8.2f}{r['ratio']:>9.4f}"
            f"{r['psnr_db']:>8.2f}{r['ssim']:>8.4f}"
            f"{r['bit_accuracy']:>10.4f}{r['exact_recovery']:>8.3f}"
            f"{r['capacity_bits']:>10.0f}"
        )
    lines.append("=" * 110)
    text = "\n".join(lines)
    out_path.write_text(text + "\n", encoding="utf-8")
    return text


def write_csv(rows: list[dict], out_path: Path) -> None:
    cols = [
        "step", "label", "K_mean", "ratio", "psnr_db", "ssim",
        "bit_accuracy", "exact_recovery", "capacity_bits", "bpp", "capacity_bpp",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in cols})


def write_plot(
    rows:         list[dict],
    baseline_K:   Optional[float],
    out_path:     Path,
) -> None:
    if not rows:
        return
    steps        = [r["step"] for r in rows]
    K_vals       = [r["K_mean"] for r in rows]
    ratios       = [r["ratio"] for r in rows]
    psnr_vals    = [r["psnr_db"] for r in rows]
    bit_accs     = [r["bit_accuracy"] for r in rows]
    exact_recs   = [r["exact_recovery"] for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=110)

    # Top-left: K_mean
    ax = axes[0, 0]
    ax.plot(steps, K_vals, "o-", color="C0")
    if baseline_K is not None:
        ax.axhline(baseline_K, linestyle="--", color="gray",
                   label=f"baseline K = {baseline_K:.2f}")
        ax.legend(fontsize=9)
    ax.set_xlabel("Training step")
    ax.set_ylabel("K_mean (DSTG adaptive count)")
    ax.set_title("Mean adaptive K vs training step")
    ax.grid(alpha=0.3)

    # Top-right: mid-band ratio
    ax = axes[0, 1]
    ax.plot(steps, ratios, "o-", color="C2")
    ax.set_xlabel("Training step")
    ax.set_ylabel("DCT mid-band energy ratio")
    ax.set_title("Mid-band DCT energy ratio vs step")
    ax.grid(alpha=0.3)

    # Bottom-left: bit_accuracy + exact_recovery
    ax = axes[1, 0]
    ax.plot(steps, bit_accs,   "o-", color="C0", label="bit_accuracy")
    ax.plot(steps, exact_recs, "s-", color="C3", label="exact_recovery")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Recovery quality")
    ax.set_title("Bit accuracy + exact recovery vs step")
    ax.legend(fontsize=9)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)

    # Bottom-right: PSNR
    ax = axes[1, 1]
    ax.plot(steps, psnr_vals, "o-", color="C1")
    ax.axhline(PSNR_TARGET_DB, linestyle="--", color="gray",
               label=f"{PSNR_TARGET_DB:.0f} dB target")
    ax.set_xlabel("Training step")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("PSNR vs step")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint_root", type=str, default="./finetune_output")
    parser.add_argument("--sweep_output",    type=str, default="./validation_output/sweep")
    parser.add_argument("--num_images",      type=int, default=4)
    parser.add_argument("--stride",          type=int, default=6)
    parser.add_argument("--steps",           type=str, default=None,
                        help="Comma-separated explicit step list (overrides --stride).")
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--skip_baseline",   action="store_true",
                        help="Skip vanilla SD 1.5 as the reference 'step 0' row.")
    parser.add_argument("--lora_rank",       type=int, default=v.LORA_RANK)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if device == "cuda" else torch.float32
    if device == "cpu":
        print("WARNING: running on CPU — generation will be very slow.", file=sys.stderr)

    ckpt_root = Path(args.checkpoint_root)
    sweep_out = Path(args.sweep_output)
    sweep_out.mkdir(parents=True, exist_ok=True)

    explicit_steps: Optional[list[int]] = None
    if args.steps is not None:
        try:
            explicit_steps = [int(s.strip()) for s in args.steps.split(",") if s.strip()]
        except ValueError:
            print(f"ERROR: invalid --steps: {args.steps!r}", file=sys.stderr)
            sys.exit(1)

    # Discover
    all_ckpts = discover_checkpoints(ckpt_root)
    if not all_ckpts:
        print(f"ERROR: no checkpoints found under {ckpt_root}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(all_ckpts)} checkpoint(s) under {ckpt_root}:")
    for step, path, label in all_ckpts:
        print(f"  step={step:>7}  label={label:<14}  {path}")

    # Select
    selected = select_checkpoints(all_ckpts, args.stride, explicit_steps)
    if not selected:
        print("ERROR: no checkpoints matched the selection.", file=sys.stderr)
        sys.exit(1)

    print(f"\nSelected {len(selected)} checkpoint(s) for evaluation:")
    for step, _, label in selected:
        print(f"  {label}  (step {step})")

    # Plan + secret  (shared across baseline and all checkpoints)
    plan = v.build_plan(args.num_images, args.seed)
    print(f"\nPlan: {len(plan)} (prompt, seed) cover(s) per checkpoint")
    for pi, seed in plan:
        print(f"  prompt={pi}  seed={seed}  | {v.PROMPTS[pi]!r}")
    secret_text, secret_bits = v.make_secret(v.SECRET_BITS, seed=0)
    print(f"Secret: {v.SECRET_BITS} bits  ({len(secret_text)} chars)")

    # Collect rows
    rows: list[dict] = []

    if not args.skip_baseline:
        rows.append(evaluate_checkpoint(
            label="baseline", step=0, ckpt_path=None,
            plan=plan, secret_text=secret_text, secret_bits=secret_bits,
            sweep_out=sweep_out, device=device, dtype=dtype,
            lora_rank=args.lora_rank,
        ))

    for step, path, label in selected:
        rows.append(evaluate_checkpoint(
            label=label, step=step, ckpt_path=path,
            plan=plan, secret_text=secret_text, secret_bits=secret_bits,
            sweep_out=sweep_out, device=device, dtype=dtype,
            lora_rank=args.lora_rank,
        ))

    # Outputs
    table_path = sweep_out / "sweep_metrics.txt"
    csv_path   = sweep_out / "sweep_metrics.csv"
    plot_path  = sweep_out / "sweep_plot.png"

    table_text = write_table(rows, table_path)
    print()
    print(table_text)

    write_csv(rows, csv_path)
    print(f"\nWrote CSV : {csv_path}")

    baseline_K = next((r["K_mean"] for r in rows if r["label"] == "baseline"), None)
    write_plot(rows, baseline_K, plot_path)
    print(f"Wrote plot: {plot_path}")

    # Best fine-tuned checkpoint by reliability.
    # NOTE: must NOT call .sort() on tuples containing dicts — Python can't
    # compare dicts and raises TypeError. Use max() with an explicit key fn.
    fine_rows = [r for r in rows if r["label"] != "baseline"]
    if fine_rows:
        best = max(
            fine_rows,
            key=lambda r: (r["exact_recovery"], r["bit_accuracy"], r["K_mean"]),
        )
        best_line = (
            f"\nBest fine-tuned checkpoint by reliability ({KEY_DESC}):\n"
            f"  {best['label']:<14}  step={best['step']:>7}  "
            f"exact={best['exact_recovery']:.3f}  "
            f"bit_acc={best['bit_accuracy']:.4f}  "
            f"K_mean={best['K_mean']:.2f}"
        )
        print(best_line)
        with open(table_path, "a", encoding="utf-8") as f:
            f.write(best_line + "\n")


if __name__ == "__main__":
    main()
