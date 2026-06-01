#!/usr/bin/env python3
"""
finetune.py  —  Stage 4 of the SD 1.5 texture fine-tuning pipeline.

LoRA-fine-tunes Stable Diffusion v1.5 with a custom DCT eligibility reward
to bias generations toward textures rich in mid-band Y-channel coefficients,
while a frozen anchor UNet prevents reward-hacking drift.

Loss
----
  L = L_diffusion + lambda_mid * L_mid_reward + lambda_anchor * L_anchor

Usage
-----
  python finetune.py                                          # full run (80k steps)
  python finetune.py --steps 200 --checkpoint_every 50        # smoke test
  python finetune.py --resume finetune_output/checkpoint_step0050000
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import math
import sys
import time
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from peft import LoraConfig, get_peft_model
from transformers import CLIPTextModel, CLIPTokenizer


# ===========================================================================
# Paths and model
# ===========================================================================
SD15_MODEL_ID  = "runwayml/stable-diffusion-v1-5"
PROCESSED_DIR  = Path("./processed_dataset")
OUTPUT_DIR     = Path("./finetune_output")
MANIFEST_FILE  = PROCESSED_DIR / "manifest.csv"


# ===========================================================================
# JPEG luminance Q-table  (same as filter_dataset.py / preprocess.py)
# ===========================================================================
JPEG_Q_NP = np.array(
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


# ===========================================================================
# Defaults
# ===========================================================================
BATCH_SIZE              = 2
GRAD_ACCUM_STEPS        = 4         # effective batch = 8
TRAIN_STEPS             = 80_000
WARMUP_STEPS            = 1_000
LR                      = 1e-5
WEIGHT_DECAY            = 1e-2
GRAD_CLIP               = 1.0
EMA_DECAY               = 0.9999
CHECKPOINT_EVERY        = 2_000
LORA_RANK               = 4
LORA_ALPHA              = 8
LORA_DROPOUT            = 0.05
LORA_TARGETS            = ["to_q", "to_k", "to_v"]
LAMBDA_MID_DEFAULT      = 0.005
LAMBDA_ANCHOR_DEFAULT   = 1.0
USE_ANCHOR_DEFAULT      = True
MID_SCORE_FN_DEFAULT    = "linear_hinge"


# ===========================================================================
# Windows sleep prevention
# ===========================================================================
_ES_CONTINUOUS      = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


def prevent_sleep() -> None:
    """Keep the system awake for the duration of training (Windows only)."""
    if sys.platform == "win32":
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(
                _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED
            )
        except Exception:
            pass


def allow_sleep() -> None:
    """Restore default sleep behavior."""
    if sys.platform == "win32":
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
        except Exception:
            pass


# ===========================================================================
# Orthonormal 8x8 DCT-II basis  (matches scipy.fft.dctn(..., norm="ortho"))
# ===========================================================================
def build_dct_basis() -> torch.Tensor:
    D = torch.zeros(8, 8, dtype=torch.float32)
    for k in range(8):
        alpha = math.sqrt(1.0 / 8.0) if k == 0 else math.sqrt(2.0 / 8.0)
        for n in range(8):
            D[k, n] = alpha * math.cos(math.pi * (2 * n + 1) * k / 16.0)
    return D


# ===========================================================================
# Score functions  (operate on |q| = |coeff / Q|)
# ===========================================================================
def linear_hinge_score(q_abs: torch.Tensor) -> torch.Tensor:
    """Linear ramp 0 (at |q|=0) -> 1 (at |q|=2), flat thereafter."""
    return torch.clamp(q_abs * 0.5, max=1.0)


def sigmoid_score(q_abs: torch.Tensor) -> torch.Tensor:
    """Sigmoid centred at |q|=1, slope 4 (~0.02 at |q|=0, ~0.98 at |q|=2)."""
    return torch.sigmoid(4.0 * (q_abs - 1.0))


SCORE_FNS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "linear_hinge": linear_hinge_score,
    "sigmoid":      sigmoid_score,
}


# ===========================================================================
# Dataset
# ===========================================================================
class ProcessedDataset(Dataset):
    """Loads (image_tensor, dct_tensor, caption_string) from processed_dataset/."""

    def __init__(self, root: Path, manifest_path: Path) -> None:
        self.root = root
        with open(manifest_path, newline="", encoding="utf-8") as f:
            self.records = [
                (row["filename_stem"], row.get("caption", "") or "")
                for row in csv.DictReader(f)
            ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        stem, caption = self.records[idx]
        img = Image.open(self.root / f"{stem}.png").convert("RGB")
        arr = np.array(img, dtype=np.float32) / 127.5 - 1.0   # [-1, 1]
        img_t = torch.from_numpy(arr.transpose(2, 0, 1))       # (3, 256, 256)
        dct = torch.from_numpy(np.load(self.root / f"{stem}_dct.npy"))
        return img_t, dct, caption


# ===========================================================================
# Mid-band DCT reward
# ===========================================================================
def compute_mid_reward(
    x_pixel:  torch.Tensor,      # (B, 3, H, W) in roughly [-1, 1]
    D:        torch.Tensor,      # (8, 8) DCT basis
    Q:        torch.Tensor,      # (8, 8) JPEG luminance table
    mid_mask: torch.Tensor,      # (8, 8) 39 ones at stable mid positions
    score_fn: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (eligibility_per_block_avg, mid_reward_loss).

    eligibility_per_block_avg : scalar in [0, 39] — mean # eligible mid coeffs / block.
    mid_reward_loss           : -eligibility, so minimising raises eligibility.
    """
    B, _, H, W = x_pixel.shape
    assert H % 8 == 0 and W % 8 == 0, "image must be divisible by 8"
    bh, bw = H // 8, W // 8

    # Scale [-1, 1] -> [0, 255], compute Y via BT.601, centre.
    x_255 = (x_pixel + 1.0) * 127.5
    Y     = 0.299 * x_255[:, 0] + 0.587 * x_255[:, 1] + 0.114 * x_255[:, 2]   # (B, H, W)
    Y_c   = Y - 128.0

    # Partition into 8x8 blocks  ->  (B, bh, bw, 8, 8)
    Yb = Y_c.reshape(B, bh, 8, bw, 8).permute(0, 1, 3, 2, 4).contiguous()

    # 2D DCT-II:  dct[k,l] = Σ_i Σ_j D[k,i] · X[i,j] · D[l,j]
    dct = torch.einsum("ki,bnmij,lj->bnmkl", D, Yb, D)        # (B, bh, bw, 8, 8)

    # Continuous q (NOT rounded - we need gradients through this).
    q_abs = (dct / Q).abs()

    # Score, mask to stable mid positions, sum per block, mean over batch x blocks.
    score     = score_fn(q_abs) * mid_mask                    # mid_mask broadcasts
    per_block = score.sum(dim=(-2, -1))                       # (B, bh, bw)
    eligibility = per_block.mean()                            # scalar in [0, 39]
    return eligibility, -eligibility


# ===========================================================================
# EMA over LoRA params
# ===========================================================================
class EMA:
    def __init__(self, named_params: dict[str, torch.Tensor], decay: float) -> None:
        self.decay  = decay
        self.shadow = {k: v.detach().clone() for k, v in named_params.items()}

    @torch.no_grad()
    def update(self, named_params: dict[str, torch.Tensor]) -> None:
        for k, v in named_params.items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.shadow

    def load_state_dict(self, sd: dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.clone() for k, v in sd.items()}


# ===========================================================================
# Checkpoint I/O
# ===========================================================================
def save_checkpoint(
    step:       int,
    unet:       torch.nn.Module,
    vae:        torch.nn.Module,
    optimizer:  torch.optim.Optimizer,
    ema:        EMA,
    output_dir: Path,
    dirname:    Optional[str] = None,
) -> Path:
    name = dirname if dirname is not None else f"checkpoint_step{step:07d}"
    ckpt = output_dir / name
    (ckpt / "ema_weights").mkdir(parents=True, exist_ok=True)

    torch.save(unet.state_dict(),      ckpt / "unet.pt")
    torch.save(vae.state_dict(),       ckpt / "vae.pt")
    torch.save(optimizer.state_dict(), ckpt / "optim.pt")
    torch.save(ema.state_dict(),       ckpt / "ema.pt")

    # EMA-applied UNet state dict (substitute EMA values into trainable slots).
    full = unet.state_dict()
    for k, v in ema.state_dict().items():
        if k in full:
            full[k] = v
    torch.save(full, ckpt / "ema_weights" / "unet_ema.pt")
    return ckpt


def load_checkpoint(
    resume_path: Path,
    unet:        torch.nn.Module,
    vae:         torch.nn.Module,
    optimizer:   torch.optim.Optimizer,
    ema:         EMA,
    map_location: str = "cuda",
) -> int:
    resume_path = Path(resume_path)
    unet.load_state_dict(torch.load(resume_path / "unet.pt",  map_location=map_location))
    vae.load_state_dict (torch.load(resume_path / "vae.pt",   map_location=map_location))
    optimizer.load_state_dict(torch.load(resume_path / "optim.pt", map_location=map_location))
    ema.load_state_dict(torch.load(resume_path / "ema.pt",    map_location=map_location))

    # Recover step number from directory name (e.g. "checkpoint_step0001000").
    name = resume_path.name
    if name.startswith("checkpoint_step"):
        try:
            return int(name[len("checkpoint_step"):])
        except ValueError:
            pass
    print(f"  (cannot parse step from {name!r}; resuming at step 0)", file=sys.stderr)
    return 0


# ===========================================================================
# Infinite DataLoader  (re-shuffles each epoch)
# ===========================================================================
def infinite_loader(loader: DataLoader) -> Iterator:
    while True:
        for batch in loader:
            yield batch


# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune SD 1.5 with DCT eligibility reward + anchor.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--steps",            type=int,   default=TRAIN_STEPS)
    parser.add_argument("--checkpoint_every", type=int,   default=CHECKPOINT_EVERY)
    parser.add_argument("--lambda_mid",       type=float, default=LAMBDA_MID_DEFAULT)
    parser.add_argument("--lambda_anchor",    type=float, default=LAMBDA_ANCHOR_DEFAULT)
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--use_anchor", dest="use_anchor", action="store_true",  default=USE_ANCHOR_DEFAULT)
    grp.add_argument("--no_anchor",  dest="use_anchor", action="store_false")
    parser.add_argument("--mid_score_fn",     choices=list(SCORE_FNS.keys()), default=MID_SCORE_FN_DEFAULT)
    parser.add_argument("--lora_rank",        type=int,   default=LORA_RANK)
    parser.add_argument("--use_wandb",        action="store_true")
    parser.add_argument("--resume",           type=str,   default=None)

    # Extra knobs (not in the contract but useful)
    parser.add_argument("--lr",            type=float, default=LR)
    parser.add_argument("--warmup_steps",  type=int,   default=WARMUP_STEPS)
    parser.add_argument("--batch_size",    type=int,   default=BATCH_SIZE)
    parser.add_argument("--grad_accum",    type=int,   default=GRAD_ACCUM_STEPS)
    parser.add_argument("--num_workers",   type=int,   default=2)
    parser.add_argument(
        "--output_dir", type=str,
        default=r"D:\image_stego_runs\finetune_output",
        help="Where to write checkpoints. Default is on D: to spare C: space.",
    )
    args = parser.parse_args()

    prevent_sleep()
    try:
        run_training(args)
    finally:
        allow_sleep()


def run_training(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("ERROR: CUDA not available. SD 1.5 fine-tuning is CUDA-only.", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints will be written to: {output_dir}")
    weight_dtype = torch.bfloat16

    # -----------------------------------------------------------------------
    # Load all SD 1.5 components
    # -----------------------------------------------------------------------
    print(f"Loading SD 1.5 components from {SD15_MODEL_ID} ...")
    tokenizer    = CLIPTokenizer.from_pretrained(SD15_MODEL_ID, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(SD15_MODEL_ID, subfolder="text_encoder")
    vae          = AutoencoderKL.from_pretrained(SD15_MODEL_ID, subfolder="vae")
    unet         = UNet2DConditionModel.from_pretrained(SD15_MODEL_ID, subfolder="unet")
    unet_anchor  = (
        UNet2DConditionModel.from_pretrained(SD15_MODEL_ID, subfolder="unet")
        if args.use_anchor else None
    )
    scheduler    = DDPMScheduler.from_pretrained(SD15_MODEL_ID, subfolder="scheduler")

    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    if unet_anchor is not None:
        unet_anchor.requires_grad_(False)

    # -----------------------------------------------------------------------
    # Wrap UNet with PEFT LoRA  (only LoRA params will train)
    # -----------------------------------------------------------------------
    lora_cfg = LoraConfig(
        r              = args.lora_rank,
        lora_alpha     = LORA_ALPHA,
        target_modules = LORA_TARGETS,
        lora_dropout   = LORA_DROPOUT,
        bias           = "none",
    )
    unet = get_peft_model(unet, lora_cfg)
    unet.print_trainable_parameters()

    # -----------------------------------------------------------------------
    # Precision: frozen modules -> bf16 ; LoRA params -> fp32 (AdamW stability)
    # -----------------------------------------------------------------------
    text_encoder.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    unet.to(device)  # cast per-param below
    if unet_anchor is not None:
        unet_anchor.to(device, dtype=weight_dtype)
        unet_anchor.eval()

    for _, param in unet.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)   # LoRA in fp32
        else:
            param.data = param.data.to(weight_dtype)    # base in bf16

    text_encoder.eval()
    vae.eval()
    unet.train()

    # Gradient checkpointing to keep activation memory in check.
    try:
        unet.enable_gradient_checkpointing()
    except Exception:
        if hasattr(unet, "base_model"):
            unet.base_model.enable_gradient_checkpointing()

    # -----------------------------------------------------------------------
    # DCT helpers on device
    # -----------------------------------------------------------------------
    D        = build_dct_basis().to(device, dtype=torch.float32)
    Q        = torch.from_numpy(JPEG_Q_NP).to(device, dtype=torch.float32)
    mid_mask = (Q >= 8.0).to(torch.float32)
    assert int(mid_mask.sum().item()) == 39, "stable-mid mask must have 39 positions"
    score_fn = SCORE_FNS[args.mid_score_fn]

    # -----------------------------------------------------------------------
    # Dataset / dataloader
    # -----------------------------------------------------------------------
    if not MANIFEST_FILE.exists():
        print(f"ERROR: {MANIFEST_FILE} not found. Run preprocess.py first.", file=sys.stderr)
        sys.exit(1)
    dataset = ProcessedDataset(PROCESSED_DIR, MANIFEST_FILE)
    print(f"Dataset: {len(dataset)} samples")
    if len(dataset) == 0:
        print("ERROR: empty manifest.", file=sys.stderr)
        sys.exit(1)

    loader = DataLoader(
        dataset,
        batch_size         = args.batch_size,
        shuffle            = True,
        num_workers        = args.num_workers,
        pin_memory         = True,
        drop_last          = True,
        persistent_workers = (args.num_workers > 0),
    )
    data_iter = infinite_loader(loader)

    # -----------------------------------------------------------------------
    # Optimiser + LR schedule + EMA
    # -----------------------------------------------------------------------
    trainable_named = {n: p for n, p in unet.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(
        list(trainable_named.values()),
        lr           = args.lr,
        weight_decay = WEIGHT_DECAY,
        betas        = (0.9, 0.999),
    )

    def lr_lambda(step: int) -> float:
        if args.warmup_steps > 0 and step < args.warmup_steps:
            return float(step) / float(args.warmup_steps)
        return 1.0

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    ema = EMA(trainable_named, EMA_DECAY)

    # -----------------------------------------------------------------------
    # Optional wandb
    # -----------------------------------------------------------------------
    wandb_run = None
    if args.use_wandb:
        try:
            import wandb  # type: ignore[import]
            wandb_run = wandb.init(project="sd15-texture-finetune", config=vars(args))
        except Exception as exc:
            print(f"wandb disabled ({exc})", file=sys.stderr)

    # -----------------------------------------------------------------------
    # Resume
    # -----------------------------------------------------------------------
    start_step = 0
    if args.resume:
        print(f"Resuming from {args.resume} ...")
        start_step = load_checkpoint(Path(args.resume), unet, vae, optimizer, ema, map_location=device)
        for _ in range(start_step):
            lr_scheduler.step()
        print(f"Resumed at step {start_step}")

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    first_caption_logged = False
    optimizer.zero_grad(set_to_none=True)
    t_start = time.time()

    pbar = tqdm(
        range(start_step, args.steps),
        initial = start_step,
        total   = args.steps,
        desc    = "Training",
    )

    for step in pbar:
        step_idx = step + 1   # 1-based for logging / checkpoints

        l_diff_acc = 0.0
        l_anch_acc = 0.0
        elig_acc   = 0.0
        l_tot_acc  = 0.0

        for _ in range(args.grad_accum):
            images, _dct_pre, captions = next(data_iter)
            images = images.to(device, dtype=weight_dtype, non_blocking=True)

            # Sanity log: first real caption ever seen.
            if not first_caption_logged:
                tqdm.write(f"[sanity] First caption seen: {captions[0]!r}")
                first_caption_logged = True

            # --- Tokenise + encode captions (frozen text encoder) ---
            with torch.no_grad():
                tokens = tokenizer(
                    list(captions),
                    padding       = "max_length",
                    truncation    = True,
                    max_length    = tokenizer.model_max_length,
                    return_tensors= "pt",
                )
                input_ids  = tokens.input_ids.to(device)
                caption_emb = text_encoder(input_ids)[0].to(dtype=weight_dtype)

            # --- VAE encode (frozen) ---
            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor

            # --- Sample timesteps + noise ---
            B = latents.shape[0]
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps, (B,),
                device=device, dtype=torch.long,
            )
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            # --- Trainable UNet forward (under autocast) ---
            with torch.autocast(device_type="cuda", dtype=weight_dtype):
                noise_pred = unet(
                    noisy_latents, timesteps,
                    encoder_hidden_states=caption_emb,
                ).sample

            # --- Anchor UNet forward (frozen, no grad) ---
            if unet_anchor is not None:
                with torch.no_grad():
                    noise_pred_anchor = unet_anchor(
                        noisy_latents, timesteps,
                        encoder_hidden_states=caption_emb,
                    ).sample
                l_anch = F.mse_loss(noise_pred.float(), noise_pred_anchor.detach().float())
            else:
                l_anch = torch.zeros((), device=device, dtype=torch.float32)

            # --- Diffusion loss (epsilon-prediction MSE) ---
            l_diff = F.mse_loss(noise_pred.float(), noise.float())

            # --- x_hat_0 in latent space ---
            alpha_bar = scheduler.alphas_cumprod.to(device, dtype=torch.float32)[timesteps]
            alpha_bar = alpha_bar.view(-1, 1, 1, 1)
            nlf = noisy_latents.float()
            npf = noise_pred.float()
            x0_latent = (nlf - torch.sqrt(1.0 - alpha_bar) * npf) / torch.sqrt(alpha_bar)

            # --- Decode to pixel space (gradient flows through the decoder) ---
            x0_lat_dec = (x0_latent / vae.config.scaling_factor).to(dtype=weight_dtype)
            x_pixel    = vae.decode(x0_lat_dec).sample.float()        # (B, 3, 256, 256)

            # --- Mid-band reward ---
            elig, l_mid_reward = compute_mid_reward(x_pixel, D, Q, mid_mask, score_fn)

            # --- Total loss ---
            loss = l_diff + args.lambda_mid * l_mid_reward + args.lambda_anchor * l_anch
            (loss / args.grad_accum).backward()

            l_diff_acc += l_diff.detach().item()
            l_anch_acc += l_anch.detach().item()
            elig_acc   += elig.detach().item()
            l_tot_acc  += loss.detach().item()

        # --- Gradient clip, optimiser step, LR step, EMA update ---
        torch.nn.utils.clip_grad_norm_(list(trainable_named.values()), GRAD_CLIP)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        ema.update(trainable_named)

        # --- Logging ---
        l_diff_avg = l_diff_acc / args.grad_accum
        l_anch_avg = l_anch_acc / args.grad_accum
        elig_avg   = elig_acc   / args.grad_accum
        l_tot_avg  = l_tot_acc  / args.grad_accum

        pbar.set_postfix({
            "loss":   f"{l_tot_avg:.4f}",
            "l_diff": f"{l_diff_avg:.4f}",
            "elig":   f"{elig_avg:.2f}",
            "l_anch": f"{l_anch_avg:.4f}",
        })

        if wandb_run is not None:
            wandb_run.log({
                "step":                       step_idx,
                "loss/total":                 l_tot_avg,
                "loss/diffusion":             l_diff_avg,
                "loss/anchor":                l_anch_avg,
                "metric/eligibility_per_block": elig_avg,
                "lr":                         lr_scheduler.get_last_lr()[0],
            }, step=step_idx)

        # --- Checkpoint ---
        if args.checkpoint_every > 0 and step_idx % args.checkpoint_every == 0:
            ckpt_path = save_checkpoint(step_idx, unet, vae, optimizer, ema, output_dir)
            tqdm.write(f"  Saved checkpoint: {ckpt_path}")

    pbar.close()
    final_path = save_checkpoint(args.steps, unet, vae, optimizer, ema, output_dir, dirname="final")
    elapsed = time.time() - t_start
    print(f"\nTraining complete in {elapsed/3600:.2f} h. Final checkpoint: {final_path}")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
