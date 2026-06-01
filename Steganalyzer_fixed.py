"""
Steganalyzer_fixed.py    SRNet-style DCT-stego detector (corrected)

Key fixes vs the original:
  1. load_state_dict is checked: missing/unexpected keys are PRINTED and, by
     default, a mismatch raises instead of silently leaving layers random.
  2. The SRM filter bank now contains 30 DISTINCT high-pass kernels (the original
     padded 28 of them with identity filters, which is not steganalysis at all).
     NOTE: if your checkpoint SAVED its srm.w buffer, we load that instead, so the
     detector uses exactly the filters it was trained with.
  3. cv2.resize is removed.  Steganographic signal lives in LSB-level changes that
     resampling destroys; the detector must see the exact pixels.  We assert size.
  4. Preprocessing is centralised in one place so you can match training exactly.
  5. A batch evaluator computes accuracy + AUC over a folder of cover/stego pairs 
     a single image is not a result.

You MUST confirm two things for the numbers to be trustworthy (see __main__):
  - the checkpoint loads with NO missing keys (else layers are random), and
  - preprocess() matches how the model was TRAINED (channel, scale, grayscale).
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_DIR = os.path.dirname(os.path.abspath(__file__))


# =========================
# SRM FILTERS  (30 distinct high-pass kernels)
# =========================
def _srm_kernels():
    """A small but genuine bank of 30 high-pass residual kernels (5x5).

    This is a representative SRM-style set (first/second-order differences,
    edge and spot detectors, KB kernel, etc.).  It is NOT guaranteed to match
    the exact bank your checkpoint was trained with  if the checkpoint stored
    its own srm.w buffer, load_model() below will overwrite these with it.
    """
    Fk = []
    z = lambda: np.zeros((5, 5), dtype=np.float64)

    # --- first-order horizontal / vertical / diagonal differences ---
    f = z(); f[2, 1:4] = [-1, 2, -1];                Fk.append(f / 2.0)
    f = z(); f[1:4, 2] = [-1, 2, -1];                Fk.append(f / 2.0)
    f = z(); f[1, 1] = -1; f[2, 2] = 2; f[3, 3] = -1; Fk.append(f / 2.0)
    f = z(); f[1, 3] = -1; f[2, 2] = 2; f[3, 1] = -1; Fk.append(f / 2.0)

    # --- second-order differences ---
    f = z(); f[2, 1:4] = [1, -2, 1];                 Fk.append(f / 2.0)
    f = z(); f[1:4, 2] = [1, -2, 1];                 Fk.append(f / 2.0)

    # --- edge 3x3 (SQUARE) kernels ---
    e1 = np.array([[-1, 2, -1], [2, -4, 2], [0, 0, 0]], dtype=np.float64) / 4.0
    for k in range(4):  # 4 rotations
        f = z(); f[1:4, 1:4] = np.rot90(e1, k); Fk.append(f)

    # --- KB (square 3x3) kernel ---
    kb = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float64) / 4.0
    f = z(); f[1:4, 1:4] = kb; Fk.append(f)

    # --- 5x5 second-order 'edge5' style ---
    e5 = np.array([
        [-1,  2,  -2,  2, -1],
        [ 2, -6,   8, -6,  2],
        [-2,  8, -12,  8, -2],
        [ 2, -6,   8, -6,  2],
        [-1,  2,  -2,  2, -1],
    ], dtype=np.float64) / 12.0
    Fk.append(e5)

    # --- simple directional gradients to round out the bank ---
    grads = [
        [[0, 0, 0], [-1, 1, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 1, -1], [0, 0, 0]],
        [[0, -1, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 1, 0], [0, -1, 0]],
    ]
    for g in grads:
        f = z(); f[1:4, 1:4] = np.array(g, dtype=np.float64); Fk.append(f)

    # --- fill the remainder with distinct random-but-zero-mean high-pass kernels ---
    rng = np.random.RandomState(0)
    while len(Fk) < 30:
        f = rng.randn(5, 5)
        f -= f.mean()          # zero-mean -> high-pass (kills the DC/cover content)
        f /= np.abs(f).sum()
        Fk.append(f)

    return torch.tensor(np.stack(Fk[:30])[:, None], dtype=torch.float32)


class SRM(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("w", _srm_kernels())

    def forward(self, x):
        # x is expected as (B,1,H,W) in [0,1]; SRM operates on the [0,255] residual.
        return torch.tanh(F.conv2d(x * 255.0, self.w, padding=2) / 3.0)


# =========================
# NETWORK BLOCKS  (unchanged structurally from your file)
# =========================
class SE(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(x).view(x.size(0), -1, 1, 1)
        return x * w


class Blk(nn.Module):
    def __init__(self, ci, co, se=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1, bias=False),  # 0
            nn.BatchNorm2d(co),                            # 1
            nn.ReLU(True),                                 # 2
            nn.ReLU(True),                                 # 3
            nn.Conv2d(co, co, 3, padding=1, bias=False),  # 4
            nn.BatchNorm2d(co)                             # 5
        )
        self.skip = nn.Identity() if ci == co else nn.Conv2d(ci, co, 1, bias=False)
        self.se = SE(co) if se else nn.Identity()

    def forward(self, x):
        return F.relu(self.se(self.net(x)) + self.skip(x), True)


def _dn(ci, co):
    return nn.Sequential(
        nn.Conv2d(ci, co, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(co),
        nn.ReLU(True)
    )


class SRNet(nn.Module):
    def __init__(self, nc=2, drop=0.7):
        super().__init__()
        self.srm = SRM()
        self.entry = nn.Sequential(
            nn.Conv2d(30, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.s1 = nn.Sequential(Blk(64, 64),            Blk(64, 64))
        self.d1 = _dn(64, 64)
        self.s2 = nn.Sequential(Blk(64, 128),           Blk(128, 128))
        self.d2 = _dn(128, 128)
        self.s3 = nn.Sequential(Blk(128, 256, se=True), Blk(256, 256, se=True))
        self.d3 = _dn(256, 256)
        self.s4 = nn.Sequential(Blk(256, 512, se=True), Blk(512, 512, se=True))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(drop),
            nn.Linear(512, nc)
        )

    def forward(self, x):
        x = self.srm(x)
        x = self.entry(x)
        x = self.d1(self.s1(x))
        x = self.d2(self.s2(x))
        x = self.d3(self.s3(x))
        x = self.s4(x)
        return self.head(x)


CLASS_NAMES = ["COVER", "DCT_STEGO"]


# =========================
# LOAD MODEL  (with diagnostics  this is the important fix)
# =========================
def load_model(path, allow_partial=False):
    """Load checkpoint and REPORT exactly what matched.

    If any layer is missing (would be left at random init) the function raises,
    unless allow_partial=True.  This is the difference between a real detector
    and one that silently runs on random weights.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SRNet().to(device)

    ckpt = torch.load(path, map_location=device)
    # checkpoints vary: try common layouts
    if isinstance(ckpt, dict) and "state" in ckpt:
        state = ckpt["state"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state = ckpt
    else:
        raise ValueError(f"Unrecognised checkpoint format. Top-level keys: "
                         f"{list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}")

    # strip a possible 'module.' prefix from DataParallel training
    state = { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }

    result = model.load_state_dict(state, strict=False)

    print(f"Device: {device}")
    print(f"Checkpoint tensors: {len(state)}")
    print(f"MISSING keys (left at RANDOM init): {len(result.missing_keys)}")
    if result.missing_keys:
        for k in result.missing_keys:
            print(f"    missing  {k}")
    print(f"UNEXPECTED keys (in ckpt, ignored): {len(result.unexpected_keys)}")
    if result.unexpected_keys:
        for k in result.unexpected_keys:
            print(f"    unused   {k}")

    # If the checkpoint stored the SRM filter bank, that overwrites our kernels.
    if any(k.endswith("srm.w") for k in state):
        print("NOTE: checkpoint provided srm.w  using the trained filter bank.")

    n_missing = len(result.missing_keys)
    # ignore srm.w in the missing count if SRM is a fixed buffer not in the ckpt
    real_missing = [k for k in result.missing_keys if not k.endswith("srm.w")]
    if real_missing and not allow_partial:
        raise RuntimeError(
            f"{len(real_missing)} model layers got NO weights and are RANDOM. "
            f"Predictions would be meaningless. Architecture likely does not match "
            f"the checkpoint. Set allow_partial=True only if you understand this. "
            f"First few: {real_missing[:6]}"
        )

    model.eval()
    if not real_missing:
        print("OK: every model layer received trained weights.")
    return model, device


# =========================
# PREPROCESS  (must match TRAINING  edit to match how you trained)
# =========================
def preprocess(img_bgr, expect_size=256, channel="gray"):
    """Convert a BGR uint8 image to the model input tensor (1,1,H,W) in [0,1].

    IMPORTANT: this MUST match training.  The SRM layer above averages to one
    channel internally only if you pass 3 channels; here we pass a single
    channel directly, so pick the SAME channel you trained on:
      channel='gray' -> ITU-R grayscale   (cv2 COLOR_BGR2GRAY)
      channel='y'    -> Y of YCrCb         (matches your embedding channel)
      channel='green'-> green channel only
    NO resizing: steganographic signal must not be resampled.
    """
    h, w = img_bgr.shape[:2]
    if (h, w) != (expect_size, expect_size):
        raise ValueError(
            f"Image is {w}x{h}, expected {expect_size}x{expect_size}. "
            f"Do NOT resize (it destroys the stego signal)  use matching covers."
        )
    if channel == "gray":
        ch = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    elif channel == "y":
        ch = cv2.split(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb))[0]
    elif channel == "green":
        ch = img_bgr[:, :, 1]
    else:
        raise ValueError("channel must be 'gray', 'y', or 'green'")
    x = ch.astype(np.float32) / 255.0
    return torch.from_numpy(x)[None, None]  # (1,1,H,W)


# =========================
# INFERENCE  (single image)
# =========================
@torch.no_grad()
def detect(model, img_path, device, channel="gray"):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image not found: {img_path}")
    x = preprocess(img, channel=channel).to(device)
    out = model(x)
    prob = torch.softmax(out, dim=1)[0].cpu().numpy()
    pred = int(np.argmax(prob))
    print(f"\n{os.path.basename(img_path)} -> {CLASS_NAMES[pred]} {prob[pred]*100:.2f}% "
          f"(cover={prob[0]*100:.1f}%, stego={prob[1]*100:.1f}%)")
    return pred, prob


# =========================
# BATCH EVALUATION  (the real metric: accuracy + AUC over a test set)
# =========================
@torch.no_grad()
def evaluate(model, cover_dir, stego_dir, device, channel="gray"):
    """Run over matched folders and report accuracy + ROC-AUC.

    Expects cover_dir and stego_dir to contain the SAME filenames (cover[i] and
    its stego counterpart share a name). A detector that cannot beat AUC~0.5 is
    being EVADED by your method  that is the headline number for the thesis.
    """
    try:
        from sklearn.metrics import roc_auc_score, accuracy_score
    except ImportError:
        roc_auc_score = accuracy_score = None

    names = sorted(f for f in os.listdir(stego_dir)
                   if f.lower().endswith((".png", ".bmp", ".ppm", ".tif", ".tiff")))
    y_true, y_score, y_pred = [], [], []
    for nm in names:
        cp = os.path.join(cover_dir, nm)
        sp = os.path.join(stego_dir, nm)
        if not os.path.exists(cp):
            continue
        for path, label in ((cp, 0), (sp, 1)):
            img = cv2.imread(path)
            if img is None:
                continue
            x = preprocess(img, channel=channel).to(device)
            prob = torch.softmax(model(x), dim=1)[0].cpu().numpy()
            y_true.append(label)
            y_score.append(float(prob[1]))      # P(stego)
            y_pred.append(int(prob[1] >= 0.5))

    n = len(y_true)
    if n == 0:
        print("No matched cover/stego pairs found  check the folder paths/names.")
        return None

    y_true = np.array(y_true); y_score = np.array(y_score); y_pred = np.array(y_pred)
    acc = float((y_true == y_pred).mean())
    print(f"\n==== BATCH RESULT over {n} images "
          f"({(y_true==0).sum()} cover / {(y_true==1).sum()} stego) ====")
    print(f"  Detection accuracy : {acc*100:.2f}%   (50% = random = undetectable)")
    if roc_auc_score is not None and len(set(y_true)) == 2:
        auc = roc_auc_score(y_true, y_score)
        print(f"  ROC-AUC            : {auc:.4f}   (0.50 = perfect evasion)")
    # cover/stego mean P(stego)  a sanity check that the model discriminates at all
    print(f"  mean P(stego|cover): {y_score[y_true==0].mean():.3f}")
    print(f"  mean P(stego|stego): {y_score[y_true==1].mean():.3f}")
    return acc


# =========================
# RUN
# =========================
if __name__ == "__main__":
    CKPT = os.path.join(_DIR, "steganalyzer.pth")

    COVER_PATH = 'validation_output\\baseline\\cover_p2_s46.png'
    STEGO_PATH = 'validation_output\\baseline\\stego_p2_s46.png'

    # Load the trained detector (raises if architecture does not match the ckpt)
    model, device = load_model(CKPT, allow_partial=False)

    # Try all three channels. Whichever the detector was TRAINED on is the one to
    # trust. Compare cover vs stego: if both get a similar stego%, the detector
    # cannot tell them apart (i.e. your method evades it).
    for ch in ["gray", "y", "green"]:
        print(f"\n================  CHANNEL = {ch}  ================")
        print("COVER image:")
        detect(model, COVER_PATH, device, channel=ch)
        print("STEGO image:")
        detect(model, STEGO_PATH, device, channel=ch)
