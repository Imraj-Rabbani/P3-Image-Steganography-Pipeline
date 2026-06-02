# Texture-Biased Stable Diffusion for DCT Steganography

Fine-tune Stable Diffusion 1.5 to generate **texture-rich cover images** whose
DCT coefficients are friendly to a JPEG-domain steganography scheme (DSTG), then
measure the result with a full embed → PNG round-trip → extract loop.

> **The idea.** Steganography that hides bits in JPEG DCT coefficients needs
> covers with many *embeddable* coefficients (magnitudes large enough that
> flipping the LSB survives quantization). Out-of-the-box SD 1.5 produces smooth
> images with little mid-band energy. A small LoRA adapter, trained with a
> DCT-eligibility reward, biases generations toward textured images with far
> more eligible coefficients — without breaking text-to-image fidelity.

For the full narrative, design rationale, and recorded experimental results, see
[`explanation.txt`](explanation.txt).

---

## Pipeline

```
filter_dataset.py → preprocess.py → check.py → finetune.py → validate.py
                                                                  │
                                                        sweep_checkpoints.py
```

| Stage | Script | What it does |
|------:|--------|--------------|
| 1 | `filter_dataset.py` | Scan a HuggingFace dataset, keep images with many embeddable mid-band DCT coefficients → `filtered_dataset/` |
| 2 | `preprocess.py` | Centre-crop, resize to 256×256, precompute block DCT → `processed_dataset/` |
| 3 | `check.py` | Pre-flight checks (dataset integrity, CUDA, dataloader throughput) |
| 4 | `finetune.py` | Train a rank-4 LoRA with `L_diffusion + λ_mid·L_mid_reward + λ_anchor·L_anchor` → `finetune_output/` |
| 5 | `validate.py` | Baseline vs fine-tuned head-to-head: generate covers, embed/extract, report metrics → `validation_output/` |
| 5b | `sweep_checkpoints.py` | Re-run validation across many checkpoints to see how metrics evolve over training |

### Supporting modules
- **`dct.py`** — the DSTG engine. Canonical-Huffman compression + DCT-domain
  embed/extract. Used by `validate.py` and `sweep_checkpoints.py`, and runnable
  standalone (see below).
- **`Steganalyzer_fixed.py`** — an SRNet-style DCT-stego detector used to score
  how detectable the stego images are.

---

## The DSTG engine (`dct.py`)

Embeds a UTF-8 secret into the **Y (luma) channel** so the payload survives a
BGR ↔ YCrCb / PNG round-trip losslessly. Pipeline per image:

1. **Canonical Huffman compression** of the secret *before* embedding (and
   decompression after extraction). The compressed blob is fully
   self-describing — it carries the original length and the per-symbol code
   lengths — so the extractor rebuilds the identical code with **no sidecar
   file or key**.
2. 8×8 block DCT on the level-shifted Y channel, JPEG luminance quantization.
3. A 64-bit **CRC-protected header** in blocks `(0,0)` and `(0,1)`.
4. `K` adaptive eligible mid-band positions per payload block (`K` derived from
   the cover's mean eligible-coefficient count).
5. **LSB-match** writes with min-distortion choice and a never-below-`|q|≥2`
   guarantee, then a verify-fix + exact-Y repair pass so extraction is exact.

> ⚠️ Canonical Huffman needs a per-symbol code-length table, which is pure
> overhead for short, high-entropy secrets — random data can *expand* slightly.
> Natural-language secrets compress. Capacity is large enough that either fits.

### Run the engine standalone

```bash
python dct.py                      # prompts for a secret, embeds into the demo cover
python dct.py path/to/message.txt  # embeds the file contents
```

It writes `stego_adaptive.png`, then verifies an in-memory and a PNG-reload
round-trip.

Programmatic use:

```python
import cv2, dct

cover = cv2.imread("cover.png")           # BGR uint8, dims multiple of 8
stego = dct.embed(cover, "my secret")
cv2.imwrite("stego.png", stego)

message = dct.extract(cv2.imread("stego.png"))
```

---

## Validation

```bash
python validate.py --finetuned_dir ./finetune_output/final
```

Generates **2 seeds for every prompt** in `validate.py`'s `PROMPTS` list, for
both the baseline and fine-tuned models, embeds a 512-bit secret in each, and
writes to `validation_output/`:

- `metrics.txt` — aggregate + per-prompt table, plus a PASS/FAIL verdict
- `metrics.csv` — one row per `(model, prompt, seed)`
- `comparison_grid.png` — one row per prompt (prompt · baseline · fine-tuned)
- `baseline/`, `finetuned/` — cover + stego PNGs

Key flags: `--finetuned_dir`, `--seed`, `--output_dir`, `--lora_rank`.
(`--num_images` is retained but ignored — the plan is always all prompts × 2 seeds.)

---

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Unix: source .venv/bin/activate

# Install a CUDA build of torch first (see pytorch.org), then:
pip install diffusers transformers peft accelerate \
            scipy scikit-image opencv-python-headless \
            matplotlib tqdm Pillow huggingface_hub
```

**Environment** (reference machine): NVIDIA RTX 4080 (16 GB), Windows 11,
Python 3.10+. Base model: `runwayml/stable-diffusion-v1-5`. Fine-tune
checkpoints are large (~3.5 GB each) — redirect `--output_dir` and `HF_HOME` to
a roomy drive.

---

## Repository notes

- Large/regenerable artifacts (`finetune_output/`, datasets, model weights) are
  git-ignored; `validation_output/` is **tracked** so results can be reviewed in
  the repo.
- `steganalyzer.pth` is force-tracked despite the global `*.pth` ignore.
