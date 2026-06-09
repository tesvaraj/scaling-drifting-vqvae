"""Generate paper-quality reconstruction comparison grids.

Loads EMA and Drift★ (no_pp_ste) checkpoints from the Modal volume, runs the same
fixed set of validation images through both models, and produces a labeled PNG grid.

Layout (per row = per dataset):
    [Original × N] | [EMA recon × N] | [Drift★ recon × N]

Usage:
    modal run experiments/scripts/make_paper_grid.py               # default: 4 datasets, N=8
    modal run experiments/scripts/make_paper_grid.py --n_images 6  # 6 images per group
    modal run experiments/scripts/make_paper_grid.py --out_dir figures/paper_grid
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import modal

# ---------- reuse app / image / volume from main harness ----------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.modal_app import app, image, volume

OUT_DIR = Path('figures/paper_grid')

# ---------- which runs to compare (seed0 throughout for consistency) ----------
# (dataset_key, label, phase_ema, run_id_ema, phase_drift, run_id_drift)
# dataset_key must match build_dataset() argument
CONFIGS = [
    (
        'cifar10', 'CIFAR-10',
        'phase1_convergence', 'cifar10_vanilla_ema_K512_seed0',
        'phase2b_confirmation', 'cifar10_drift_no_pp_ste_K512_seed0',
    ),
    (
        'cifar100', 'CIFAR-100',
        'phase_cifar100', 'cifar100_vanilla_ema_K512_seed0',
        'phase_cifar100', 'cifar100_drift_no_pp_ste_K512_seed0',
    ),
    (
        'stl10', 'STL-10',
        'phase_stl10', 'stl10_vanilla_ema_K512_seed0',
        'phase_stl10', 'stl10_drift_no_pp_ste_K512_seed0',
    ),
    (
        'tiny_imagenet', 'Tiny ImageNet',
        'phase_tiny_imagenet', 'tinyimagenet_vanilla_ema_K512_seed0',
        'phase_tiny_imagenet', 'tinyimagenet_drift_no_pp_ste_K512_seed0',
    ),
]


# ---------- remote function ----------

@app.function(image=image, gpu='L40S', volumes={'/vol': volume}, timeout=600)
def _make_grid_remote(
    configs: list,
    n_images: int = 8,
    img_size_override: int | None = None,  # force resize (for display), None = native
) -> dict[str, bytes]:
    """Returns {dataset_label: png_bytes} for each dataset, plus 'combined'."""
    import io
    import sys
    import math

    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from torch.utils.data import DataLoader

    sys.path.insert(0, '/root/scaling_drifting_vqvae')
    from experiments.prior import load_vqvae_frozen
    from experiments.data import build_dataset

    device = 'cuda'
    PADDING = 3   # pixels between images
    LABEL_W = 90  # left-side label column width
    FONT_SIZE = 14

    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', FONT_SIZE)
        font_small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', FONT_SIZE - 2)
    except Exception:
        font = ImageFont.load_default()
        font_small = font

    def to_uint8(t):
        """(B, C, H, W) in [-1, 1] → list of (H, W, C) uint8 arrays."""
        arr = ((t * 0.5 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy() * 255)
        return arr.astype(np.uint8)

    def make_row_image(images_uint8, target_hw: tuple[int, int]) -> Image.Image:
        """Stack N images in a horizontal strip; resize each to target_hw."""
        th, tw = target_hw
        strip = Image.new('RGB', (n_images * (tw + PADDING) + PADDING, th + 2 * PADDING),
                          color=(240, 240, 240))
        for i, img_arr in enumerate(images_uint8):
            img = Image.fromarray(img_arr).resize((tw, th), Image.BICUBIC)
            strip.paste(img, (PADDING + i * (tw + PADDING), PADDING))
        return strip

    results = {}
    section_imgs = []

    for (dataset_key, dataset_label, phase_ema, run_id_ema, phase_drift, run_id_drift) in configs:
        print(f'  [{dataset_label}] loading checkpoints …')
        ema_ckpt = f'/vol/runs/{phase_ema}/{run_id_ema}/checkpoints/final.pt'
        drift_ckpt = f'/vol/runs/{phase_drift}/{run_id_drift}/checkpoints/final.pt'

        ema_model, _ = load_vqvae_frozen(ema_ckpt, device)
        drift_model, _ = load_vqvae_frozen(drift_ckpt, device)

        # Fixed val set — no shuffle, so always the same images
        ds = build_dataset(dataset_key, '/vol/data')
        loader = DataLoader(ds.val, batch_size=n_images, shuffle=False, num_workers=2)
        batch = next(iter(loader))
        x = (batch[0] if isinstance(batch, (list, tuple)) else batch)[:n_images].to(device)

        with torch.no_grad():
            ema_recon, _, _ = ema_model(x)
            drift_recon, _, _ = drift_model(x)
        ema_recon = ema_recon.clamp(-1, 1)
        drift_recon = drift_recon.clamp(-1, 1)

        orig_np = to_uint8(x)
        ema_np = to_uint8(ema_recon)
        drift_np = to_uint8(drift_recon)

        # Native image size; optionally upscale small images (e.g. CIFAR 32→96 for visibility)
        native_h, native_w = orig_np[0].shape[:2]
        if img_size_override:
            disp_h = disp_w = img_size_override
        elif native_h <= 32:
            disp_h = disp_w = 96   # 3× upscale for CIFAR
        else:
            disp_h = disp_w = native_h

        row_orig = make_row_image(orig_np, (disp_h, disp_w))
        row_ema = make_row_image(ema_np, (disp_h, disp_w))
        row_drift = make_row_image(drift_np, (disp_h, disp_w))

        row_w = row_orig.width
        row_h = row_orig.height

        SEP = 2  # pixels between rows within a section

        # Compose 3 rows + dataset label on the left
        total_h = row_h * 3 + SEP * 2
        section = Image.new('RGB', (LABEL_W + row_w, total_h), color=(255, 255, 255))

        # Paste rows
        section.paste(row_orig, (LABEL_W, 0))
        section.paste(row_ema, (LABEL_W, row_h + SEP))
        section.paste(row_drift, (LABEL_W, (row_h + SEP) * 2))

        # Left-side dataset label (vertical text via rotation)
        draw = ImageDraw.Draw(section)
        lbl_img = Image.new('RGB', (total_h, LABEL_W), color=(255, 255, 255))
        lbl_draw = ImageDraw.Draw(lbl_img)
        lbl_draw.text((4, 4), dataset_label, fill=(30, 30, 30), font=font)
        lbl_img = lbl_img.rotate(90, expand=True)
        section.paste(lbl_img, (0, 0))

        # Row labels (right of the dataset label column, above each row)
        METHOD_LABELS = ['Original', 'EMA VQ', 'Drift★']
        for idx, lbl in enumerate(METHOD_LABELS):
            y_center = idx * (row_h + SEP) + row_h // 2
            # small label in the label column below the dataset name
            draw.text((4, y_center - FONT_SIZE // 2), lbl, fill=(80, 80, 80), font=font_small)

        # Per-dataset individual PNG
        buf = io.BytesIO()
        section.save(buf, format='PNG', dpi=(150, 150))
        results[dataset_label] = buf.getvalue()
        section_imgs.append(section)

        print(f'  [{dataset_label}] done  ({disp_h}×{disp_w} px per image)')

    # --- combined: stack all sections vertically with a separator line ---
    SECTION_SEP = 8
    total_w = max(s.width for s in section_imgs)
    total_h = sum(s.height for s in section_imgs) + SECTION_SEP * (len(section_imgs) - 1)
    combined = Image.new('RGB', (total_w, total_h), color=(220, 220, 220))
    y = 0
    for sec in section_imgs:
        combined.paste(sec, (0, y))
        y += sec.height + SECTION_SEP

    # Top legend (method labels across the three groups)
    # This requires knowing column widths — add a small header bar
    header_h = 24
    header = Image.new('RGB', (total_w, header_h), color=(255, 255, 255))
    hdraw = ImageDraw.Draw(header)
    row_w = section_imgs[0].width - LABEL_W
    col_w = row_w // 3
    labels_top = ['Original', 'EMA VQ (K=512)', 'Drift★ (K=512)']
    for ci, lbl in enumerate(labels_top):
        x_center = LABEL_W + ci * col_w + col_w // 2
        hdraw.text((x_center - 40, 5), lbl, fill=(20, 20, 20), font=font)
    combined_with_header = Image.new('RGB', (total_w, total_h + header_h), color=(220, 220, 220))
    combined_with_header.paste(header, (0, 0))
    combined_with_header.paste(combined, (0, header_h))

    buf = io.BytesIO()
    combined_with_header.save(buf, format='PNG', dpi=(150, 150))
    results['combined'] = buf.getvalue()
    return results


# ---------- local entrypoint ----------

@app.local_entrypoint()
def main(
    n_images: int = 8,
    out_dir: str = str(OUT_DIR),
    img_size: int = 0,  # 0 = auto (96 for CIFAR-32, native otherwise)
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    img_size_arg = img_size if img_size > 0 else None

    print(f'Generating paper reconstruction grids …')
    print(f'  n_images={n_images}  out={out}')
    print(f'  Datasets: {[c[1] for c in CONFIGS]}')
    print()

    results = _make_grid_remote.remote(CONFIGS, n_images=n_images,
                                       img_size_override=img_size_arg)

    for label, png_bytes in results.items():
        fname = label.lower().replace(' ', '_').replace('/', '_') + '.png'
        fpath = out / fname
        fpath.write_bytes(png_bytes)
        print(f'  saved → {fpath}  ({len(png_bytes)//1024} KB)')

    print()
    print(f'Done. Open {out}/combined.png for the full paper figure.')
    print(f'Individual dataset PNGs are also in {out}/ for cropping.')
