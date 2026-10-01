#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract the tissue region from raw ultrasound frames (inference)
================================================================

What this script does
    Takes raw ultrasound frames, runs the trained U-Net over them, and
    writes out the tissue region as a crop — replacing what would
    otherwise be done by hand.

This is the preprocessing stage that sits in front of a downstream
lymph node classification model.

Requires
    A trained checkpoint, e.g. models/unet_tissue_v1.pt

Usage
    # a single image
    python src/extract_tissue.py --input path/to/image.png

    # a directory (searched recursively)
    python src/extract_tissue.py --input path/to/folder

    # also write visual check images
    python src/extract_tissue.py --input ... --save-overlay

Outputs
    crops/        tissue crops — feed these to the classifier
    masks/        binary masks
    overlays/     image with the mask drawn on top (--save-overlay only)
    manifest.csv  which crop came from which source frame

A note on the number of outputs
    Frames from a dual-pane scanner contain two panels. Each panel is
    located separately, so one frame usually yields *two* crops:
        2182.png  ->  2182_1.png , 2182_2.png
    Single-pane frames yield one crop.
"""

import argparse
import csv
import glob
import os

import cv2
import numpy as np
import torch

# Reuse the architecture from the training script so it is defined once.
from train_unet import UNet


IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


# ---------------------------------------------------------------------------
# Loading the model
# ---------------------------------------------------------------------------

def load_model(ckpt_path, device):
    """Load a trained checkpoint written by train_unet.py."""
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = UNet(base=ck.get('base', 24))
    model.load_state_dict(ck['model'])
    model.eval().to(device)
    info = {
        'img_h': ck.get('img_h', 384),
        'img_w': ck.get('img_w', 512),
        'epoch': ck.get('epoch', '?'),
        'val_dice': ck.get('val_dice', None),
    }
    return model, info


# ---------------------------------------------------------------------------
# Running the model on one image
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_mask(model, img_bgr, device, img_h, img_w, threshold=0.5):
    """
    Input  : a raw frame of any size
    Output : a 0/1 mask at the *original* image size

    The network runs at a fixed small resolution; the result is then
    scaled back up.
    """
    H, W = img_bgr.shape[:2]

    small = cv2.resize(img_bgr, (img_w, img_h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    x = torch.from_numpy(gray)[None, None].to(device)

    prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()

    # Upscale the *probability map*, not the thresholded mask, so the
    # boundary does not come out stair-stepped.
    prob = cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR)
    return (prob > threshold).astype(np.uint8)


def find_split_column(gray, region_mask, box):
    """
    Return the column at which to separate two side-by-side panels:
    the midpoint of the region.

    Why not look for a dark divider line?
        We tried, and measured it on real data: brightness on the
        divider is 1.01x that of the columns either side — i.e. no
        difference at all. The dark seam visible to the eye exists only
        in the black region below the scan, not within the tissue band.
        There is simply no image signal to key on.

    Why does the midpoint work?
        The two panels are always equal halves of the scan area.
        Measured over 233 frames: the true divider sits at a median of
        x=695, while the region midpoint lands around x=705 — an error
        of roughly 10 px. The gap between the two operator crops has a
        median width of 37 px, so a 10 px error still falls inside that
        gap and does not cut into either panel's tissue.
    """
    x, y, w, h = box
    return x + w // 2


def split_regions(mask, gray=None, min_area_ratio=0.01, split_width_frac=0.55):
    """
    Break the mask into separate regions — one per panel.

    Two steps:

    1) Find disconnected regions (connected components).

    2) If a region is very wide, two panels have merged into one blob
       and it must be split down the middle.

       Criterion: region width as a fraction of total image width.

       Why this criterion? (measured over 444 frames)
           single-pane : width spans ~0.33-0.42 of the image
           dual-pane   : width spans ~0.67-1.00 of the image
           The range between is nearly empty, so 0.55 separates cleanly.

       Width-to-height ratio was also tried and did *not* work:
       dual-pane 1.31-2.36 against single-pane 0.86-2.30 — the
       distributions overlap almost completely.

    WARNING for a different scanner:
       0.55 encodes this scanner's layout (two panels, each ~37% of the
       frame width). On a different layout this value must be retuned.
       For frames known to be single-pane, pass --split-width-frac 2.
    """
    H, W = mask.shape
    min_area = min_area_ratio * H * W
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        comp = (lab == i).astype(np.uint8)

        if (w / W) > split_width_frac:
            cut = find_split_column(gray, comp, (x, y, w, h))
            for xa, xb in ((x, cut), (cut, x + w)):
                if xb - xa < 40:
                    continue
                part = np.zeros_like(comp)
                part[:, xa:xb] = comp[:, xa:xb]
                ys = np.where(part.any(axis=1))[0]
                xs = np.where(part.any(axis=0))[0]
                if ys.size and xs.size:
                    out.append((part, (int(xs.min()), int(ys.min()),
                                       int(xs.max() - xs.min() + 1),
                                       int(ys.max() - ys.min() + 1))))
        else:
            out.append((comp, (x, y, w, h)))

    # Sort left to right so numbering is stable across runs.
    out.sort(key=lambda r: r[1][0])
    return out


def make_crop(img_bgr, region_mask, box, blacken_outside=False, pad=0):
    """
    Cut out the tissue.

    blacken_outside=False : plain rectangular crop around the tissue
                            (matches how the operator cropped by hand)
    blacken_outside=True  : everything outside the tissue shape is zeroed
    """
    H, W = img_bgr.shape[:2]
    x, y, w, h = box
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(W, x + w + pad); y1 = min(H, y + h + pad)

    crop = img_bgr[y0:y1, x0:x1].copy()
    if blacken_outside:
        m = region_mask[y0:y1, x0:x1]
        crop[m == 0] = 0
    return crop, (x0, y0, x1 - x0, y1 - y0)


def make_overlay(img_bgr, mask, regions):
    """Draw the mask in orange over the frame, for visual checking."""
    out = img_bgr.copy()
    col = np.array([0, 160, 255], dtype=np.float32)     # orange in BGR
    out[mask > 0] = (0.55 * out[mask > 0] + 0.45 * col).astype(np.uint8)
    for i, (_, (x, y, w, h)) in enumerate(regions, 1):
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.putText(out, str(i), (x + 8, y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    return out


# ---------------------------------------------------------------------------
# Collecting input files
# ---------------------------------------------------------------------------

def collect_inputs(path):
    """The input may be a single file, a directory, or a glob pattern."""
    if os.path.isdir(path):
        files = []
        for ext in IMG_EXT:
            files += glob.glob(os.path.join(path, '**', '*' + ext),
                               recursive=True)
        return sorted(files)
    if any(ch in path for ch in '*?['):
        return sorted(p for p in glob.glob(path, recursive=True)
                      if p.lower().endswith(IMG_EXT))
    return [path] if os.path.isfile(path) else []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description='Extract tissue regions from ultrasound frames '
                    'using a trained U-Net')
    ap.add_argument('--input', required=True,
                    help='file, directory, or glob such as "data/*.png"')
    ap.add_argument('--out-dir', default='extracted')
    ap.add_argument('--model', default='models/unet_tissue_v1.pt')
    ap.add_argument('--threshold', type=float, default=0.5,
                    help='decision threshold; higher gives a more '
                         'conservative mask')
    ap.add_argument('--min-area', type=float, default=0.01,
                    help='regions smaller than this fraction of the frame '
                         'are discarded')
    ap.add_argument('--pad', type=int, default=0,
                    help='pixels of margin to add around each crop')
    ap.add_argument('--blacken-outside', action='store_true',
                    help='zero everything outside the tissue shape instead '
                         'of taking a rectangular crop')
    ap.add_argument('--split-width-frac', type=float, default=0.55,
                    help='a region wider than this fraction of the image is '
                         'treated as two panels and split down the middle; '
                         'pass 2 for known single-pane frames so they are '
                         'never split')
    ap.add_argument('--save-overlay', action='store_true',
                    help='also write visual check images')
    ap.add_argument('--save-mask', action='store_true', default=True)
    args = ap.parse_args()

    files = collect_inputs(args.input)
    if not files:
        print(f'Error: no images found at: {args.input}')
        return
    if not os.path.exists(args.model):
        print(f'Error: model file not found: {args.model}')
        print('Train one first:  python src/train_unet.py')
        return

    # --- compute device ---
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model, info = load_model(args.model, device)
    dice_txt = (f'{info["val_dice"]:.4f}' if info['val_dice'] else '?')
    print(f'Loaded model: {args.model}')
    print(f'  (epoch {info["epoch"]}, validation Dice: {dice_txt})')
    print(f'Device: {device}')
    print(f'Input images: {len(files)}\n')

    crop_dir = os.path.join(args.out_dir, 'crops')
    mask_dir = os.path.join(args.out_dir, 'masks')
    over_dir = os.path.join(args.out_dir, 'overlays')
    os.makedirs(crop_dir, exist_ok=True)
    if args.save_mask:
        os.makedirs(mask_dir, exist_ok=True)
    if args.save_overlay:
        os.makedirs(over_dir, exist_ok=True)

    manifest = []
    n_crops = 0
    skipped = []

    for idx, path in enumerate(files, 1):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            skipped.append((path, 'could not be read'))
            continue

        mask = predict_mask(model, img, device,
                            info['img_h'], info['img_w'], args.threshold)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        regions = split_regions(mask, gray, args.min_area,
                                args.split_width_frac)

        stem = os.path.splitext(os.path.basename(path))[0]

        if not regions:
            skipped.append((path, 'no tissue found'))
            continue

        if args.save_mask:
            cv2.imwrite(os.path.join(mask_dir, stem + '.png'), mask * 255)
        if args.save_overlay:
            cv2.imwrite(os.path.join(over_dir, stem + '.jpg'),
                        make_overlay(img, mask, regions),
                        [cv2.IMWRITE_JPEG_QUALITY, 88])

        for k, (rmask, box) in enumerate(regions, 1):
            crop, final_box = make_crop(img, rmask, box,
                                        args.blacken_outside, args.pad)
            name = f'{stem}_{k}.png'
            cv2.imwrite(os.path.join(crop_dir, name), crop)
            n_crops += 1
            x, y, w, h = final_box
            manifest.append({
                'crop_file': name,
                'source_image': path,
                'region_index': k,
                'x': x, 'y': y, 'w': w, 'h': h,
                'tissue_fill_percent': round(
                    float(rmask[y:y + h, x:x + w].mean()) * 100, 1),
            })

        if idx % 50 == 0:
            print(f'  {idx}/{len(files)} frames processed')

    # --- manifest ---
    if manifest:
        mp = os.path.join(args.out_dir, 'manifest.csv')
        with open(mp, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
            w.writeheader()
            w.writerows(manifest)

    # --- summary ---
    print('\n' + '=' * 58)
    print(f'Input frames   : {len(files)}')
    print(f'Crops produced : {n_crops}')
    print(f'Crops written  : {crop_dir}')
    if manifest:
        print(f'Manifest       : {os.path.join(args.out_dir, "manifest.csv")}')
        per = {}
        for m in manifest:
            per[m['source_image']] = per.get(m['source_image'], 0) + 1
        from collections import Counter
        dist = Counter(per.values())
        print('\nCrops per frame: '
              + ', '.join(f'{v} frames -> {k} crop(s)'
                          for k, v in sorted(dist.items())))
    if skipped:
        print(f'\nSkipped: {len(skipped)}')
        for p, why in skipped[:10]:
            print(f'  {why}: {os.path.basename(p)}')
        if len(skipped) > 10:
            print(f'  ... and {len(skipped)-10} more')
        print('\n  If tissue was missed, the threshold may be too high.')
        print('  Try:  --threshold 0.3')
    print('=' * 58)


if __name__ == '__main__':
    main()
