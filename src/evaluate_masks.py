#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_masks.py — score the extractor against masks drawn by hand
====================================================================

What it does
    Runs the trained model on every frame that has a hand-drawn mask and reports
    how well they agree (Dice and IoU), with a 95% bootstrap interval and the
    worst frames. It can also score two reference points so the number means
    something:

      constant mask   the best you can do with no image information at all: for each
                      frame, use the majority mask of all the OTHER frames. If this is
                      already high, your scanner has a near-fixed layout and a model
                      earns little over a fixed rectangle. (Leave-one-out, so a frame
                      never helps predict itself.)

      BUSClean        the scan-area crop of the published BUSClean tool, run unmodified,
                      if you point --busclean-repo at a clone of
                      https://github.com/hawaii-ai/bus-cleaning

Usage
    python src/evaluate_masks.py --annotator alice \\
        --model models/unet_tissue_v1.pt

    # two annotators, plus the baselines
    python src/evaluate_masks.py --annotator alice --annotator bob \\
        --model models/unet_tissue_v1.pt --constant-baseline \\
        --busclean-repo /path/to/bus-cleaning

Masks are read from  <work-dir>/masks_<annotator>/  (the folder used by
annotate_masks.py; the first annotator may omit --annotator and uses  masks/ ).

How to read the output
    * Compare the model's Dice with the AGREEMENT between your annotators
      (annotate_masks.py agreement). That agreement is the ceiling: no method can be
      expected to beat how consistent people are with each other.
    * A difference of ~0.01 Dice between two methods on a few dozen frames is
      usually inside the noise. The paired table gives an interval; if it spans 0,
      you have not shown that one method is better.
    * The model is run with the same post-processing as extract_tissue.py (holes are
      filled by default). Use --no-fill-holes to reproduce raw network output.
"""

import argparse
import csv
import glob
import os
import sys
import tempfile

import cv2
import numpy as np
import torch

from extract_tissue import load_model, predict_mask, fill_holes


IMG_EXT = ('.jpg', '.jpeg', '.png')


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def natural_key(name):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', name)]


def dice_iou(a, b):
    inter = np.logical_and(a, b).sum()
    sa, sb = a.sum(), b.sum()
    union = np.logical_or(a, b).sum()
    return (1.0 if sa + sb == 0 else 2.0 * inter / (sa + sb),
            1.0 if union == 0 else inter / union)


def bootstrap_ci(values, n=10000, seed=0):
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    m = [rng.choice(v, len(v)).mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def bounding_rect(mask):
    ys, xs = np.where(mask)
    out = np.zeros_like(mask)
    if ys.size:
        out[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
    return out


def box_mask(shape, box):
    left, top, right, bottom = box
    m = np.zeros(shape, bool)
    m[top:bottom, left:right] = True
    return m


# ---------------------------------------------------------------------------
# BUSClean (optional): the published scan-area crop, run as shipped
# ---------------------------------------------------------------------------

def load_busclean(repo):
    """
    Import BUSClean's cropping module from a local clone.

    Its cropping.py uses `os` and `scipy` without importing them (an omission in the
    published code), so they are supplied here. Nothing else is changed: thresholds and
    logic are exactly as published.
    """
    import scipy
    import scipy.stats
    sys.path.insert(0, repo)
    import modules.cropping as C
    C.os = os
    C.scipy = scipy
    return C


def busclean_box(C, img_bgr):
    """Return BUSClean's crop as (left, top, right, bottom), or None if it fails."""
    with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
        # BUSClean only reads PNG. Re-encoding the decoded pixels is lossless.
        cv2.imwrite(os.path.join(src, 'frame.png'), img_bgr)
        held = {}
        original = C.find_cropping_coordinates

        def spy(image):
            held['box'] = original(image)
            return held['box']

        C.find_cropping_coordinates = spy
        try:
            C.crop_scan('frame.png', src, dst)
        except Exception as e:          # published code can raise on unusual frames
            print(f'    BUSClean failed on a frame: {type(e).__name__}')
            return None
        finally:
            C.find_cropping_coordinates = original
    top, bottom, left, right = held['box']
    return int(left), int(top), int(right), int(bottom)


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def summarise(label, values):
    v = np.asarray(values, float)
    lo, hi = bootstrap_ci(v)
    print(f'  {label:34s} mean {v.mean():.4f}  median {np.median(v):.4f}  '
          f'min {v.min():.4f}   95% CI [{lo:.4f}, {hi:.4f}]')


def paired(label_a, a, label_b, b, tie=0.002):
    d = np.asarray(a, float) - np.asarray(b, float)
    lo, hi = bootstrap_ci(d)
    wins, ties, losses = int((d > tie).sum()), int((np.abs(d) <= tie).sum()), int((d < -tie).sum())
    verdict = ('interval spans 0: no difference shown' if lo <= 0 <= hi
               else ('first is better' if lo > 0 else 'second is better'))
    print(f'  {label_a} minus {label_b}: {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]   '
          f'wins {wins} / ties {ties} / losses {losses}  ({verdict})')


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='Score the extractor against hand-drawn masks')
    ap.add_argument('--work-dir', default='annotation_work')
    ap.add_argument('--annotator', action='append', default=None,
                    help='annotator name (repeatable). Omit for the default folder "masks".')
    ap.add_argument('--model', default='models/unet_tissue_v1.pt')
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--no-fill-holes', action='store_true')
    ap.add_argument('--constant-baseline', action='store_true',
                    help='also score the leave-one-out constant-mask baseline')
    ap.add_argument('--busclean-repo', default=None,
                    help='path to a clone of hawaii-ai/bus-cleaning, to score its scan-area crop')
    ap.add_argument('--out', default=None, help='per-frame CSV (default <work-dir>/evaluation.csv)')
    args = ap.parse_args()

    annotators = args.annotator or [None]
    img_dir = os.path.join(args.work_dir, 'images')
    images = {}
    for p in glob.glob(os.path.join(img_dir, '*')):
        if p.lower().endswith(IMG_EXT):
            images[os.path.splitext(os.path.basename(p))[0]] = p
    if not images:
        print(f'No frames in {img_dir}')
        return 1

    device = get_device()
    model, info = load_model(args.model, device)
    print(f'Model {args.model}   device {device}   fill holes: {not args.no_fill_holes}\n')

    C = load_busclean(args.busclean_repo) if args.busclean_repo else None

    # run the model and BUSClean once per frame (shared by all annotators)
    stems = sorted(images, key=natural_key)
    pred, rect, bc = {}, {}, {}
    for s in stems:
        img = cv2.imread(images[s], cv2.IMREAD_COLOR)
        m = predict_mask(model, img, device, info['img_h'], info['img_w'], args.threshold)
        if not args.no_fill_holes:
            m = fill_holes(m)
        pred[s] = m > 0
        rect[s] = bounding_rect(pred[s])
        if C is not None:
            b = busclean_box(C, img)
            bc[s] = None if b is None else box_mask(pred[s].shape, b)

    rows = []
    for who in annotators:
        mdir = os.path.join(args.work_dir, 'masks' if not who else f'masks_{who}')
        gts = {}
        for p in glob.glob(os.path.join(mdir, '*.png')):
            s = os.path.splitext(os.path.basename(p))[0]
            if s in pred:
                g = cv2.imread(p, cv2.IMREAD_GRAYSCALE) > 127
                if g.shape == pred[s].shape:
                    gts[s] = g
        names = sorted(gts, key=natural_key)
        if not names:
            print(f'No usable masks in {mdir}')
            continue

        # leave-one-out constant mask, only among frames of the same size
        const = {}
        if args.constant_baseline:
            for s in names:
                others = [gts[o] for o in names if o != s and gts[o].shape == gts[s].shape]
                if others:
                    const[s] = np.mean(others, axis=0) > 0.5

        label = who or '(default)'
        print(f'=== annotator {label}: {len(names)} frames ===')
        series = {'model': [], 'model as rectangle': []}
        if const:
            series['constant mask'] = []
        if C is not None:
            series['BUSClean crop'] = []
        used = []
        for s in names:
            d_m, i_m = dice_iou(pred[s], gts[s])
            d_r, _ = dice_iou(rect[s], gts[s])
            row = {'annotator': label, 'file': s, 'dice_model': round(d_m, 4), 'iou_model': round(i_m, 4),
                   'dice_model_rect': round(d_r, 4), 'dice_constant': '', 'dice_busclean': ''}
            if s in const:
                row['dice_constant'] = round(dice_iou(const[s], gts[s])[0], 4)
            if C is not None and bc.get(s) is not None:
                row['dice_busclean'] = round(dice_iou(bc[s], gts[s])[0], 4)
            rows.append(row)
            used.append(row)
        for r in used:
            series['model'].append(r['dice_model'])
            series['model as rectangle'].append(r['dice_model_rect'])
        if const:
            series['constant mask'] = [r['dice_constant'] for r in used if r['dice_constant'] != '']
        if C is not None:
            series['BUSClean crop'] = [r['dice_busclean'] for r in used if r['dice_busclean'] != '']
        for k, v in series.items():
            if v:
                summarise(k, v)

        if C is not None:
            ok = [r for r in used if r['dice_busclean'] != '']
            if ok:
                print('\n  Paired comparison on the same frames:')
                bcv = [r['dice_busclean'] for r in ok]
                paired('model', [r['dice_model'] for r in ok], 'BUSClean', bcv)
                paired('model as rectangle', [r['dice_model_rect'] for r in ok], 'BUSClean', bcv)
        worst = sorted(used, key=lambda r: r['dice_model'])[:5]
        print('\n  Lowest-scoring frames for the model (look at these):')
        for r in worst:
            print(f'    {r["file"]}  Dice {r["dice_model"]:.3f}')
        print()

    out = args.out or os.path.join(args.work_dir, 'evaluation.csv')
    if rows:
        with open(out, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f'Per-frame table: {out}')
    if len(annotators) >= 2:
        print('\nWith two or more annotators, also run:  annotate_masks.py agreement --a <A> --b <B>')
        print('and compare the model score with that agreement: it is the ceiling.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
