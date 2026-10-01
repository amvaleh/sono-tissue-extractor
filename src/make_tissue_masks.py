#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate tissue masks for ultrasound frames (the "teacher" stage)
=================================================================

What this script does
    Takes a raw ultrasound frame and produces a binary mask:
        white (255) = tissue
        black (0)   = not tissue (text, background, vendor marks)

    There is no machine learning here — only measured rules.

Why it exists
    These masks become the ground truth for training the U-Net in the
    next stage. Rather than having someone hand-paint hundreds of
    frames, the rules label the whole set at no annotation cost.

How it works
    Ultrasound tissue has three properties that nothing else in the
    frame shares. These numbers were measured on real data from this
    project:

        region             brightness   speckle (local sd)   colourfulness
        ----------------   ----------   ------------------   -------------
        tissue                     65                   15               4
        black background           17                    4               1
        white text                157                  100               0
        coloured marks            110                   84             127

    So:
      - tissue is neither very dark nor very bright  -> brightness filter
      - tissue is speckled, but not as sharp as text -> speckle filter
      - tissue is grey, not coloured                 -> colour filter

Usage
    # try it on 20 frames first and check the result
    python src/make_tissue_masks.py --limit 20

    # then run over everything
    python src/make_tissue_masks.py

Outputs
    masks/        the masks (named after the source frame)
    overlays/     frame with the mask drawn in green, for visual checks
    mask_qc.csv   a quality score per frame
"""

import argparse
import csv
import os
import glob
import collections

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Settings — tune these to change mask quality
# ---------------------------------------------------------------------------

WIN = 11            # window size for the speckle measurement (must be odd)

SD_MIN, SD_MAX = 7, 45      # speckle range for tissue
                            # below 7  -> flat, i.e. black background
                            # above 45 -> too sharp, i.e. text or a line

MU_MIN, MU_MAX = 28, 190    # brightness range for tissue
                            # below 28  -> background dark
                            # above 190 -> text white

SAT_MAX = 12        # maximum colourfulness. Tissue is grey, so the
                    # scanner's coloured marks (the orange "S", blue
                    # lines) are rejected.

MIN_AREA = 15000    # blobs smaller than this (in pixels) are discarded.
                    # Real tissue is always one large region.

# --- stage 2 settings (locating the tissue depth) ---

DEPTH_MIN_DENSITY = 0.35   # a column counts as tissue down to the point
                           # where at least this fraction of its
                           # neighbourhood is tissue.
                           # Lower it -> deeper masks.

DEPTH_MEDIAN = 81          # median filter window, in columns.
                           # Rejects outlier columns.
                           # Raise it -> fewer green tongues.

DEPTH_CEILING_MARGIN = 60  # no column may run more than this many pixels
                           # deeper than the panel's typical depth.

DEPTH_SMOOTH = 31          # final averaging to soften the boundary,
                           # in columns. Raise it -> smoother but less
                           # precise.


# ---------------------------------------------------------------------------
# Helper: median filter over a 1-D array
# ---------------------------------------------------------------------------

def _median_filter_1d(arr, k):
    """
    Replace each entry with the median of the k entries around it.

    Why median and not mean?
        Suppose the depths are:  [530, 535, 981, 540, 528]
        mean   -> 981 drags its neighbours down with it.
        median -> 981 is discarded.
    """
    if k % 2 == 0:
        k += 1
    pad = k // 2
    padded = np.pad(arr, (pad, pad), mode='edge')
    # sliding windows, without a Python loop
    windows = np.lib.stride_tricks.sliding_window_view(padded, k)
    return np.median(windows, axis=1)


# ---------------------------------------------------------------------------
# Main routine: raw frame -> tissue mask
# ---------------------------------------------------------------------------

def tissue_mask(img_bgr):
    """
    Input  : a raw colour frame (BGR, as cv2.imread returns)
    Output : a 0/1 mask the same size as the frame

    ------------------------------------------------------------------
    Why two stages?
    ------------------------------------------------------------------
    The first version of this code checked each pixel independently.
    The result: when one of the two panels was darker than the other
    (we measured brightness 48 against 64), the darker panel's interior
    came out riddled with holes.

    The cause was fixed thresholds. The darker panel simply fell below
    them and was deleted.

    The fix:
      stage 1 - use the fixed thresholds only to locate *where the
                panels are*. (This part worked well; it correctly
                excluded margins and text.)
      stage 2 - inside each panel, derive the thresholds from *that
                panel itself* rather than from constants. Then, for
                each column, find how deep the tissue runs and fill
                down to there.

    ------------------------------------------------------------------
    Important: why dark regions are kept
    ------------------------------------------------------------------
    Lymph nodes and vessels are *dark* on ultrasound. Removing dark
    areas would throw away exactly what we are trying to detect.
    So the mask must cover the whole tissue region, not only the
    bright parts of it.
    """

    # ---- setup: three base measurements ------------------------------
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = gray.shape

    # "Colourfulness": a grey pixel has R=G=B, so the spread is zero.
    # The scanner's coloured marks have a large spread.
    saturation = (np.max(img_bgr, axis=2).astype(np.int16)
                  - np.min(img_bgr, axis=2).astype(np.int16))

    # Brightness and speckle in a WIN x WIN window around each pixel.
    # Fast standard deviation: sqrt( mean of squares - square of mean )
    mean = cv2.blur(gray, (WIN, WIN))
    std = np.sqrt(np.maximum(cv2.blur(gray * gray, (WIN, WIN))
                             - mean * mean, 0))

    is_gray = cv2.blur(saturation.astype(np.float32), (WIN, WIN)) < SAT_MAX

    # ================================================================
    # Stage 1: where are the panels?
    # ================================================================
    seed = ((std > SD_MIN) & (std < SD_MAX)
            & (mean > MU_MIN) & (mean < MU_MAX)
            & is_gray).astype(np.uint8)

    # OPEN  = removes isolated specks
    # CLOSE = closes small gaps
    seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(seed, 8)
    # Large blobs are real panels; small ones are text or noise.
    panels = [i for i in range(1, n_labels)
              if stats[i, cv2.CC_STAT_AREA] > MIN_AREA]

    if not panels:
        return np.zeros((H, W), np.uint8)

    # ================================================================
    # Stage 2: find the true tissue boundary inside each panel
    # ================================================================
    out = np.zeros((H, W), np.uint8)

    for i in panels:
        x0 = stats[i, cv2.CC_STAT_LEFT]
        x1 = x0 + stats[i, cv2.CC_STAT_WIDTH]
        comp = (labels == i)

        region = is_gray[:, x0:x1]
        if region.sum() < 1000:
            continue

        # --- adaptive thresholds, derived from this panel ---
        # A dark panel gets correspondingly lower thresholds. This is
        # what fixes the hole problem.
        sd_thr = max(3.5, np.percentile(std[:, x0:x1][region], 55) * 0.50)
        mu_thr = max(10.0, np.percentile(mean[:, x0:x1][region], 60) * 0.30)
        tissue = ((std[:, x0:x1] > sd_thr)
                  & (mean[:, x0:x1] > mu_thr)
                  & region)

        # --- how deep does the tissue run in each column? ---
        # Spread the tissue density over a 41x41 window so the estimate
        # is not fragmented, then take the lowest row still showing
        # tissue.
        density = cv2.blur(tissue.astype(np.float32), (41, 41))
        depth = np.zeros(x1 - x0, np.float32)
        for c in range(x1 - x0):
            rows = np.where(density[:, c] > DEPTH_MIN_DENSITY)[0]
            depth[c] = rows.max() if rows.size else 0

        # --- smooth the depth profile ---
        # Three steps, and the order matters:
        #
        # 1) Median filter. A few stray columns can land far deeper than
        #    the rest (we saw a median of 533 against one column at 981).
        #    Averaging would drag the neighbours down with the outlier
        #    and produce a long tongue in the mask. The median discards
        #    it outright.
        depth = _median_filter_1d(depth, DEPTH_MEDIAN)

        # 2) A safe ceiling: no column may run far deeper than this
        #    panel's typical depth.
        ceiling = np.percentile(depth, 90) + DEPTH_CEILING_MARGIN
        depth = np.minimum(depth, ceiling)

        # 3) Light averaging, purely to soften the final boundary.
        k = DEPTH_SMOOTH
        depth = np.convolve(np.pad(depth, (k // 2, k // 2), mode='edge'),
                            np.ones(k) / k, mode='valid')

        # --- fill from the skin line down to that depth ---
        # Everything in between counts as tissue, dark regions included.
        top = int(np.where(comp.any(axis=1))[0].min())
        for c in np.where(comp.any(axis=0))[0]:
            d = int(depth[c - x0])
            if d > top + 40:
                out[top:d, c] = 1

    # ---- final cleanup: discard small blobs ---------------------------
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(out, 8)
    clean = np.zeros_like(out)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] > MIN_AREA:
            clean[labels == i] = 1

    return clean


# ---------------------------------------------------------------------------
# Visual check image: mask drawn in green over the frame
# ---------------------------------------------------------------------------

def make_overlay(img_bgr, mask, boxes=()):
    """
    green = tissue found by this script
    red   = the operator's manual crop, where one is available

    If green and red line up, the rules worked.
    """
    out = img_bgr.copy()
    green = np.array([0, 230, 0], dtype=np.float32)
    out[mask > 0] = (0.45 * out[mask > 0] + 0.55 * green).astype(np.uint8)

    for (x, y, w, h) in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 4)
    return out


# ---------------------------------------------------------------------------
# Reading the manual crops, used only for scoring
# ---------------------------------------------------------------------------

def load_reference_boxes(csv_path):
    """
    Read roi_boxes.csv.
    Returns {raw image path: [(x, y, w, h), ...]}

    These rectangles were recovered from the operator's manual crops by
    template matching. They are used only to *score* the masks — never
    to build them.
    """
    boxes = collections.defaultdict(list)
    if not os.path.exists(csv_path):
        return boxes
    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            boxes[row['raw_path']].append((int(row['x']), int(row['y']),
                                           int(row['w']), int(row['h'])))
    return boxes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description='Generate tissue masks from raw ultrasound frames')
    ap.add_argument('--raw-dir', required=True,
                    help='directory of raw frames')
    ap.add_argument('--out-dir', default='work',
                    help='output directory')
    ap.add_argument('--boxes-csv', default='',
                    help='optional roi_boxes.csv, used for scoring only')
    ap.add_argument('--limit', type=int, default=0,
                    help='process only this many frames (0 = all)')
    ap.add_argument('--no-overlays', action='store_true',
                    help='skip the visual check images (faster)')
    args = ap.parse_args()

    mask_dir = os.path.join(args.out_dir, 'masks')
    over_dir = os.path.join(args.out_dir, 'overlays')
    os.makedirs(mask_dir, exist_ok=True)
    if not args.no_overlays:
        os.makedirs(over_dir, exist_ok=True)

    ref_boxes = load_reference_boxes(args.boxes_csv) if args.boxes_csv else {}

    files = sorted(glob.glob(os.path.join(args.raw_dir, '*', '**', '*.png'),
                             recursive=True))
    if not files:
        files = sorted(glob.glob(os.path.join(args.raw_dir, '**', '*.png'),
                                 recursive=True))
    if args.limit:
        files = files[:args.limit]

    print(f'Frames to process: {len(files)}')
    print(f'Masks will be written to: {mask_dir}\n')

    report = []
    for i, path in enumerate(files, 1):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f'  [skipped] could not read: {path}')
            continue

        mask = tissue_mask(img)

        # Output name: patient id + original name, to avoid collisions
        parts = path.split(os.sep)
        base = args.raw_dir.rstrip(os.sep).split(os.sep)
        patient = parts[len(base)] if len(parts) > len(base) else 'x'
        stem = f'{patient}__{os.path.splitext(os.path.basename(path))[0]}'

        cv2.imwrite(os.path.join(mask_dir, stem + '.png'), mask * 255)

        boxes = ref_boxes.get(path, [])
        if not args.no_overlays:
            cv2.imwrite(os.path.join(over_dir, stem + '.jpg'),
                        make_overlay(img, mask, boxes),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])

        # --- scoring ---
        # coverage = what fraction of the operator's crop the mask covers.
        # Closer to 100% is better.
        coverage = ''
        if boxes:
            vals = [mask[y:y + h, x:x + w].mean() for (x, y, w, h) in boxes]
            coverage = round(float(np.mean(vals)) * 100, 1)

        report.append({
            'raw_path': path,
            'mask_file': stem + '.png',
            'tissue_percent': round(float(mask.mean()) * 100, 1),
            'coverage_of_manual_crop': coverage,
            'n_manual_boxes': len(boxes),
        })

        if i % 50 == 0:
            print(f'  {i}/{len(files)} done')

    # --- write the report ---
    qc_path = os.path.join(args.out_dir, 'mask_qc.csv')
    with open(qc_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        w.writeheader()
        w.writerows(report)

    # --- summary ---
    scored = [r['coverage_of_manual_crop'] for r in report
              if r['coverage_of_manual_crop'] != '']
    print('\n' + '=' * 55)
    print(f'Masks written : {len(report)}')
    print(f'QC report     : {qc_path}')
    if scored:
        arr = np.array(scored, dtype=float)
        print('\nScore (coverage of the manual crop):')
        print(f'  mean          : {arr.mean():.1f}%')
        print(f'  median        : {np.median(arr):.1f}%')
        print(f'  worst         : {arr.min():.1f}%')
        print(f'  above 90%     : {(arr > 90).sum()} of {len(arr)}')
        print('\n  A mean above 90% means the masks are ready for training.')
        weak = [r for r in report
                if r['coverage_of_manual_crop'] != ''
                and float(r['coverage_of_manual_crop']) < 80]
        if weak:
            print(f'\n  {len(weak)} frames scored below 80%. Look at these')
            print('  first, in the overlays directory.')
    print('=' * 55)


if __name__ == '__main__':
    main()
