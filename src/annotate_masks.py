#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
annotate_masks.py — draw reference masks by hand, and measure how well people agree
====================================================================================

Why this exists
    To say how accurate the extractor is on a new scanner or institution you need
    masks that a person drew. Without them, any number reported for that site only
    shows that the output is well-formed, not that it is correct.

    One annotator gives a score with no ceiling: you cannot tell whether the
    remaining error is the model's or simply two reasonable people disagreeing
    about an edge. Two annotators on the same frames give that ceiling. This tool is
    built so several people can draw the same frames independently and then be
    compared.

Four subcommands
    prepare    pick frames at random (one per patient) and copy them into a work
               folder. Safe to run again: it only ever ADDS frames and never renumbers
               or overwrites existing ones, so masks you already drew stay attached to
               the right image.

    draw       open each frame and draw the mask with the mouse. Every annotator gets
               their own mask folder. Time per frame is logged.

    check      sanity-check drawn masks (wrong size, empty, covers the header, ...).

    agreement  compare two annotators frame by frame (Dice / IoU).

Typical session
    python src/annotate_masks.py prepare --raw-dir /path/to/frames --n 100
    python src/annotate_masks.py draw    --annotator alice
    python src/annotate_masks.py check   --annotator alice --header-rows 100
    python src/annotate_masks.py agreement --a alice --b bob

WHAT TO DRAW is defined in docs/ANNOTATION_GUIDE.md (protocol v1). Read it first: all
annotators must follow the same rule, otherwise their disagreement measures the rule,
not the model.

Protocol v1 in one line: one polygon around the whole image field, dark areas
included; header, settings column, depth ruler and text outside the field excluded.

Keys while drawing
    left click   add a point
    right click  (or Z) undo the last point
    SPACE        close this polygon and start another one
    ENTER        save and go to the next frame
    T            toggle contrast enhancement (makes dark edges easier to see)
    H            show / hide the protocol reminder
    C            clear everything on this frame
    S            skip this frame (nothing saved)
    B            go back to the previous frame
    Q            quit (progress is saved; the next run continues where you stopped)

Dependencies: numpy and opencv only (no torch needed for this file).
"""

import argparse
import csv
import datetime
import glob
import os
import random
import re
import shutil
import sys
import time

import cv2
import numpy as np


PROTOCOL_VERSION = 'v1'

# Shown on screen when you press H. Keep it short; the full text is in the guide.
PROTOCOL_LINES = [
    'PROTOCOL v1  -  whole image field',
    '1. One polygon around the area where the ultrasound picture is shown:',
    '   from the skin line (top) to the bottom edge, left edge to right edge.',
    '2. INCLUDE dark parts inside that area: lesions, vessels, shadow, black',
    '   strips where the probe lost contact. Calipers/text lying on it stay in.',
    '3. EXCLUDE everything outside it: header, settings column, depth ruler,',
    '   text beside the picture, black margins.',
    '4. Two panels side by side: one polygon per panel.',
    'Corners need not be pixel-perfect; 6-15 points is enough.',
]

DEFAULT_WORK = 'annotation_work'
IMG_EXT = ('.jpg', '.jpeg', '.png')


# ===========================================================================
# helpers
# ===========================================================================

def natural_key(name):
    """m2 < m10 (plain alphabetical order would put m10 before m2)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', name)]


def mask_dir_for(work_dir, annotator):
    """First annotator may omit --annotator and uses 'masks'; others use masks_<name>."""
    return os.path.join(work_dir, 'masks' if not annotator else f'masks_{annotator}')


def dice_iou(a, b):
    """Dice and IoU of two boolean masks. Two empty masks count as perfect."""
    inter = np.logical_and(a, b).sum()
    sa, sb = a.sum(), b.sum()
    union = np.logical_or(a, b).sum()
    dice = 1.0 if sa + sb == 0 else 2.0 * inter / (sa + sb)
    iou = 1.0 if union == 0 else inter / union
    return float(dice), float(iou)


def bootstrap_ci(values, n=10000, seed=0):
    """95% bootstrap interval of the mean."""
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = [rng.choice(v, len(v)).mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ===========================================================================
# prepare
# ===========================================================================

def read_selection(path):
    if not os.path.exists(path):
        return []
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def patient_of(path, raw_dir):
    """Patient id = first folder below raw_dir (frames are stored as raw_dir/<patient>/...)."""
    rel = os.path.relpath(path, raw_dir)
    parts = rel.split(os.sep)
    return parts[0] if len(parts) > 1 else '(root)'


def cmd_prepare(args):
    files = []
    for ext in IMG_EXT:
        files += glob.glob(os.path.join(args.raw_dir, '**', '*' + ext), recursive=True)
    files = sorted(set(files))
    if not files:
        print(f'Error: no images found under {args.raw_dir}')
        return 1

    img_dir = os.path.join(args.work_dir, 'images')
    sel_path = os.path.join(args.work_dir, 'selection.csv')
    os.makedirs(img_dir, exist_ok=True)

    existing = read_selection(sel_path)
    used_patients = {patient_of(r['source'], args.raw_dir) for r in existing}
    used_sources = {os.path.abspath(r['source']) for r in existing}
    next_index = 1 + max([int(re.sub(r'\D', '', r['file'])) for r in existing] or [0])

    # Frames the caller wants kept out of the sample, e.g. ones that were chosen by hand
    # earlier. Selecting from hand-picked frames would flatter the result.
    excluded = set()
    if args.exclude_dir and os.path.isdir(args.exclude_dir):
        for f in os.listdir(args.exclude_dir):
            if f.lower().endswith(IMG_EXT):
                stem = os.path.splitext(f)[0]
                if stem.lower().endswith(('.jpg', '.png')):     # e.g. name.jpg.png
                    stem = stem[:-4]
                excluded.add(stem)

    ignore = set(args.ignore_folder or [])
    by_patient = {}
    for p in files:
        pid = patient_of(p, args.raw_dir)
        if pid in ignore or pid in used_patients:
            continue
        if os.path.abspath(p) in used_sources:
            continue
        if os.path.splitext(os.path.basename(p))[0] in excluded:
            continue
        by_patient.setdefault(pid, []).append(p)

    have = len(existing)
    want = args.n - have if args.n else args.add
    if want <= 0:
        print(f'Already have {have} frames (target {args.n}). Nothing to add.')
        return 0

    patients = sorted(by_patient)
    if len(patients) < want:
        print(f'Note: only {len(patients)} unused patients left; adding that many.')
        want = len(patients)
    rng = random.Random(args.seed + have)          # deterministic for a given state
    rng.shuffle(patients)

    new_rows = []
    for pid in patients[:want]:
        src = rng.choice(sorted(by_patient[pid]))
        # A neutral name, so the patient id is not in front of the annotator while drawing.
        name = f'm{next_index:02d}{os.path.splitext(src)[1].lower()}'
        dst = os.path.join(img_dir, name)
        if os.path.exists(dst):                    # never overwrite
            raise SystemExit(f'Refusing to overwrite {dst}. selection.csv is out of sync.')
        shutil.copy2(src, dst)
        new_rows.append({'file': name, 'source': src})
        next_index += 1

    with open(sel_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['file', 'source'])
        w.writeheader()
        w.writerows(existing + new_rows)

    print(f'Frames before : {have}')
    print(f'Added         : {len(new_rows)}   (one per patient, none from a patient already used)')
    print(f'Frames now    : {have + len(new_rows)}')
    print(f'Images        : {img_dir}')
    print(f'Selection log : {sel_path}   (keep it: it maps m01.. back to the source frames)')
    print('\nNext:  python src/annotate_masks.py draw --annotator <your name>')
    print('\nWARNING: frame headers usually carry patient identifiers. Keep these images')
    print('local; do not upload or publish them.')
    return 0


# ===========================================================================
# drawing logic (kept separate from the window so it can be tested without a screen)
# ===========================================================================

class Annotation:
    """
    A list of polygons. The last one is the polygon currently being drawn.
    Coordinates are always in ORIGINAL image pixels, never display pixels.
    """

    def __init__(self):
        self.done = []
        self.current = []

    def add(self, x, y):
        self.current.append((int(round(x)), int(round(y))))

    def undo(self):
        if self.current:
            self.current.pop()
        elif self.done:                       # step back into the previous polygon
            self.current = self.done.pop()
            if self.current:
                self.current.pop()

    def close(self):
        if len(self.current) >= 3:
            self.done.append(self.current)
            self.current = []
            return True
        return False                          # fewer than 3 points is not a shape

    def clear(self):
        self.done, self.current = [], []

    def finish(self):
        """Also closes a half-drawn polygon (if it has at least 3 points)."""
        self.close()
        return len(self.done) > 0

    def to_mask(self, h, w):
        m = np.zeros((h, w), np.uint8)
        for poly in self.done:
            cv2.fillPoly(m, [np.array(poly, np.int32)], 255)
        return m


def enhance(img):
    """Local contrast boost (CLAHE on luminance) so dark edges are easier to see."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


TOP_BAR = 34
HELP_KEYS = ['LMB point', 'RMB/Z undo', 'SPACE close', 'ENTER save', 'T contrast',
             'H protocol', 'C clear', 'S skip', 'B back', 'Q quit']


def render(img, ann, scale, name, idx, total, show_help):
    h, w = img.shape[:2]
    disp = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    if ann.done:                              # closed polygons: translucent fill
        over = disp.copy()
        for poly in ann.done:
            p = (np.array(poly, np.float32) * scale).astype(np.int32)
            cv2.fillPoly(over, [p], (0, 200, 0))
        disp = cv2.addWeighted(over, 0.28, disp, 0.72, 0)
        for poly in ann.done:
            p = (np.array(poly, np.float32) * scale).astype(np.int32)
            cv2.polylines(disp, [p], True, (0, 255, 0), 2)

    if ann.current:
        p = (np.array(ann.current, np.float32) * scale).astype(np.int32)
        cv2.polylines(disp, [p], False, (0, 255, 255), 2)
        for q in p:
            cv2.circle(disp, tuple(int(v) for v in q), 4, (0, 0, 255), -1)

    if show_help:
        panel_h = 18 * len(PROTOCOL_LINES) + 12
        sub = disp[:panel_h, :min(disp.shape[1], 640)].copy()
        disp[:panel_h, :sub.shape[1]] = (sub * 0.25).astype(np.uint8)
        for k, line in enumerate(PROTOCOL_LINES):
            cv2.putText(disp, line, (8, 18 + 18 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)

    bar = np.zeros((TOP_BAR, disp.shape[1], 3), np.uint8)
    cv2.putText(bar, f'{name}  [{idx}/{total}]   polygons: {len(ann.done)}', (8, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    foot = np.zeros((26, disp.shape[1], 3), np.uint8)
    cv2.putText(foot, '  |  '.join(HELP_KEYS), (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (200, 200, 200), 1)
    return np.vstack([bar, disp, foot])


def append_log(path, row):
    new = not os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['timestamp', 'annotator', 'protocol', 'file',
                                          'action', 'seconds', 'polygons', 'area_percent'])
        if new:
            w.writeheader()
        w.writerow(row)


def cmd_draw(args):
    img_dir = os.path.join(args.work_dir, 'images')
    mask_dir = mask_dir_for(args.work_dir, args.annotator)
    files = sorted(glob.glob(os.path.join(img_dir, '*')), key=lambda p: natural_key(os.path.basename(p)))
    files = [f for f in files if f.lower().endswith(IMG_EXT)]
    if not files:
        print('No frames found. Run first:  python src/annotate_masks.py prepare --raw-dir <folder>')
        return 1
    os.makedirs(mask_dir, exist_ok=True)

    def mpath(f):
        return os.path.join(mask_dir, os.path.splitext(os.path.basename(f))[0] + '.png')

    redo = set()
    if args.redo:
        redo = {os.path.splitext(os.path.basename(f))[0] for f in files} if args.redo == 'all' \
            else {s.strip() for s in args.redo.split(',') if s.strip()}
    todo = [f for f in files
            if (not os.path.exists(mpath(f))) or os.path.splitext(os.path.basename(f))[0] in redo]
    print(f'Frames: {len(files)}   already drawn: {len(files) - len(todo)}   to do: {len(todo)}')
    print(f'Masks go to: {mask_dir}')
    if not todo:
        print('Nothing left to draw.')
        return 0

    win = 'draw mask  (H = protocol reminder)'
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    state = {'ann': Annotation(), 'scale': 1.0}

    def on_mouse(event, x, y, flags, _):
        s = state['scale']
        if event == cv2.EVENT_LBUTTONDOWN and y >= TOP_BAR:
            state['ann'].add(x / s, (y - TOP_BAR) / s)
        elif event == cv2.EVENT_RBUTTONDOWN:
            state['ann'].undo()

    cv2.setMouseCallback(win, on_mouse)
    log_path = os.path.join(args.work_dir, 'annotations_log.csv')
    who = args.annotator or '(default)'

    def log(f, action, secs, ann, h, w):
        area = ''
        polys = 0
        if ann is not None:
            polys = len(ann.done)
            if polys:
                area = round(float((ann.to_mask(h, w) > 0).mean()) * 100, 1)
        append_log(log_path, {
            'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
            'annotator': who, 'protocol': PROTOCOL_VERSION,
            'file': os.path.basename(f), 'action': action,
            'seconds': round(secs, 1), 'polygons': polys, 'area_percent': area})

    i = 0
    while 0 <= i < len(todo):
        f = todo[i]
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        state['scale'] = min(1.0, args.max_height / h)
        state['ann'] = Annotation()
        contrast, show_help = False, (i == 0)    # protocol shown automatically on the first frame
        name = os.path.basename(f)
        t0 = time.time()

        while True:
            shown = enhance(img) if contrast else img
            cv2.imshow(win, render(shown, state['ann'], state['scale'], name, i + 1, len(todo), show_help))
            k = cv2.waitKey(30) & 0xFF
            ann = state['ann']
            if k == 255:
                continue
            if k in (ord('q'), ord('Q'), 27):
                cv2.destroyAllWindows()
                print('Quit. Progress is saved.')
                return 0
            elif k in (ord('z'), ord('Z')):
                ann.undo()
            elif k == ord(' '):
                if not ann.close():
                    print('  need at least 3 points')
            elif k in (ord('c'), ord('C')):
                ann.clear()
            elif k in (ord('t'), ord('T')):
                contrast = not contrast
            elif k in (ord('h'), ord('H')):
                show_help = not show_help
            elif k in (ord('s'), ord('S')):
                log(f, 'skipped', time.time() - t0, None, h, w)
                print(f'  skipped: {name}')
                i += 1
                break
            elif k in (ord('b'), ord('B')):
                i = max(0, i - 1)
                break
            elif k in (13, 10):
                if ann.finish():
                    m = ann.to_mask(h, w)
                    cv2.imwrite(mpath(f), m)
                    secs = time.time() - t0
                    log(f, 'saved', secs, ann, h, w)
                    print(f'  saved: {name}  ({len(ann.done)} polygon(s), '
                          f'{m.mean() / 255 * 100:.0f}% of frame, {secs:.0f}s)')
                    i += 1
                    break
                print('  nothing drawn yet (need at least 3 points)')

    cv2.destroyAllWindows()
    left = [f for f in files if not os.path.exists(mpath(f))]
    print(f'\nDone. Drawn: {len(files) - len(left)} of {len(files)}')
    return 0


# ===========================================================================
# check
# ===========================================================================

def cmd_check(args):
    img_dir = os.path.join(args.work_dir, 'images')
    mask_dir = mask_dir_for(args.work_dir, args.annotator)
    masks = sorted(glob.glob(os.path.join(mask_dir, '*.png')), key=lambda p: natural_key(os.path.basename(p)))
    if not masks:
        print(f'No masks in {mask_dir}')
        return 1
    flagged = []
    for mp in masks:
        stem = os.path.splitext(os.path.basename(mp))[0]
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        problems = []
        imgp = [p for p in glob.glob(os.path.join(img_dir, stem + '.*')) if p.lower().endswith(IMG_EXT)]
        if not imgp:
            problems.append('no matching image')
        else:
            im = cv2.imread(imgp[0], cv2.IMREAD_COLOR)
            if im.shape[:2] != m.shape:
                problems.append(f'size {m.shape[::-1]} differs from image {im.shape[1::-1]}')
        b = m > 127
        area = b.mean() * 100
        if area < args.min_area:
            problems.append(f'very small ({area:.1f}% of frame)')
        if area > args.max_area:
            problems.append(f'covers almost the whole frame ({area:.1f}%)')
        ncomp = cv2.connectedComponents(b.astype(np.uint8), connectivity=8)[0] - 1
        if ncomp > args.max_regions:
            problems.append(f'{ncomp} separate regions')
        if args.header_rows and b[:args.header_rows].any():
            frac = b[:args.header_rows].mean() * 100
            problems.append(f'reaches into the top {args.header_rows} rows, where the header is ({frac:.1f}% of that strip)')
        if problems:
            flagged.append((stem, problems))
    print(f'Masks checked: {len(masks)}   flagged: {len(flagged)}')
    for stem, probs in flagged:
        print(f'  {stem}: ' + '; '.join(probs))
    if not flagged:
        print('  nothing suspicious.')
    else:
        print('\nLook at each flagged frame. To redraw:  draw --redo ' + ','.join(s for s, _ in flagged[:3]) + ' ...')
    return 0


# ===========================================================================
# agreement
# ===========================================================================

def cmd_agreement(args):
    da, db = mask_dir_for(args.work_dir, args.a), mask_dir_for(args.work_dir, args.b)
    names = sorted({os.path.basename(p) for p in glob.glob(os.path.join(da, '*.png'))} &
                   {os.path.basename(p) for p in glob.glob(os.path.join(db, '*.png'))},
                   key=natural_key)
    if not names:
        print(f'No frames drawn by both annotators.\n  {da}\n  {db}')
        return 1
    rows = []
    for n in names:
        a = cv2.imread(os.path.join(da, n), cv2.IMREAD_GRAYSCALE) > 127
        b = cv2.imread(os.path.join(db, n), cv2.IMREAD_GRAYSCALE) > 127
        if a.shape != b.shape:
            print(f'  skipped {n}: different sizes')
            continue
        d, i = dice_iou(a, b)
        rows.append({'file': os.path.splitext(n)[0], 'dice': round(d, 4), 'iou': round(i, 4)})
    dice = np.array([r['dice'] for r in rows])
    lo, hi = bootstrap_ci(dice)
    print(f'Frames drawn by both : {len(rows)}')
    print(f'Dice  mean {dice.mean():.4f}   median {np.median(dice):.4f}   min {dice.min():.4f}   '
          f'95% CI of mean [{lo:.4f}, {hi:.4f}]')
    print(f'IoU   mean {np.mean([r["iou"] for r in rows]):.4f}')
    low = [r for r in rows if r['dice'] < args.flag_below]
    if low:
        print(f'\nFrames below Dice {args.flag_below} (look at these first):')
        for r in sorted(low, key=lambda r: r['dice']):
            print(f'  {r["file"]}  {r["dice"]:.3f}')
    out = args.out or os.path.join(args.work_dir, f'agreement_{args.a}_vs_{args.b}.csv')
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['file', 'dice', 'iou'])
        w.writeheader()
        w.writerows(rows)
    print(f'\nPer-frame table: {out}')
    print('\nHow to read it: this Dice is the CEILING for any automatic method on these frames.')
    print('A model that scores close to it is as consistent with a person as people are with each other.')
    return 0


# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description='Draw reference tissue masks and measure agreement')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('prepare', help='pick frames and copy them to the work folder (adds only)')
    p.add_argument('--raw-dir', required=True, help='folder of frames; one sub-folder per patient')
    p.add_argument('--work-dir', default=DEFAULT_WORK)
    g = p.add_mutually_exclusive_group()
    g.add_argument('--n', type=int, default=None, help='target TOTAL number of frames')
    g.add_argument('--add', type=int, default=30, help='add this many more frames (default 30)')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--ignore-folder', action='append', default=['PIDunknown'],
                   help='patient folder to skip (repeatable). Default skips PIDunknown, which '
                        'holds frames of unknown patients and is not one patient.')
    p.add_argument('--exclude-dir', default=None,
                   help='folder of earlier hand-picked crops whose source frames must not be sampled')
    p.set_defaults(fn=cmd_prepare)

    d = sub.add_parser('draw', help='draw masks')
    d.add_argument('--work-dir', default=DEFAULT_WORK)
    d.add_argument('--annotator', default=None, help='your name; each person gets their own folder')
    d.add_argument('--redo', default=None, help="redraw: 'all' or a comma list like m05,m09")
    d.add_argument('--max-height', type=int, default=700, help='display height in pixels')
    d.set_defaults(fn=cmd_draw)

    c = sub.add_parser('check', help='sanity-check drawn masks')
    c.add_argument('--work-dir', default=DEFAULT_WORK)
    c.add_argument('--annotator', default=None)
    c.add_argument('--header-rows', type=int, default=0,
                   help='flag masks reaching into the top N rows (the header strip)')
    c.add_argument('--min-area', type=float, default=10.0, help='flag masks below this %% of the frame')
    c.add_argument('--max-area', type=float, default=97.0, help='flag masks above this %% of the frame')
    c.add_argument('--max-regions', type=int, default=2)
    c.set_defaults(fn=cmd_check)

    a = sub.add_parser('agreement', help='Dice/IoU between two annotators')
    a.add_argument('--work-dir', default=DEFAULT_WORK)
    a.add_argument('--a', default=None, help='annotator A (omit for the default folder "masks")')
    a.add_argument('--b', required=True, help='annotator B')
    a.add_argument('--flag-below', type=float, default=0.90)
    a.add_argument('--out', default=None)
    a.set_defaults(fn=cmd_agreement)

    args = ap.parse_args()
    sys.exit(args.fn(args) or 0)


if __name__ == '__main__':
    main()
