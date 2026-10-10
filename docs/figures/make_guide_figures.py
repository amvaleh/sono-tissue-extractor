#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Builds the figures used in docs/ANNOTATION_GUIDE.md.

Every frame here is SYNTHETIC: speckle texture generated from random noise, with
invented header text. No patient data is read or used, which is why these images are
allowed in a public repository while real frames are not.

Requires matplotlib (a development convenience; the tools themselves do not need it).

    python docs/figures/make_guide_figures.py
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

OUT = os.path.dirname(os.path.abspath(__file__))
W, H = 1080, 810
FX0, FX1, FY0, FY1 = 190, 870, 125, 690          # the "image field" of the synthetic scanner
GREEN, RED, BLUE, ORANGE = '#1fbf3a', '#e0312a', '#2b6cb0', '#ff8c00'


# ---------------------------------------------------------------- synthetic frame
def speckle(h, w, rng):
    n = rng.rayleigh(1.0, (h, w)).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), 1.3)
    return (n - n.mean()) / n.std()


def field_pixels(w, h, rng, lesion=None, dark_left=0, shadow=0):
    """Grey-scale picture that fills the image field."""
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    img = 88 * (1 - 0.5 * y) + 15 * speckle(h, w, rng)
    for pos, amp in [(0.10, 40), (0.20, 28), (0.34, 36), (0.52, 24), (0.66, 18)]:     # fascia
        yy = (pos * h + 6 * np.sin(np.linspace(0, 5, w) + pos * 20)).astype(int)
        for x in range(w):
            img[max(0, yy[x] - 1):yy[x] + 2, x] += amp
    img[:5] = 190                                                                       # skin line
    if lesion:
        cx, cy, a, b = lesion
        yy, xx = np.mgrid[0:h, 0:w]
        d = ((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2
        img[d < 1] = 18 + 8 * speckle(h, w, rng)[d < 1]
        post = (np.abs(xx - cx) < a * 0.9) & (yy > cy + b) & (yy < cy + b + 140)       # posterior enhancement
        img[post] += 22
    if dark_left:
        img[5:, :dark_left] = 6 + 3 * np.abs(rng.normal(0, 1, (h - 5, dark_left)))
    if shadow:
        img[int(h * (1 - shadow)):, :] = 6 + 3 * np.abs(rng.normal(0, 1, (h - int(h * (1 - shadow)), w)))
    return np.clip(img, 0, 255).astype(np.uint8)


def put(img, text, xy, scale=0.55, color=(220, 220, 220), thick=1):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def frame(seed=1, lesion=True, dark_left=0, shadow=0, dual=False):
    """Returns an RGB frame and a dict describing where everything is."""
    rng = np.random.default_rng(seed)
    canvas = np.zeros((H, W), np.uint8)
    info = {'field': (FX0, FY0, FX1, FY1), 'lesion': None, 'panels': []}
    if dual:
        mid = (FX0 + FX1) // 2
        for (x0, x1, s) in ((FX0, mid - 1, seed), (mid + 1, FX1, seed + 5)):
            r = np.random.default_rng(s)
            les = (int((x1 - x0) * 0.5), int((FY1 - FY0) * 0.38), 42, 26) if lesion else None
            canvas[FY0:FY1, x0:x1] = field_pixels(x1 - x0, FY1 - FY0, r, les)
            info['panels'].append((x0, FY0, x1, FY1))
            if lesion and s == seed:
                info['lesion'] = (x0 + les[0], FY0 + les[1], les[2], les[3])
    else:
        les = (430, 190, 62, 34) if lesion else None
        canvas[FY0:FY1, FX0:FX1] = field_pixels(FX1 - FX0, FY1 - FY0, rng, les, dark_left, shadow)
        info['panels'].append((FX0, FY0, FX1, FY1))
        if lesion:
            info['lesion'] = (FX0 + les[0], FY0 + les[1], les[2], les[3])
    img = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    # --- scanner furniture (all invented text)
    put(img, 'S', (24, 48), 1.3, (240, 240, 240), 2)
    put(img, 'EXAMPLE CLINIC', (110, 28), 0.6)
    put(img, '01/01/2000 12:00:00', (790, 28), 0.6)
    put(img, 'ID: SAMPLE-0000', (430, 60), 0.6)
    put(img, 'L18-5 / Breast', (790, 62), 0.55)
    for i, t in enumerate(['B', 'Gen/H', 'M 11/68 dB/Med', 'T 1480 m/s', 'G 55 %', 'Fr. 49 Hz']):
        put(img, t, (14, 130 + 24 * i), 0.45, (190, 190, 190))
    put(img, 'Z 100 %', (14, 330), 0.45, (190, 190, 190))
    for k, lab in enumerate(['0.0', '1.0', '2.0', '3.0']):                                # depth ruler
        yy = FY0 + int(k * (FY1 - FY0) / 3.6)
        cv2.line(img, (FX1 + 6, yy), (FX1 + 20, yy), (200, 200, 200), 1)
        put(img, lab, (FX1 + 26, yy + 5), 0.45, (200, 200, 200))
    img[140:340, 1010:1030] = np.tile(np.linspace(255, 20, 200)[:, None, None], (1, 20, 3)).astype(np.uint8)
    put(img, '+Dist 10.1 mm', (FX1 - 190, FY1 - 62), 0.5, (255, 255, 255), 1)             # measurement text ON the picture
    put(img, 'xDist 4.0 mm', (FX1 - 190, FY1 - 38), 0.5, (255, 255, 255), 1)
    put(img, 'Fr: 100/100', (14, 792), 0.45, (190, 190, 190))
    put(img, '*', (1000, 796), 0.9, (210, 190, 120), 2)
    if info['lesion']:                                                                      # calipers
        cx, cy, a, b = info['lesion']
        for p, q in (((cx - a - 6, cy), (cx + a + 6, cy)), ((cx, cy - b - 6), (cx, cy + b + 6))):
            cv2.line(img, p, q, (235, 235, 235), 1, cv2.LINE_AA)
        for p in ((cx - a - 6, cy), (cx + a + 6, cy), (cx, cy - b - 6), (cx, cy + b + 6)):
            put(img, 'x', (p[0] - 4, p[1] + 4), 0.4, (255, 255, 255))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), info


def field_poly(info_field=(FX0, FY0, FX1, FY1), jitter=2, seed=0, n_side=2):
    """A hand-drawn-looking polygon around the field: ~10 points with a little wobble."""
    r = np.random.default_rng(seed)
    x0, y0, x1, y1 = info_field
    xs, ys = np.linspace(x0, x1, n_side + 2), np.linspace(y0, y1, n_side + 2)
    pts = ([(x, y0) for x in xs] + [(x1, y) for y in ys[1:]] +
           [(x, y1) for x in xs[::-1][1:]] + [(x0, y) for y in ys[::-1][1:-1]])
    return [(int(px + r.integers(-jitter, jitter + 1)), int(py + r.integers(-jitter, jitter + 1))) for px, py in pts]


def draw_poly(ax, pts, color, lw=2.2, fill=0.16, dots=True):
    ax.add_patch(Polygon(pts, closed=True, fill=True, fc=color, alpha=fill, ec='none'))
    ax.add_patch(Polygon(pts, closed=True, fill=False, ec=color, lw=lw))
    if dots:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], 'o', color=color, ms=3.6, mec='white', mew=0.5)


def clean(ax):
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')


# ---------------------------------------------------------------- Figure 1: anatomy
def fig_anatomy():
    img, info = frame(seed=3, lesion=True, dark_left=0, shadow=0)
    fig, ax = plt.subplots(figsize=(9.2, 6.9))
    ax.imshow(img); clean(ax)
    ax.add_patch(plt.Rectangle((FX0, FY0), FX1 - FX0, FY1 - FY0, fill=False, ec=GREEN, lw=2.6, ls='--'))

    def tag(txt, xy, xytext, ok):
        c = GREEN if ok else RED
        ax.annotate(txt, xy=xy, xytext=xytext, color='white', fontsize=8.5, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.25', fc=c, ec='none'),
                    arrowprops=dict(arrowstyle='->', color=c, lw=1.8), ha='center', va='center')

    tag('HEADER: outside\n(clinic, date, patient ID)', (500, 45), (610, 100), False)
    tag('SETTINGS COLUMN:\noutside', (70, 220), (80, 410), False)
    tag('DEPTH RULER:\noutside', (FX1 + 40, 290), (975, 470), False)
    tag('SKIN LINE: top edge\nof the image field', (330, FY0 + 2), (270, 80), True)
    tag('IMAGE FIELD\n(inside the mask)', (FX0 + 40, 560), (330, 745), True)
    tag('dark lesion:\nINSIDE', (info['lesion'][0] + 62, info['lesion'][1] + 22), (690, 330), True)
    tag('calipers and measurement text\nlying on the picture: INSIDE', (FX1 - 130, FY1 - 50), (560, 748), True)
    tag('FOOTER: outside', (70, 792), (190, 770), False)
    fig.text(0.5, 0.012, 'Synthetic frame for illustration (no patient data).   Green = goes inside the mask,  red = stays outside.',
             ha='center', fontsize=8.5, color='#444')
    plt.tight_layout(pad=0.3, rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(OUT, 'guide_anatomy.png'), dpi=130, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


# ---------------------------------------------------------------- Figure 2: correct examples
def fig_correct():
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.45))
    specs = [
        dict(kw=dict(seed=4, lesion=True), title='(a) typical frame\none polygon, ~10 points'),
        dict(kw=dict(seed=6, lesion=True, dark_left=170), title='(b) black strip where the probe\nlost contact: still INSIDE'),
        dict(kw=dict(seed=8, lesion=True, dual=True), title='(c) two panels side by side:\none polygon per panel'),
    ]
    for ax, sp in zip(axes, specs):
        img, info = frame(**sp['kw'])
        ax.imshow(img); clean(ax); ax.set_title(sp['title'], fontsize=9)
        for k, p in enumerate(info['panels']):
            draw_poly(ax, field_poly(p, jitter=2, seed=k + 3, n_side=1 if len(info['panels']) > 1 else 2), GREEN)
    fig.text(0.5, 0.005, 'Synthetic frames (no patient data). Corners need not be pixel-perfect.', ha='center', fontsize=8.5, color='#444')
    plt.tight_layout(pad=0.3, rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(OUT, 'guide_correct.png'), dpi=130, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


# ---------------------------------------------------------------- Figure 3: common mistakes
def fig_mistakes():
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.9))
    base = (FX0, FY0, FX1, FY1)

    def notch_poly(info):
        cx, cy, a, b = info['lesion']
        top = [(FX0, FY0), (FX1, FY0), (FX1, FY1), (FX0, FY1)]
        # a rectangle with a notch bitten out around the lesion, reached from the left side
        return [(FX0, FY0), (FX1, FY0), (FX1, FY1), (FX0, FY1), (FX0, cy + b + 14), (cx - a - 18, cy + b + 14),
                (cx - a - 18, cy - b - 14), (cx + a + 18, cy - b - 14), (cx + a + 18, cy + b + 14), (FX0, cy + b + 14)]

    cases = [
        ('WRONG: lesion left out', dict(seed=11), lambda i: notch_poly(i), RED,
         'The dark lesion is exactly what the\nclassifier needs. Never cut it out.'),
        ('WRONG: depth ruler included', dict(seed=12), lambda i: field_poly((FX0, FY0, FX1 + 110, FY1), 2, 1), RED,
         'Rulers and numbers beside the\npicture are outside the field.'),
        ('WRONG: header included', dict(seed=13), lambda i: field_poly((FX0 - 150, 20, FX1, FY1), 2, 2), RED,
         'The header carries the patient ID\nand is never part of the field.'),
        ('WRONG: skin line cut off', dict(seed=14), lambda i: field_poly((FX0, FY0 + 70, FX1, FY1), 2, 3), RED,
         'The field starts AT the skin line\n(bright line at the very top).'),
        ('WRONG: black area left out', dict(seed=15, shadow=0.32), lambda i: field_poly((FX0, FY0, FX1, FY1 - 190), 2, 4), RED,
         'Protocol v1: dark areas inside the\nfield stay in, down to its bottom edge.'),
        ('RIGHT', dict(seed=16, shadow=0.32), lambda i: field_poly(base, 2, 5), GREEN,
         'The whole field, dark parts and\nthe lesion included; nothing outside.'),
    ]
    for ax, (title, kw, polyf, col, note) in zip(axes.ravel(), cases):
        img, info = frame(lesion=True, **kw)
        ax.imshow(img); clean(ax)
        draw_poly(ax, polyf(info), col)
        ax.set_title(title, fontsize=10, color=col, fontweight='bold')
        ax.text(W / 2, H - 4, note, ha='center', va='bottom', color='white', fontsize=8.2,
                bbox=dict(fc='black', ec='none', alpha=0.6, pad=2))
    fig.text(0.5, 0.005, 'Synthetic frames (no patient data).', ha='center', fontsize=8.5, color='#444')
    plt.tight_layout(pad=0.4, rect=(0, 0.02, 1, 1))
    fig.savefig(os.path.join(OUT, 'guide_mistakes.png'), dpi=130, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


if __name__ == '__main__':
    fig_anatomy(); fig_correct(); fig_mistakes()
    for f in ('guide_anatomy.png', 'guide_correct.png', 'guide_mistakes.png'):
        a = cv2.imread(os.path.join(OUT, f))
        print(f, a.shape[1], 'x', a.shape[0])
