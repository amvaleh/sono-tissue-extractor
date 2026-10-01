#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a U-Net to segment ultrasound tissue (the "student" stage)
================================================================

What this script does
    Learns, from pairs of (raw frame, mask), which pixels of an
    ultrasound frame are tissue.

Why train a model when the rule-based code already scores ~99%?
    The rules in make_tissue_masks.py are tuned to one scanner. On a
    different machine, with different brightness and different on-screen
    text, they are likely to break.

    This model sees deliberately corrupted frames during training —
    random text pasted on, brightness shifted, noise added. That forces
    it to learn *the tissue itself* rather than memorising one
    scanner's appearance.

    In short: the rules are the teacher, this is the student, and the
    student is meant to be more robust than the teacher.

Requires
    make_tissue_masks.py must have been run first, so the mask
    directory is populated.

Usage
    # quick check that it runs at all
    python src/train_unet.py --raw-dir ... --mask-dir ... --epochs 2

    # real training
    python src/train_unet.py --raw-dir ... --mask-dir ...

Outputs
    unet_run/best.pt         best checkpoint
    unet_run/last.pt         most recent checkpoint
    unet_run/history.csv     per-epoch metrics
    unet_run/preview_*.jpg   sample outputs on validation frames
"""

import argparse
import csv
import os
import glob
import random
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ===========================================================================
# Settings
# ===========================================================================

IMG_H, IMG_W = 384, 512     # frames are resized to this for training.
                            # The source is 1080x1440, so the aspect
                            # ratio is preserved.
                            # Larger -> more precise but slower.

VAL_RATIO = 0.20            # fraction of *patients* held out for validation


# ===========================================================================
# Part 1 — reading the data
# ===========================================================================

def build_pairs(raw_dir, mask_dir):
    """
    Match each mask to its source frame.

    Mask filenames look like:  2138__2024_11_10_131414_682.png
    that is: patient id + double underscore + the original frame name.
    """
    raws = {}
    for p in glob.glob(os.path.join(raw_dir, '*', '**', '*.png'), recursive=True):
        raws[os.path.splitext(os.path.basename(p))[0]] = p

    pairs = []
    for m in sorted(glob.glob(os.path.join(mask_dir, '*.png'))):
        stem = os.path.splitext(os.path.basename(m))[0]
        if '__' not in stem:
            continue
        patient, raw_stem = stem.split('__', 1)
        if raw_stem in raws:
            pairs.append({'raw': raws[raw_stem], 'mask': m, 'patient': patient})
    return pairs


def split_by_patient(pairs, val_ratio, seed=42):
    """
    Split train/validation by *patient*, never by image.

    Why this matters so much:
        Each patient contributes several frames that look nearly
        identical. Split at random and frame 1 of a patient lands in
        training while frame 2 lands in validation — the model has
        already seen the answer. The score comes out excellent and
        means nothing.
    """
    patients = sorted({p['patient'] for p in pairs})
    rng = random.Random(seed)
    rng.shuffle(patients)
    n_val = max(1, int(len(patients) * val_ratio))
    val_set = set(patients[:n_val])

    train = [p for p in pairs if p['patient'] not in val_set]
    val = [p for p in pairs if p['patient'] in val_set]
    return train, val


# ===========================================================================
# Part 2 — deliberate corruption (the most important part)
# ===========================================================================
#
# This is what lets the model work on other scanners.
#
# The model is shown frames whose text, brightness and noise have been
# randomised, while *the mask stays exactly the same*.
#
# The message to the model: "however these things change, the answer
# does not. So stop looking at them and look at the tissue."
#
# Note: text is pasted onto the image but the mask is left untouched.
# So when text lands on tissue, the model must still call that region
# tissue — it has to read through the overlay.

_FONTS = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX,
          cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_TRIPLEX]

_WORDS = ['RT AXILLA', 'LT AXILLA', 'BREAST', 'L18-5', 'MI 0.9',
          'Gen/H', 'SC/SR 2', 'Z 100 %', 'Fr. 23 Hz', 'G 48 %',
          'TIS 0.4', 'D 5.0 cm', '12:34:56', 'RT 4-5 NZ']


def aug_add_text(img, rng):
    """Paste random text onto the frame, mimicking scanner annotations."""
    H, W = img.shape[:2]
    for _ in range(rng.randint(1, 6)):
        txt = rng.choice(_WORDS)
        scale = rng.uniform(0.4, 1.3)
        thick = rng.randint(1, 3)
        x = rng.randint(0, max(1, W - 120))
        y = rng.randint(20, H - 10)
        shade = rng.randint(180, 255)
        cv2.putText(img, txt, (x, y), rng.choice(_FONTS), scale,
                    (shade, shade, shade), thick, cv2.LINE_AA)
    return img


def aug_add_marks(img, rng):
    """Random rulers, ticks, arrows and coloured marks."""
    H, W = img.shape[:2]
    for _ in range(rng.randint(0, 4)):
        kind = rng.random()
        if kind < 0.4:                                  # edge ruler
            x = rng.randint(0, W - 1)
            for y in range(rng.randint(0, 40), H, rng.randint(25, 60)):
                cv2.line(img, (x, y), (x + rng.randint(4, 12), y),
                         (230, 230, 230), 1)
        elif kind < 0.7:                                # coloured marker
            c = [(0, 165, 255), (255, 200, 0), (0, 0, 255)][rng.randint(0, 2)]
            cv2.putText(img, 'S', (rng.randint(0, W - 30), rng.randint(20, 60)),
                        cv2.FONT_HERSHEY_SIMPLEX, rng.uniform(0.8, 1.6), c, 3)
        else:                                           # thin guide line
            if rng.random() < 0.5:
                y = rng.randint(0, H - 1)
                cv2.line(img, (0, y), (W, y), (200, 200, 200), 1)
            else:
                x = rng.randint(0, W - 1)
                cv2.line(img, (x, 0), (x, H), (200, 200, 200), 1)
    return img


def aug_photometric(img, rng):
    """
    Randomise brightness, contrast and gamma.

    This is exactly what caused the "dark panel" failure in the
    rule-based code. By seeing both dark and bright examples, the model
    learns to decide independently of overall brightness.
    """
    f = img.astype(np.float32)
    f = f * rng.uniform(0.55, 1.45)                    # overall brightness
    f = (f - 128) * rng.uniform(0.6, 1.5) + 128        # contrast
    f = np.clip(f, 0, 255)
    gamma = rng.uniform(0.65, 1.5)                     # gamma
    f = 255.0 * ((f / 255.0) ** gamma)
    return np.clip(f, 0, 255).astype(np.uint8)


def aug_noise(img, rng):
    """
    Add speckle noise.

    Different machines produce different speckle characteristics. This
    stops the model depending on one particular grain.
    """
    f = img.astype(np.float32)
    if rng.random() < 0.5:
        f *= 1.0 + np.random.normal(0, rng.uniform(0.04, 0.16), f.shape)
    else:
        f += np.random.normal(0, rng.uniform(3, 14), f.shape)
    if rng.random() < 0.3:                             # slight blur
        k = rng.choice([3, 5])
        f = cv2.GaussianBlur(f, (k, k), 0)
    return np.clip(f, 0, 255).astype(np.uint8)


def aug_geometry(img, mask, rng):
    """Random crop and flip. Here the mask must change with the image."""
    H, W = img.shape[:2]
    if rng.random() < 0.5:                             # horizontal flip
        img, mask = img[:, ::-1].copy(), mask[:, ::-1].copy()
    if rng.random() < 0.7:                             # crop and zoom
        s = rng.uniform(0.80, 1.0)
        h, w = int(H * s), int(W * s)
        y0 = rng.randint(0, H - h); x0 = rng.randint(0, W - w)
        img = cv2.resize(img[y0:y0 + h, x0:x0 + w], (W, H),
                         interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask[y0:y0 + h, x0:x0 + w], (W, H),
                          interpolation=cv2.INTER_NEAREST)
    return img, mask


class SonoDataset(Dataset):
    """
    Each time the model asks for a frame, this class reads it, resizes
    it, and (in training mode) corrupts it at random.
    """

    def __init__(self, pairs, train=True):
        self.pairs = pairs
        self.train = train

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        rec = self.pairs[i]
        img = cv2.imread(rec['raw'], cv2.IMREAD_COLOR)
        mask = cv2.imread(rec['mask'], cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)

        if self.train:
            rng = random.Random()
            img, mask = aug_geometry(img, mask, rng)
            # Order matters: text first, then the brightness change, so
            # the text is affected too and does not look pasted on.
            if rng.random() < 0.8:
                img = aug_add_text(img.copy(), rng)
            if rng.random() < 0.6:
                img = aug_add_marks(img, rng)
            if rng.random() < 0.9:
                img = aug_photometric(img, rng)
            if rng.random() < 0.7:
                img = aug_noise(img, rng)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        return (torch.from_numpy(gray)[None],
                torch.from_numpy(mask)[None])


# ===========================================================================
# Part 3 — the U-Net
# ===========================================================================
#
# A U-Net is shaped like the letter U:
#   left side (down)  : shrinks the image and captures "meaning"
#   right side (up)   : expands it again so every pixel gets an answer
#   skip connections  : carry edge detail straight across, so the
#                       tissue boundary comes out sharp

def conv_block(cin, cout):
    """Two convolutions back to back, with normalisation and ReLU."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, base=24):
        super().__init__()
        b = base
        self.d1 = conv_block(1, b)            # 384x512
        self.d2 = conv_block(b, b * 2)        # 192x256
        self.d3 = conv_block(b * 2, b * 4)    # 96x128
        self.d4 = conv_block(b * 4, b * 8)    # 48x64
        self.bottom = conv_block(b * 8, b * 16)

        self.u4 = nn.ConvTranspose2d(b * 16, b * 8, 2, 2)
        self.c4 = conv_block(b * 16, b * 8)
        self.u3 = nn.ConvTranspose2d(b * 8, b * 4, 2, 2)
        self.c3 = conv_block(b * 8, b * 4)
        self.u2 = nn.ConvTranspose2d(b * 4, b * 2, 2, 2)
        self.c2 = conv_block(b * 4, b * 2)
        self.u1 = nn.ConvTranspose2d(b * 2, b, 2, 2)
        self.c1 = conv_block(b * 2, b)

        self.out = nn.Conv2d(b, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.pool(d1))
        d3 = self.d3(self.pool(d2))
        d4 = self.d4(self.pool(d3))
        bt = self.bottom(self.pool(d4))

        x = self.c4(torch.cat([self.u4(bt), d4], 1))
        x = self.c3(torch.cat([self.u3(x), d3], 1))
        x = self.c2(torch.cat([self.u2(x), d2], 1))
        x = self.c1(torch.cat([self.u1(x), d1], 1))
        return self.out(x)                    # raw logits, no sigmoid


# ===========================================================================
# Part 4 — loss and metrics
# ===========================================================================

def dice_loss(logits, target, eps=1.0):
    """
    Dice measures how much two shapes overlap.
    1 = identical, 0 = no overlap at all.

    Why isn't BCE alone enough?
        BCE looks at each pixel independently. If 80% of the frame is
        background, the model can score well simply by answering
        "background everywhere". Dice looks at the shape as a whole and
        closes off that shortcut.
    """
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(1, 2, 3)) + eps
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return 1 - (num / den).mean()


@torch.no_grad()
def evaluate(model, loader, device):
    """Compute Dice and IoU over the validation set."""
    model.eval()
    dices, ious = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        p = (torch.sigmoid(model(x)) > 0.5).float()
        inter = (p * y).sum(dim=(1, 2, 3))
        union = ((p + y) > 0).float().sum(dim=(1, 2, 3))
        tot = p.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3))
        dices += ((2 * inter + 1) / (tot + 1)).cpu().tolist()
        ious += ((inter + 1) / (union + 1)).cpu().tolist()
    return float(np.mean(dices)), float(np.mean(ious))


@torch.no_grad()
def save_preview(model, pairs, device, path, n=4):
    """Put model outputs next to the targets, for visual checking."""
    model.eval()
    tiles = []
    for rec in pairs[:n]:
        img = cv2.imread(rec['raw'], cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        gt = cv2.resize(cv2.imread(rec['mask'], cv2.IMREAD_GRAYSCALE),
                        (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST) > 127

        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        t = torch.from_numpy(g)[None, None].to(device)
        pr = (torch.sigmoid(model(t))[0, 0].cpu().numpy() > 0.5)

        a = img.copy(); a[gt] = (0.5 * a[gt] + 0.5 * np.array([0, 220, 0])).astype(np.uint8)
        b = img.copy(); b[pr] = (0.5 * b[pr] + 0.5 * np.array([0, 160, 255])).astype(np.uint8)
        cv2.putText(a, 'TEACHER', (10, 28), 0, 0.8, (255, 255, 255), 2)
        cv2.putText(b, 'UNET', (10, 28), 0, 0.8, (255, 255, 255), 2)
        tiles.append(cv2.hconcat([a, b]))
    if tiles:
        cv2.imwrite(path, cv2.vconcat(tiles), [cv2.IMWRITE_JPEG_QUALITY, 88])


# ===========================================================================
# Part 5 — the training loop
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description='Train the tissue-segmentation U-Net')
    ap.add_argument('--raw-dir', required=True)
    ap.add_argument('--mask-dir', required=True)
    ap.add_argument('--out-dir', default='unet_run')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--base', type=int, default=24, help='model width')
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- compute device ---
    if torch.backends.mps.is_available():
        device = torch.device('mps')          # Apple Silicon GPU
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'Device: {device}')

    # --- data ---
    pairs = build_pairs(args.raw_dir, args.mask_dir)
    if not pairs:
        print('\nError: no image/mask pairs found.')
        print('Run this first:  python src/make_tissue_masks.py')
        return

    train_pairs, val_pairs = split_by_patient(pairs, VAL_RATIO)
    n_tr_pat = len({p["patient"] for p in train_pairs})
    n_va_pat = len({p["patient"] for p in val_pairs})
    print(f'Total pairs : {len(pairs)}')
    print(f'Train       : {len(train_pairs):4d} frames from {n_tr_pat} patients')
    print(f'Validation  : {len(val_pairs):4d} frames from {n_va_pat} patients')
    print('(split by patient — no patient appears on both sides)\n')

    tr_loader = DataLoader(SonoDataset(train_pairs, True), args.batch_size,
                           shuffle=True, num_workers=args.workers,
                           drop_last=True, persistent_workers=args.workers > 0)
    va_loader = DataLoader(SonoDataset(val_pairs, False), args.batch_size,
                           shuffle=False, num_workers=args.workers,
                           persistent_workers=args.workers > 0)

    # --- model ---
    model = UNet(base=args.base).to(device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {n_param/1e6:.2f} M\n')

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    bce = nn.BCEWithLogitsLoss()

    hist_path = os.path.join(args.out_dir, 'history.csv')
    hist = open(hist_path, 'w', newline='', encoding='utf-8')
    writer = csv.writer(hist)
    writer.writerow(['epoch', 'train_loss', 'val_dice', 'val_iou', 'lr', 'sec'])

    best = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            # Call the model *once* and keep the output. Writing
            # model(x) twice makes the network run the whole forward
            # pass twice for the same answer, doubling time and memory.
            logits = model(x)
            # final loss = half BCE + half Dice
            loss = 0.5 * bce(logits, y) + 0.5 * dice_loss(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()

        dice, iou = evaluate(model, va_loader, device)
        dt = time.time() - t0
        lr_now = opt.param_groups[0]['lr']
        print(f'epoch {ep:3d}/{args.epochs}  loss {np.mean(losses):.4f}  '
              f'Dice {dice:.4f}  IoU {iou:.4f}  ({dt:.0f}s)')
        writer.writerow([ep, round(float(np.mean(losses)), 5),
                         round(dice, 5), round(iou, 5),
                         f'{lr_now:.2e}', round(dt, 1)])
        hist.flush()

        torch.save({'model': model.state_dict(), 'base': args.base,
                    'img_h': IMG_H, 'img_w': IMG_W, 'epoch': ep},
                   os.path.join(args.out_dir, 'last.pt'))
        if dice > best:
            best = dice
            torch.save({'model': model.state_dict(), 'base': args.base,
                        'img_h': IMG_H, 'img_w': IMG_W,
                        'epoch': ep, 'val_dice': dice},
                       os.path.join(args.out_dir, 'best.pt'))
            save_preview(model, val_pairs, device,
                         os.path.join(args.out_dir, 'preview_best.jpg'))

    hist.close()
    print('\n' + '=' * 58)
    print(f'Best validation Dice : {best:.4f}')
    print(f'Model saved to       : {args.out_dir}/best.pt')
    print(f'Sample output        : {args.out_dir}/preview_best.jpg')
    print('\nReading the Dice score:')
    print('   above 0.95   excellent')
    print('   0.90 - 0.95  good')
    print('   below 0.85   something is wrong')
    print('\nAlways look at preview_best.jpg.')
    print('Green = the teacher\'s answer, orange = the model\'s.')
    print('=' * 58)


if __name__ == '__main__':
    main()
