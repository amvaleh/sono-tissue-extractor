# sono-tissue-extractor

Automatic tissue-region extraction from breast/axillary ultrasound images.

Given a raw ultrasound screenshot — complete with vendor header, sidebar
settings, depth rulers, caliper annotations and black background — this tool
returns only the diagnostic tissue region, cropped and ready for a downstream
classifier.

It was built to remove a manual bottleneck: hand-cropping thousands of
ultrasound frames before they can be used to train an axillary lymph node
classification model.

---

## The problem

A raw ultrasound frame is mostly *not* tissue. A typical frame contains:

- a vendor header with institution name, date and patient identifiers
- a sidebar of acquisition settings (gain, frequency, depth)
- depth rulers and tick marks along the edges
- caliper measurements and text annotations drawn over the image
- large areas of black acoustic shadow below the useful signal
- sometimes a colour Doppler bar, or two side-by-side scan panels

Feeding all of that to a classifier wastes capacity on vendor furniture.
Cropping it away by hand does not scale past a few hundred images.

## The approach: a classical teacher, a learned student

The core difficulty is that there are no pixel-level tissue annotations, and
producing them by hand is exactly the cost we are trying to avoid.

This tool sidesteps that with a two-stage design:

**Stage 1 — the teacher (`make_tissue_masks.py`).**
A rule-based segmenter that exploits three measurable properties separating
ultrasound tissue from everything else in the frame. Measured on real data:

| region         | brightness | speckle (local σ) | colourfulness |
|----------------|-----------:|------------------:|--------------:|
| tissue         |         65 |                15 |             4 |
| black background |       17 |                 4 |             1 |
| white text     |        157 |               100 |             0 |
| coloured marks |        110 |                84 |           127 |

Tissue is mid-brightness, moderately speckled and grey. Text is sharp and
bright; background is flat and dark; vendor marks are saturated. Thresholds
are computed *per panel* rather than globally, because the two panels of a
dual-pane frame can differ substantially in gain (we measured 64 vs 48).

The teacher produces pixel masks for the entire training set at no annotation
cost. It is not the deliverable — its thresholds are tuned to one scanner.

**Stage 2 — the student (`train_unet.py`).**
A compact U-Net trained on the teacher's masks, under deliberately aggressive
augmentation: synthetic text pasted over the image, randomised brightness,
contrast and gamma, speckle and Gaussian noise, fake rulers and coloured
marks, flips and random crops.

The masks are *not* augmented alongside the pasted text. When text lands on
tissue, the mask still says tissue — so the network is forced to read the
texture underneath rather than keying on the overlay. This is what lets the
student outperform the teacher it learned from on unseen acquisition setups.

**Inference (`extract_tissue.py`).**
Runs the trained student, splits side-by-side panels where present, and writes
per-panel crops plus a manifest linking each crop to its source frame.

### A note on dark regions

The masks deliberately retain hypoechoic (dark) regions *inside* the tissue
band. Lymph nodes and vessels are dark on B-mode ultrasound; excluding dark
pixels would discard precisely the structures the downstream classifier needs.

The network does not learn this perfectly, so inference fills any enclosed
hole in the mask by default. On a 7288-frame run this mattered: the mean hole
fraction was only 0.15%, but the worst frames lost 19–27% of their interior,
and inspection showed those holes were large lesions — one of them carrying
the radiologist's own caliper measurement across it. Filling removed every
hole across all 7288 frames while leaving mean mask area unchanged at 39.5%,
confirming it recovers interior structure rather than inflating the region.
Disable with `--no-fill-holes` if you want the raw network output.

---

## Results

**Held-out patients, same site** (41 patients / 91 frames, split by patient so
no patient appears in both train and test):

| metric | value |
|--------|------:|
| Dice   | 0.9638 |
| IoU    | 0.9315 |

Agreement with the teacher's masks, as an upper bound on what the student was
trained to reproduce.

**Independent crop recovery** (512 operator crops, located in their source
frames by multi-scale template matching): the predicted tissue region contains
the operator's manual crop with **100.0%** mean coverage (worst case 98.5%).
Panel count agrees with the operator's in 89% of frames; most disagreements
are frames where the operator cropped only one of two visible panels.

**Cross-site transfer** (second institution, different scanner layout — 810×1080
portrait instead of 1440×1080 landscape, different vendor header, predominantly
single-pane, with caliper text and colour Doppler bars; 120 frames sampled):

| proxy metric | value |
|--------------|------:|
| leakage into vendor header | 0.00% |
| interior holes in mask | 0.3% mean |
| frames with >5% holes | 2 / 120 |

⚠️ **These cross-site numbers are proxies, not accuracy.** No ground-truth
annotations exist for the second site, so they show only that the output is
well-formed, not that it is correct. Establishing genuine cross-site accuracy
requires manually annotating a sample from that site. Treat transfer as
promising but unproven.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Runs on Apple Silicon (MPS), CUDA, or CPU — the device is selected
automatically. Training the included model took ~30 minutes on an M-series Mac.

## Usage

### Extract tissue using the included model

```bash
python src/extract_tissue.py \
    --input  /path/to/images \
    --out-dir ./extracted \
    --model  models/unet_tissue_v1.pt \
    --save-overlay
```

`--input` accepts a single file, a directory (searched recursively) or a glob.

Outputs:

```
extracted/
├── crops/        tissue crops — the input to your downstream model
├── masks/        binary masks
├── overlays/     visual check images (with --save-overlay)
└── manifest.csv  crop → source frame, with coordinates
```

**Always look at the overlays before trusting a batch.** Every real defect
found during development was spotted by eye first and only then confirmed in
the metrics.

### Key options

| flag | purpose |
|------|---------|
| `--split-width-frac` | A region wider than this fraction of the image is treated as two panels and split. Default `0.55` suits dual-pane layouts. **Set to `2` for single-pane images** or they will be split in half. |
| `--threshold` | Decision threshold, default `0.5`. Lower it (e.g. `0.3`) if tissue is being missed. |
| `--pad` | Pixels of margin to add around each crop. |
| `--blacken-outside` | Black out everything outside the tissue shape, instead of a rectangular crop. |
| `--no-fill-holes` | Keep enclosed holes in the mask instead of filling them. See the note on dark regions above. |

`--split-width-frac` is the one setting that reliably needs attention on a new
scanner, since it encodes the panel layout.

### Retraining on your own data

```bash
# 1. generate masks with the classical teacher
python src/make_tissue_masks.py --raw-dir /path/to/raw --out-dir ./work

# 2. inspect work/overlays/ and work/mask_qc.csv before proceeding
#    (a bad teacher produces a bad student)

# 3. train
python src/train_unet.py --raw-dir /path/to/raw --mask-dir ./work/masks
```

The teacher's thresholds live in a clearly marked block at the top of
`make_tissue_masks.py` and are commented with what each one controls.

Training splits **by patient**, never by image. Frames from one patient are
near-duplicates; an image-level split leaks and inflates the reported score.

---

## Repository contents

```
src/make_tissue_masks.py   classical teacher — generates training masks
src/train_unet.py          U-Net training with augmentation
src/extract_tissue.py      inference and crop extraction
models/unet_tissue_v1.pt   trained weights (Dice 0.9638)
models/training_history.csv per-epoch metrics
```

Source comments, identifiers and console output are in English.

**No patient data is included in this repository**, and none can be
redistributed. The trained weights are derived from a private clinical dataset.

---

## Limitations

- **Cross-site accuracy is unverified.** See the caveat under Results.
- **Trained on B-mode images from two Iranian centres**, both using an L18-5
  probe. Behaviour on other probes, modalities or vendors is unknown.
- **Panel splitting is geometric.** We measured no detectable divider between
  side-by-side panels within the tissue band (brightness ratio 1.01 against
  neighbouring columns), so the split is placed at the region's midpoint —
  accurate to roughly 10 px against a typical 37 px inter-panel gap. It will
  not handle unevenly sized panels.
- **The method is not novel.** Classical-teacher/learned-student is standard
  weak supervision. The contribution here is a working, documented tool for a
  specific and tedious task, not a new technique.
- **Not a lymph node detector.** It localises the diagnostic tissue region, not
  individual nodes.

## Privacy note

Raw ultrasound frames routinely carry patient identifiers burned into the
header pixels (medical record numbers, names, dates). This tool's crops
normally exclude the header, but **verify this before publishing any figure
derived from clinical images**, and follow your institution's ethics approval
and data governance requirements.

## License

MIT — see [LICENSE](LICENSE).

## Citation

If this software is useful in your work, please cite the repository. A
software paper is in preparation; this section will be updated with its
reference.
