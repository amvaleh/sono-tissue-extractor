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

**Second institution, scored against hand-drawn masks.** The frames are 1080×810 from a
different scanner layout, with a different vendor header, mostly single-pane, with
caliper text and colour Doppler bars. The model never saw this site in training. One
annotator drew masks on 30 frames (one per patient, chosen at random), following
[protocol v1](docs/ANNOTATION_GUIDE.md): the whole image field, dark areas included.

| method | Dice, mean [95% bootstrap CI] |
|--------|------------------------------:|
| this model, default settings (holes filled) | 0.9746 [0.9666, 0.9815] |
| this model, raw network output | 0.9722 [0.9639, 0.9797] |
| this model reduced to its bounding rectangle | 0.9708 [0.9630, 0.9775] |
| constant mask (leave-one-out majority of the other frames) | 0.8988 [0.8630, 0.9302] |
| BUSClean scan-area crop, run unmodified | 0.9643 [0.9428, 0.9805] |
| the classical teacher in this repository | 0.36 (empty mask on 18 of 30 frames) |

How to read this:

- **The model transfers to this site; the classical teacher does not.** The teacher's
  thresholds are tuned to the first scanner and fail on this one. That is a statement
  about *those thresholds*, not about rule-based cropping in general.
- **The model and BUSClean are not distinguishable on these frames.** Paired difference
  (model minus BUSClean): +0.010 [−0.002, +0.027] with the default model, and +0.007
  [−0.010, +0.029] when the model is reduced to a rectangle, the same output type as
  BUSClean. Both intervals span zero. We make **no claim of better transfer** than
  BUSClean. BUSClean's worst frame scored 0.75 against 0.91 for the model; both of its low
  frames are cases where it trimmed a dark part of the picture, which is a boundary
  judgement more than a gross failure.
- **A constant mask is already strong (0.90)**, because the layout of these scanners is
  nearly fixed. The model's contribution is the step from about 0.90 to about 0.97.
- **Limits.** 30 frames, one annotator (so no ceiling: some of the remaining error is
  ordinary disagreement about an edge), and the two sites differ in file format as well as
  scanner (PNG against JPEG), so the two cannot be separated. The reference masks are
  nearly rectangular, which favours rectangular outputs.

Reproduce, or extend with your own frames, with the tools described under
[Measuring accuracy on your own data](#measuring-accuracy-on-your-own-data).

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


### Measuring accuracy on your own data

A number for a new scanner needs masks that a person drew. Two tools handle that, plus a
written protocol so that everyone draws by the same rule.

```bash
# pick frames (one per patient, random) and copy them to ./annotation_work/images
python src/annotate_masks.py prepare --raw-dir /path/to/frames --n 30

# draw masks; every annotator gets a separate folder
python src/annotate_masks.py draw --annotator alice

# sanity-check them (--header-rows = height of the scanner's header strip in pixels)
python src/annotate_masks.py check --annotator alice --header-rows 100

# score the model, with the baselines that make the number meaningful
python src/evaluate_masks.py --annotator alice --model models/unet_tissue_v1.pt \
    --constant-baseline --busclean-repo /path/to/bus-cleaning
```

**Read the [annotation guide](docs/ANNOTATION_GUIDE.md) before drawing.** It states the rule
(protocol v1: one polygon around the whole image field, dark areas included, header and
settings excluded), with illustrated examples and the mistakes to avoid.

With **two annotators** on the same frames you can also measure how much people disagree,
which is the ceiling for any automatic method:

```bash
python src/annotate_masks.py agreement --a alice --b bob
```

`prepare` can be re-run with a larger `--n`; it only adds frames and never renumbers or
overwrites the ones you have already drawn on. `--busclean-repo` is optional and points at a
clone of [hawaii-ai/bus-cleaning](https://github.com/hawaii-ai/bus-cleaning); it is used
only to score BUSClean's crop and is not redistributed here.

---

## Repository contents

```
src/make_tissue_masks.py   classical teacher — generates training masks
src/train_unet.py          U-Net training with augmentation
src/extract_tissue.py      inference and crop extraction
src/annotate_masks.py      draw reference masks by hand; check them; measure agreement
src/evaluate_masks.py      score the model (and baselines) against hand-drawn masks
docs/ANNOTATION_GUIDE.md   how to draw the masks (protocol v1), with illustrated examples
docs/figures/              synthetic example frames for the guide, and the script that makes them
models/unet_tissue_v1.pt   trained weights (Dice 0.9638 on held-out patients, site 1)
models/training_history.csv per-epoch metrics
```

Source comments, identifiers and console output are in English.

**No patient data is included in this repository**, and none can be
redistributed. The trained weights are derived from a private clinical dataset.

---

## Limitations

- **Cross-site accuracy rests on one second institution, 30 frames and one annotator.**
  See the limits listed under Results. Behaviour on other vendors, probes or modalities is
  unknown, and the file-format difference between the two sites (PNG against JPEG) is
  confounded with the scanner difference.
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
