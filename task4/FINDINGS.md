# Task 4 findings (working notes, not the hand-in report)

This file is input for your own `REPORT.md`. Write the submission yourself.

## Setup

- Data: A = course photos (101 train / 28 val). B = another group's photos
  (107 / 29). H = historical set (700 / 120). Eval: `valA` = A val,
  `cameraA` = 5 fixed augmented copies of each A-val photo (140 images),
  `testB` / `testH` = other groups' photos, never used for model selection.
- Camera-like set uses fixed seed 0, generated once: random resized crop
  0.6–1.0, rotation ±20°, colour jitter 0.3/0.3/0.2/0.05, horizontal flip.
- Protocol: baseline x3 seeds, one change at a time x1 seed, every winner x3.
  A delta counts only if it exceeds 2x the parent's seed std.
- Code: `task4/train.py`, one file, reuses the notebook's dataset, model and
  training loop. Results in `task4/results.csv`; tables in `task4/results.md`.

## Fairness bug found during the audit

The harness first gave every MobileNet variant geometric augmentation by
default, so `M_cos`, `M_best`, `M_frozen` and `M_data*` changed two things
against `B2`. Fixed: augmentation now defaults to each family's baseline
(`geo` for the Task 3 net, `none` for MobileNet). The 18 affected rows were
deleted and rerun, seeds 0/1/2. All numbers below are post-fix.

## Results (mean ± std over 3 seeds)

| variant | change vs parent | valA | cameraA | testB | testH |
|---|---|---|---|---|---|
| B1 | Task 3 net, geo aug, 15 ep | 0.548 ± 0.084 | 0.490 ± 0.022 | 0.437 ± 0.091 | 0.458 ± 0.024 |
| B2 | MobileNetV3-Small, no aug, 15 ep | 0.726 ± 0.061 | 0.624 ± 0.037 | 0.598 ± 0.033 | 0.519 ± 0.051 |
| M_aug | + geometric aug | 0.702 ± 0.073 | 0.698 ± 0.053 | 0.667 ± 0.033 | 0.544 ± 0.057 |
| M_cos | + cosine, 30 ep | 0.691 ± 0.045 | 0.626 ± 0.038 | 0.586 ± 0.028 | 0.531 ± 0.016 |
| M_best | + best epoch on valA | 0.631 ± 0.034 | 0.614 ± 0.021 | 0.552 ± 0.049 | 0.528 ± 0.051 |
| M_frozen | + frozen backbone | 0.559 ± 0.017 | 0.581 ± 0.032 | 0.632 ± 0.043 | 0.450 ± 0.025 |
| M_dataB | + B photos | 0.548 ± 0.034 | 0.581 ± 0.065 | in-train | 0.608 ± 0.012 |
| M_dataH | + H photos | 0.809 ± 0.067 | 0.669 ± 0.036 | 0.678 ± 0.016 | in-train |
| M_dataBH | + B and H photos | 0.619 ± 0.017 | 0.669 ± 0.009 | in-train | in-train |
| M_aug_cos | + aug + cosine, 30 ep | **0.762 ± 0.017** | **0.767 ± 0.032** | **0.724 ± 0.000** | **0.686 ± 0.020** |
| M_combo | + aug + cosine + best epoch | 0.750 ± 0.029 | 0.733 ± 0.047 | 0.724 ± 0.056 | 0.625 ± 0.027 |
| L_cos | Task 3 net + cosine, 30 ep | 0.655 ± 0.034 | 0.541 ± 0.020 | 0.356 ± 0.016 | 0.447 ± 0.034 |
| L_color | Task 3 net + colour jitter | 0.607 ± 0.058 | 0.567 ± 0.067 | 0.448 ± 0.102 | 0.453 ± 0.045 |

Full table, single runs and parent deltas: `task4/results.md`.

## What the experiment says

- MobileNetV3-Small is the big win: +0.18 valA, +0.13 cameraA over the
  Task 3 net. No Task 3 variant came close.
- Augmentation is the second win, and it shows up mostly on camera-like
  photos: cameraA +0.14 over B2 with the same net and training time.
- Cosine + 30 epochs helps only together with augmentation. Alone it is
  within noise for MobileNet, and it hurt the Task 3 net.
- Best-epoch weights do not pay: `M_combo` ≈ `M_aug_cos` on valA and testB,
  worse on cameraA and testH, and it biases valA because it selects on valA.
- Frozen backbone, GAP, dropout, AdamW, label smoothing and colour-only
  jitter are all neutral to harmful. Extra photos from B or H did not beat
  augmentation either; H inflates valA but cannot be tested honestly
  (testH becomes training data).
- Winner: **M_aug_cos**, seed 0. For the phone, cameraA 0.729, testH 0.658
  for that seed; seed 1/2 are similar or better (cameraA 0.807/0.764).

## Leakage audit

Checked MD5 of every image in A, B, H and cameraA (1225 files).

- A and B: no duplicate inside a dataset, no overlap between them, no files
  shared with H. Train/val splits are clean.
- cameraA images are augmented copies of A-val photos. They are not
  independent, by construction; they only measure robustness.
- Historical set H is not clean:
  - 4 images are in both H train and H val: `maple_012`=`maple_027`,
    `maple_047`=`maple_004`, `maple_057`=`maple_029`,
    `maple_161`=`maple_028`.
  - 15 other duplicates inside H train, including two photos labelled as
    two classes: `apple_032`=`cherry_098` and `apple_103`=`cherry_031`.
- Impact on reported numbers: none. `testH` is scored only on models trained
  without H. Rows trained with H mark `testH` as `in-train` in the CSV.
- Selection bias, not leakage but related: `L_best`, `M_best` and `M_combo`
  choose weights on `valA`, so their valA is optimistic. The winner and all
  reported deltas use the fixed 15/30-epoch rule, not best-epoch selection.

## Caveats for the report

- valA has 28 images: one image = 0.036, seed noise is ±0.06–0.08. Small
  deltas are not real.
- testB chestnut has 4 images, so testB moves in steps of 0.25 per image.
- cameraA and valA share source photos, so cameraA is a robustness check,
  not an independent test.
- H duplicates mean H-trained numbers should not be compared on testH.

## Exported model

`python task4/train.py --export M_aug_cos --seed 0` retrained the winner and
wrote `leaf_model.onnx`, `model_info.json`, `check_image.jpg` and
`check_expected.json` in the repo root. PyTorch vs ONNX logit difference:
4.29e-06. Input `image` [1,3,96,96], output `logits` [1,4]; classes, size and
normalization match `index.html`, so the page needs no code change.
