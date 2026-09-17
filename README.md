# face-recognition

`face-recognition` is a reusable face-recognition training and evaluation pipeline built around transfer learning. It compares MobileNetV2 and EfficientNetB0 across crop, pixel-scaling, and brightness settings. The package expects a labeled image dataset supplied by the caller. No face images, trained models, or identity mappings are included.

## What it does

- discovers any directory-per-class image dataset
- detects the largest face with OpenCV Haar detection
- crops with a configurable margin and clamps bounds to the image
- keeps each original training image and creates five in-memory augmented copies
- trains frozen ImageNet-backed MobileNetV2 or EfficientNetB0 classifiers
- evaluates one original plus five augmented test copies per held-out image
- reports accuracy, loss, precision, recall, F1-score, support, and confusion matrices
- ranks current runs by accuracy, then loss
- writes checked JSON and CSV artifacts below the requested output root

This is a fixed-class supervised classifier. It is separate from the production-oriented [`attendance-system`](https://github.com/wahyuabrory/attendance-system) project.

## Privacy boundary

The dataset is local input, not repository content. Do not commit biometric images, private class mappings, generated models, run artifacts, credentials, or environment files. The default ignored paths cover common local data and output directories. Use generic local examples such as `person_001` when documenting or testing a setup.

## Dataset layout

Provide at least two class directories and at least three supported images in each class:

```text
dataset/
├── person_001/
│   ├── image_001.jpg
│   └── image_002.jpg
├── person_002/
└── person_003/
```

The loader accepts `.bmp`, `.jpeg`, `.jpg`, `.png`, `.tif`, `.tiff`, and `.webp` files. Directory names become labels in sorted order. The package does not assume a known class list.

## Installation

Python 3.12 is required.

```bash
uv sync --locked
```

The official TensorFlow binaries require an AVX-capable CPU. See the
[TensorFlow installation requirements](https://www.tensorflow.org/install/pip#hardware-requirements).

The TensorFlow dependency is imported only when a model is built or training starts. Configuration loading and scenario listing work without importing TensorFlow.

## Scenarios

`configs/scenarios.yaml` contains exactly 16 scenarios. They are the full `2 x 2 x 2 x 2` product of:

| Factor | Values |
| --- | --- |
| Crop margin | `0.1`, `0.3` |
| Normalization | `0_1`, `minus1_1` |
| Brightness limit | `0.1`, `0.4` |
| Backbone | `mobilenetv2`, `efficientnetb0` |

Every scenario uses image size `224x224`, batch size `32`, a maximum of `20` epochs, learning rate `0.0001`, dropout `0.3`, dense layer size `128`, and seed `42`. The split target is `70:20:10`.

For each training image, the augmenter keeps one original and creates exactly five augmented copies. Each augmented copy samples rotation in `[-20, 20]` degrees, applies horizontal flip, and adds a pixel shift sampled from the scenario brightness limit. Grayscale conversion remains available as an augmentation operation, but it is disabled in the baseline configuration to match the combined training augmenter behavior.

Evaluation applies the same rule to held-out test images: one original plus five augmented copies. Validation images are not augmented.

The classifier head is `GlobalAveragePooling2D`, `Dropout(0.3)`, `Dense(128, relu)`, `Dropout(0.3)`, then the softmax output. Early stopping monitors `val_accuracy` with patience `5` and restores the best weights. `ReduceLROnPlateau` monitors `val_loss`, uses factor `0.5`, patience `3`, and minimum learning rate `1e-7`.

Normalization is plain pixel scaling for both backbones. `0_1` divides `uint8` values by `255`. `minus1_1` divides by `127.5` and subtracts `1`. The package does not silently apply a different backbone-specific preprocessing function.

List scenarios without loading TensorFlow:

```bash
uv run face-recognition list-scenarios
```

## Run training

Train one scenario with a local dataset:

```bash
uv run face-recognition train \
  --dataset ./dataset \
  --scenario 5 \
  --output ./artifacts
```

Run all sixteen scenarios:

```bash
uv run face-recognition run-all \
  --dataset ./dataset \
  --config configs/scenarios.yaml \
  --output ./artifacts
```

Fine-tuning is opt-in. It unfreezes the last 30 layers of the selected backbone at learning rate `1e-5`:

```bash
uv run face-recognition train \
  --dataset ./dataset \
  --scenario 5 \
  --fine-tune \
  --output ./artifacts
```

Use `--save-model` only when a local model file is needed. Model files and run outputs belong under an ignored output directory.

## Output files

Each evaluated scenario uses a directory like `scenario-05/` and writes:

```text
metrics.json
classification-report.json
confusion-matrix.csv
training-history.json
run-config.json
```

The output helper resolves every path and rejects traversal or symlink escapes outside the requested output root. JSON and CSV labels come from the class directories supplied for that run.

## Limitations

Haar detection can fail with poor lighting, pose, occlusion, or small faces. A fixed classifier only knows the classes used for training, so changing the class set requires retraining. Results depend on the supplied data and should not be treated as a general accuracy claim. This package does not implement liveness checks, enrollment, access control, a web API, or a production database.

## Relationship to attendance-system

The two projects have different jobs:

| This package | `attendance-system` |
| --- | --- |
| Fixed-class supervised classifier | Production-oriented recognition backend |
| MobileNetV2 or EfficientNetB0 | YuNet plus SFace embeddings |
| Class changes require training | New identities can be enrolled without classifier retraining |
| Local training and evaluation artifacts | FastAPI, PostgreSQL, and pgvector runtime |

See [`docs/methodology.md`](docs/methodology.md) for the split, crop, augmentation, model, evaluation, and privacy details.
