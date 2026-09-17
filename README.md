# face-recognition

This project trains and compares face classifiers from a local image dataset. Put each person's images in a separate folder, choose an experiment scenario, and the command-line tool handles face detection, cropping, augmentation, training, evaluation, and result files.

The goal is comparison, not deployment. The experiments compare MobileNetV2 and EfficientNetB0 while changing the face crop, pixel scaling, and brightness augmentation. A trained model can identify only the people present in its training data. Adding or changing people requires another training run.

The repository contains no face images, trained models, or identity records.

## Set up the project

You need Python 3.12, [`uv`](https://docs.astral.sh/uv/), and a CPU that supports AVX instructions. [TensorFlow's published binaries use AVX](https://www.tensorflow.org/install/pip#hardware-requirements), so training cannot run without it.

Install the locked dependencies:

```bash
uv sync --locked
```

You can inspect configuration without loading TensorFlow:

```bash
uv run face-recognition list-scenarios
```

## Prepare the dataset

Create one folder for each person. Each folder needs at least three supported images, and the dataset needs at least two people.

```text
dataset/
├── person_001/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── image_003.jpg
└── person_002/
    ├── image_001.jpg
    ├── image_002.jpg
    └── image_003.jpg
```

Folder names become class labels in sorted order. The loader accepts `.bmp`, `.jpeg`, `.jpg`, `.png`, `.tif`, `.tiff`, and `.webp` files.

Keep the dataset local. The repository ignores `dataset/`, `data/`, and `private/`, but you should still check staged files before every commit. Do not commit biometric images, private identity mappings, credentials, models, or experiment output.

## Run an experiment

The scenario configuration is in [`configs/scenarios.yaml`](configs/scenarios.yaml). It defines 16 combinations of crop margin, pixel normalization, brightness augmentation, and model backbone.

Train and evaluate one scenario:

```bash
uv run face-recognition train \
  --dataset ./dataset \
  --scenario 5 \
  --output ./artifacts
```

Run every scenario:

```bash
uv run face-recognition run-all \
  --dataset ./dataset \
  --config configs/scenarios.yaml \
  --output ./artifacts
```

Training uses frozen ImageNet weights by default. Add `--fine-tune` to unfreeze the final backbone layers after the baseline training stage:

```bash
uv run face-recognition train \
  --dataset ./dataset \
  --scenario 5 \
  --fine-tune \
  --output ./artifacts
```

Add `--save-model` only when you need a local `.keras` model file. Models and run output belong in an ignored directory such as `artifacts/`.

## Read the results

Each run writes its files under a scenario folder such as `artifacts/scenario-05/`:

```text
metrics.json
classification-report.json
confusion-matrix.csv
training-history.json
run-config.json
```

The metrics include loss, accuracy, precision, recall, F1 score, support, and a confusion matrix. A full run ranks scenarios by highest test accuracy, then lowest test loss.

Read [`docs/methodology.md`](docs/methodology.md) for the data split, face crop rules, augmentation, model structure, normalization, evaluation, and privacy boundaries.

## Know the limits

OpenCV Haar detection can miss faces when lighting, pose, occlusion, or image size is poor. Training results depend on the supplied dataset and do not represent general face-recognition accuracy.

This project does not provide liveness checks, identity enrollment, access control, a web API, or a database. For an attendance backend based on face embeddings, see [`attendance-system`](https://github.com/wahyuabrory/attendance-system).

## Find your way around

- [`src/face_recognition`](src/face_recognition) contains the pipeline and command-line interface.
- [`configs/scenarios.yaml`](configs/scenarios.yaml) is the experiment configuration.
- [`docs/methodology.md`](docs/methodology.md) describes the experiment rules in detail.
- [`tests`](tests) covers configuration, data splitting, scenarios, and face preprocessing.
