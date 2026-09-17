# Methodology

This package compares fixed-class image classifiers under a controlled set of preprocessing and augmentation choices. It is a training and evaluation tool, not a production attendance service.

## Dataset contract

The caller supplies a local directory with one directory per class:

```text
dataset/
├── person_001/
│   ├── image_001.jpg
│   └── image_002.jpg
├── person_002/
└── person_003/
```

The loader discovers directory names and sorts them to assign stable integer labels. It does not contain an identity registry or a name mapping. Hidden directories and unsupported file extensions are ignored. At least two classes and three images per class are needed for a stratified three-way split.

The splitter shuffles records with seed `42`, then allocates each class across train, validation, and test using the `70:20:10` target. Small class counts are rounded while keeping all three partitions populated. A class with fewer than three images fails with a clear error instead of producing an unstratified result.

## Face preprocessing

OpenCV's default Haar cascade detects faces. When an image has several detections, the pipeline selects the largest bounding box. The selected box expands by the scenario margin, either `0.1` or `0.3`, and each bound is clamped to the source image. The crop is resized to `224x224` and returned as contiguous RGB `uint8` data. A missing face raises an error. Crops stay in memory unless a caller explicitly writes them.

The detector is injectable. This keeps preprocessing deterministic in unit-level integrations and lets callers provide another detector without changing crop rules.

## Augmentation

Every training image remains in the training array and produces exactly five additional in-memory augmented copies, for six training entries per source image. The original is unchanged. Each augmented copy samples rotation in `[-20, 20]` degrees, applies a horizontal flip, and adjusts brightness within the scenario limit. Brightness `0.1` samples a factor from `0.9` to `1.1`; brightness `0.4` samples from `0.6` to `1.4`. Grayscale conversion is a supported operation, but the baseline leaves it off to match the combined training augmenter behavior. The global seed controls the NumPy generator used for these choices. Validation and test images are not augmented.

## Factor matrix

The configuration contains exactly sixteen scenarios, the Cartesian product of these factors:

| Factor | Values |
| --- | --- |
| Crop margin | `0.1`, `0.3` |
| Pixel normalization | `0_1`, `minus1_1` |
| Brightness limit | `0.1`, `0.4` |
| Backbone | `mobilenetv2`, `efficientnetb0` |

The baseline uses a frozen ImageNet-backed MobileNetV2 or EfficientNetB0. Its head is `GlobalAveragePooling2D`, `Dropout(0.3)`, `Dense(128, relu)`, `Dropout(0.3)`, then a softmax classifier. It trains for at most 20 epochs with batch size 32 and learning rate `0.0001`. Early stopping monitors `val_accuracy`, uses patience `5`, and restores the best weights. `ReduceLROnPlateau` monitors `val_loss`, uses factor `0.5`, patience `3`, and minimum learning rate `1e-7`.

Optional fine-tuning unfreezes the last 30 layers of the selected backbone and recompiles it at learning rate `1e-5`. Fine-tuning is an explicit run option and is not mixed into baseline scenario ranking.

## Normalization semantics

The model applies one plain scaling layer to the `uint8` input before either backbone:

- `0_1`: divide pixel values by `255`.
- `minus1_1`: divide by `127.5`, then subtract `1`.

The package does not silently call a backbone-specific `preprocess_input` helper. This keeps normalization a visible scenario factor for both backbones. The EfficientNet builder disables its internal preprocessing layer when the installed TensorFlow API supports that option.

## Evaluation and artifacts

Evaluation reports test loss, accuracy, per-class precision, recall, F1-score, support, and confusion-matrix values. It also records the scenario configuration and seed. Results rank by highest test accuracy, then lowest test loss.

When an output root is supplied, each run uses a safe directory such as `scenario-05/` and may write:

```text
metrics.json
classification-report.json
confusion-matrix.csv
training-history.json
run-config.json
```

The path helper resolves and checks every output path so a scenario directory or file cannot escape the requested root. Model files are opt-in and belong in the ignored output directory.

## Privacy and limits

The repository contains no face images, trained models, class mappings, credentials, or private paths. Use a local dataset that you are allowed to process. Do not commit the dataset or generated artifacts.

Haar detection is sensitive to pose, lighting, occlusion, and image quality. A fixed-class classifier cannot recognize a class that was not present during training. Changing the class set requires a new training run. Metrics from one local dataset do not establish performance in a different environment. This package does not provide liveness detection, enrollment, access control, or a production database.

The related `attendance-system` project is the production-oriented application. It uses YuNet face detection and SFace embeddings with FastAPI, PostgreSQL, and pgvector, so new identities can be enrolled without retraining a fixed classifier. This package remains a separate training and evaluation companion.
