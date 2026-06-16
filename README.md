# Author Switch Detector

**Author:** Mr. TP Matlala

A web application for detecting author changes between sentences in multi-author texts,
built for the PAN 2026 MAWSA (Multi-Author Writing Style Analysis) task.

## Tasks

- **[Main-Task]** Detecting Author Switch positions
- **[Sub-Task]** Is the document written by multiple authors?
  — By combining extracted features and a clustering algorithm.
- **[Sub-Task]** Detecting segments for each author

---

## End-to-End Pipeline

```
Raw Text → DocumentLoader → FeatureExtractor → Pairwise Dataset → Classifier → Predictions
```

---

## Class Reference

### DocumentLoader  (`data/loader.py`)

Formerly `DataProcessing`. Handles all file I/O.

| Kind      | Name                         | Description                              |
|-----------|------------------------------|------------------------------------------|
| attribute | `fileCount : int`            | Number of files loaded this session      |
| attribute | `sentenceCount : int`        | Total sentences across loaded documents  |
| attribute | `docLevel : DocLevel`        | Difficulty level (EASY / MEDIUM / HARD)  |
| method    | `loadFile(path)`             | Load a `.txt` or `.pdf` file → `Document`|
| method    | `validateFile(path)`         | Check existence and extension            |
| method    | `convertPdfToText(pdfPath)`  | Extract plain text from PDF              |
| method    | `extractText(raw)`           | Normalise whitespace in raw text         |
| method    | `getStats()`                 | Summary dict for all loaded documents    |

### Document  (`data/loader.py`)

| Field         | Type                  |
|---------------|-----------------------|
| `id`          | `str`                 |
| `rawText`     | `str`                 |
| `cleanedText` | `str`                 |
| `sentences`   | `List[str]`           |
| `metadata`    | `Map[str, object]`    |
| `fileType`    | `FileType` {PDF, TXT} |

### FeatureExtractor  (`features/extractor.py`)

| Kind      | Name                                        | Description                              |
|-----------|---------------------------------------------|------------------------------------------|
| attribute | `featureNames : List[str]`                  | Name of every feature dimension          |
| attribute | `numFeatures : int`                         | Total feature count                      |
| method    | `collectFeatures(document) → FeatureVector` | **Primary API** — extract all features   |
| method    | `transform(sentence, position, n_total)`    | Single-sentence 1-D array                |
| method    | `transform_document(sentences)`             | 2-D array [n_sentences × numFeatures]    |

### FeatureVector  (`features/extractor.py`)

| Field    | Type            |
|----------|-----------------|
| `values` | `float[]`       |
| `names`  | `List[str]`     |

### ModelService  (`models/classifiers.py`)

Responsible for loading a trained model from disk and running predictions.

| Kind   | Name                            | Description                              |
|--------|---------------------------------|------------------------------------------|
| method | `load() → bool`                 | Load model + extractor from `results/`   |
| method | `is_ready() → bool`             | True iff a model has been loaded         |
| method | `predict(sentences, threshold)` | Return per-pair switch predictions       |
| method | `get_metrics() → dict`          | Return saved evaluation metrics          |

### LinearRegression  (`models/linear_regression.py`)

Custom closed-form implementation of **y = mX + c** (solved via the normal equation).

| Kind   | Name                               | Description                   |
|--------|------------------------------------|-------------------------------|
| method | `fit(X, y)`                        | Learn weights m and bias c    |
| method | `predict(X) → float[]`             | Continuous predictions         |
| method | `predict_binary(X, threshold=0.5)` | 0 / 1 labels                  |
| method | `score(X, y) → float`              | R² coefficient of determination|
| method | `get_params() → dict`              | Return `{'m': ..., 'c': ...}` |

---

## File Structure

```
hyProject/
├── app.py                      # Flask entry point
├── data/
│   └── loader.py               # DocumentLoader, Document, DocLevel, FileType
├── features/
│   └── extractor.py            # FeatureExtractor, FeatureVector, FEATURE_NAMES
├── models/
│   ├── classifiers.py          # sklearn pipelines, ModelService, MODEL_REGISTRY
│   ├── evaluation.py           # cross-validation, held-out eval, feature importance
│   └── linear_regression.py   # Custom LinearRegression (y = mX + c)
├── utils/
│   ├── save_load_model.py      # pickle / JSON helpers
│   └── text_helpers.py         # pairwise dataset builder, text stats
├── templates/                  # Jinja2 HTML templates
├── static/                     # CSS styling
├── docs/                       # Deliverable documents
│   └── D05/Notes.txt
├── requirements.txt
└── Makefile
```

**Dataset** (`dataset/mawsa26-pan-zenodo-DATA/`) — three difficulty splits (easy / medium / hard),
each containing `problem-N.txt` and `truth-problem-N.json` files.

---

## Quick Start

```bash
pip install -r requirements.txt
export DATASET_ROOT=path/to/mawsa26-pan-zenodo-DATA
python app.py
```

Then open `http://localhost:5000`.

---

## References

1. Dataset: PAN 2026 MAWSA — Multi-Author Writing Style Analysis
