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
Upload file → Check format → Convert PDF (if needed) → DocumentLoader
   → FeatureExtractor → Pairwise Dataset → Classifier → Predictions → Results
```

---

## Class Reference

### DocumentLoader  (`src/data/document_loader.py`)

Handles all file I/O: loading, validating, and converting documents.

| Kind      | Name                         | Description                              |
|-----------|------------------------------|-------------------------------------------|
| attribute | `fileCount : int`            | Number of files loaded this session      |
| attribute | `sentenceCount : int`        | Total sentences across loaded documents  |
| attribute | `docLevel : DocLevel`        | Difficulty level (EASY / MEDIUM / HARD)  |
| method    | `loadFile(path)`             | Load a `.txt` or `.pdf` file → `Document`|
| method    | `validateFile(path)`         | Check existence and extension            |
| method    | `convertPdfToText(pdfPath)`  | Extract plain text from PDF              |
| method    | `extractText(raw)`           | Normalise whitespace in raw text         |
| method    | `getStats()`                 | Summary dict for all loaded documents    |

### Document  (`src/data/document_loader.py`)

| Field         | Type                  |
|---------------|-----------------------|
| `id`          | `str`                 |
| `rawText`     | `str`                 |
| `cleanedText` | `str`                 |
| `sentences`   | `List[str]`           |
| `metadata`    | `Map[str, object]`    |
| `fileType`    | `FileType` {PDF, TXT} |

### FeatureExtractor  (`src/features/extractor.py`)

No external NLP libraries — every feature is computed from raw text using stdlib + numpy.

| Kind      | Name                                        | Description                              |
|-----------|---------------------------------------------|---------------------------------------------|
| attribute | `featureNames : List[str]`                  | Name of every feature dimension          |
| attribute | `numFeatures : int`                         | Total feature count                      |
| method    | `collectFeatures(document) → FeatureVector` | **Primary API** — extract all features   |
| method    | `transform(sentence, position, n_total)`    | Single-sentence 1-D array                |
| method    | `transform_document(sentences)`             | 2-D array [n_sentences × numFeatures]    |

Feature groups: char-level (counts, punctuation, digit/uppercase ratios), word-level
(function words, length bins, readability), lexical richness (TTR, hapax ratio,
Simpson's D), and sentence position. Excluded by design (too compute-heavy for this
project): word n-grams, POS tags, TF-IDF, word embeddings, entropy, NMF.

### FeatureVector  (`src/features/feature_vector.py`)

| Field    | Type            |
|----------|-----------------|
| `values` | `float[]`       |
| `names`  | `List[str]`     |

### ModelService  (`src/models/classifiers.py`)

Responsible for loading a trained model from disk and running predictions.

| Kind   | Name                            | Description                              |
|--------|---------------------------------|----------------------------------------------|
| method | `load() → bool`                 | Load model + extractor from `results/`   |
| method | `is_ready() → bool`             | True iff a model has been loaded         |
| method | `predict(sentences, threshold)` | Return per-pair switch predictions       |
| method | `get_metrics() → dict`          | Return saved evaluation metrics          |

### LinearRegression  (`src/models/linear_regression.py`)

Custom closed-form implementation of **y = mX + c** (solved via the normal equation).
No external ML library involved — pure numpy linear algebra.

| Kind   | Name                               | Description                   |
|--------|-------------------------------------|----------------------------------|
| method | `fit(X, y)`                        | Learn weights m and bias c    |
| method | `predict(X) → float[]`             | Continuous predictions         |
| method | `predict_binary(X, threshold=0.5)` | 0 / 1 labels                  |
| method | `score(X, y) → float`              | R² coefficient of determination|
| method | `get_params() → dict`              | Return `{'m': ..., 'c': ...}` |

### Evaluation  (`src/models/evaluation.py`)

All metrics, cross-validation, and feature-importance code is implemented from
scratch — no sklearn calls anywhere in this file.

| Kind     | Name                                   | Description                                |
|----------|------------------------------------------|---------------------------------------------|
| function | `precision_score`, `recall_score`, `f1_score`, `accuracy_score` | Binary classification metrics from scratch |
| function | `confusion_matrix`, `classification_report` | Plain-text reporting, sklearn-style output |
| function | `stratified_group_kfold(y, groups, n_splits)` | Custom greedy stratified+grouped k-fold splitter |
| function | `cross_validate_model(model_factory, X, y, groups, n_splits)` | Runs CV using the custom splitter above |
| function | `evaluate_model(model, X_test, y_test)` | Held-out evaluation report |
| function | `permutation_feature_importance(...)`  | Feature ranking by shuffling each column |

### User Authentication  (`src/auth/auth.py`)

Users are stored in `dataset/users/users.json`, with SHA-256 hashed passwords (never
stored in plaintext).

| Kind     | Name                                  | Description                              |
|----------|----------------------------------------|---------------------------------------------|
| function | `register(username, password, role)`  | Create a new account                     |
| function | `login(username, password)`           | Verify credentials, start a session      |
| function | `logout()`                            | Clear the session                        |
| function | `get_current_user()`                  | Return the logged-in user's record       |
| function | `get_all_users()`                     | List all accounts (admin use)            |
| decorator| `@login_required`                     | Redirect to `/login` if not authenticated|
| decorator| `@admin_required`                     | Redirect if user is not an admin         |

---

## File Structure

```
hyProject/
├── app.py                          # Flask entry point
├── src/
│   ├── auth/
│   │   └── auth.py                 # register/login/logout, @login_required, @admin_required
│   ├── data/
│   │   └── document_loader.py      # DocumentLoader, Document, DocLevel, FileType
│   ├── features/
│   │   ├── extractor.py            # FeatureExtractor, FEATURE_NAMES, extract()
│   │   └── feature_vector.py       # FeatureVector
│   ├── models/
│   │   ├── classifiers.py          # sklearn pipelines, ModelService, MODEL_REGISTRY
│   │   ├── evaluation.py           # from-scratch CV, metrics, feature importance
│   │   └── linear_regression.py    # Custom LinearRegression (y = mX + c)
│   └── utils/
│       ├── save_load_model.py      # pickle / JSON helpers
│       └── text_helpers.py         # pairwise dataset builder, text stats
├── templates/
│   ├── auth/                       # login.html, register.html
│   ├── admin/                      # users.html
│   └── *.html                      # index, inspect, explore, train, predict, results, settings
├── static/                         # CSS styling
├── docs/                           # Deliverable documents
│   └── D05/Notes.txt
├── dataset/
│   ├── mawsa26-pan-zenodo-DATA/    # PAN 2026 MAWSA dataset (easy/medium/hard splits)
│   └── users/users.json            # registered user accounts
├── requirements.txt
└── Makefile
```

**Dataset** (`dataset/mawsa26-pan-zenodo-DATA/`) — three difficulty splits (easy / medium / hard),
each containing `problem-N.txt` and `truth-problem-N.json` files.

---

## Quick Start

```bash
pip install -r requirements.txt
export DATASET_ROOT=dataset/mawsa26-pan-zenodo-DATA
python app.py
```

Then open `http://localhost:5000`. Register an account, log in, and the dataset path
and training hyperparameters can also be changed from the **Settings** page without
needing to restart the app.

---

## Upload → Predict flow

1. User uploads a `.txt` or `.pdf` file (or pastes text directly) on `/predict`.
2. The file extension is validated; unsupported types are rejected with a clear message.
3. If the file is a PDF, `DocumentLoader.convertPdfToText` extracts the plain text first.
4. The resulting sentences are run through the trained classifier (or a random baseline
   if no model has been trained yet).
5. Results — one prediction per consecutive sentence pair — are shown on the same page.

---

## Pending / Deferred Work

- **UMAP visualisation pipeline** (`docs/D05/Notes.txt`): Token Cache → Feature Store →
  Pairwise Store → UMAP Visualization → Classifier → Feature Importance Analysis.
  Deliberately deferred — significant additional scope, to be tackled separately.

---

## References

1. Dataset: PAN 2026 MAWSA — Multi-Author Writing Style Analysis
