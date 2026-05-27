"""
    Flask web application for the Author Switch Detector.

Routes:
    /           Home — model status and quick links
    /inspect    Paste or upload text, view stats and sentences
    /explore    Dataset statistics (requires dataset on disk)
    /train      Train a classifier on the dataset
    /predict    Run author-switch detection on pasted or uploaded text
    /results    View saved evaluation metrics

"""

# TODO: Add user authentication and model management for multiple users and models.
# TODO: Add better error handling and user feedback for file uploads and model training.

import os
import traceback

import numpy as np
from flask import Flask, flash, redirect, render_template, request, url_for

from data.loader import dataset_stats, load_all, load_split
from features.extractor import FeatureExtractor, extract
from models.classifiers import MODEL_REGISTRY, train_final_model
from models.evaluation import cross_validate_model, evaluate_model
from utils.save_load_model import load_json, load_pickle, save_json, save_pickle
from utils.text_helpers import (
    build_pairwise_dataset, sentence_details, split_sentences, text_from_request, text_stats,)


app = Flask(__name__)
# I will change this before deploying anywhere public.
app.secret_key = "author-switch-dev-key"
# TODO : Add better error handling for file uploads
# TODO : Add better configurable options for dataset path and model hyperparameters.

app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2 MB upload limit

DATASET_ROOT = os.environ.get("DATASET_ROOT", "mawsa26-pan-zenodo-DATA")
RESULTS_DIR = "results"
DIFFICULTIES = ["easy", "medium", "hard"]


# Checking if a model exist in the path
def model_is_trained():
    return os.path.exists(os.path.join(RESULTS_DIR, "model.pkl"))


# Home Page

@app.route("/")
def index():
    trained = model_is_trained()
    metrics = None
    if trained and os.path.exists(os.path.join(RESULTS_DIR, "eval_metrics.json")):
        metrics = load_json(os.path.join(RESULTS_DIR, "eval_metrics.json"))
    return render_template("index.html", trained=trained, metrics=metrics)


# Data Inspection Page
# This is for a Users Own text and for exploring a single file from the dataset.

@app.route("/inspect", methods=["GET", "POST"])
def inspect():
    if request.method == "GET":
        return render_template("inspect.html")

    text, error = text_from_request()
    if error:
        flash(error, "warning")
        return render_template("inspect.html")

    sentences = split_sentences(text)
    if len(sentences) < 1:
        flash("No sentences found in the text.", "warning")
        return render_template("inspect.html", input_text=text)

    stats = text_stats(sentences)
    details, truncated = sentence_details(sentences)

    return render_template(
        "inspect.html",
        input_text=text,
        stats=stats,
        details=details,
        truncated=truncated,
    )


# Explore dataset : Easy, Medium, Hard splits with stats and examples of author switches.

@app.route("/explore")
def explore():
    stats_by_diff = {}
    errors = []

    for diff in DIFFICULTIES:
        path = os.path.join(DATASET_ROOT, diff, "train")
        try:
            problems = load_split(path, diff)
            stats_by_diff[diff] = dataset_stats(problems)
            stats_by_diff[diff]["examples"] = _switch_examples(problems, n=2)
        except Exception as e:
            errors.append(f"{diff}: {e}")
            stats_by_diff[diff] = None

    return render_template(
        "explore.html",
        stats=stats_by_diff,
        dataset_root=DATASET_ROOT,
        errors=errors,
    )


def _switch_examples(problems, n=2):
    """
    Collect up to n examples of each pair type from the problem list.
    """
    switches = []
    same = []

    for p in problems:
        if len(switches) >= n and len(same) >= n:
            break
        for i, change in enumerate(p["changes"]):
            pair = {
                "problem_id": p["problem_id"],
                "pair_index": i,
                "before": p["sentences"][i][:220],
                "after": p["sentences"][i + 1][:220],
            }
            if change == 1 and len(switches) < n:
                switches.append(pair)
            elif change == 0 and len(same) < n:
                same.append(pair)

    return {"switch": switches, "same_author": same}


# Training a model.


@app.route("/train", methods=["GET", "POST"])
def train():
    models = list(MODEL_REGISTRY.keys())

    if request.method == "POST":
        model_name = request.form.get("model_name", "logistic_regression")
        n_splits = int(request.form.get("n_splits", 3))

        try:
            log = _run_training(model_name, n_splits)
            flash("Training complete!", "success")
            return render_template("train.html", models=models, log=log, done=True)
        except Exception:
            flash("Training failed — see log for details.", "danger")
            return render_template(
                "train.html", models=models, log=traceback.format_exc(), done=False
            )

    return render_template("train.html", models=models, log=None, done=False)


def _run_training(model_name, n_splits):
    from sklearn.model_selection import StratifiedGroupKFold

    log = []

    log.append("Loading dataset...")
    data = load_all(DATASET_ROOT, difficulties=tuple(
        DIFFICULTIES), splits=("train",))
    all_problems = []
    for key, problems in data.items():
        log.append(f"  {key}: {len(problems)} problems")
        all_problems.extend(problems)
    log.append(f"  Total: {len(all_problems)} problems loaded")

    if not all_problems:
        raise ValueError(
            f"No problems found at '{DATASET_ROOT}'. "
            "Check that the dataset folder exists and follows the expected layout."
        )

    log.append("Extracting features...")
    fe = FeatureExtractor()
    fe.fit([s for p in all_problems for s in p["sentences"]])
    log.append(f"  Feature vector size: {fe.n_features} features per sentence")

    log.append("Building pairwise training set...")
    X, y, meta = build_pairwise_dataset(all_problems, fe)
    log.append(f"  {X.shape[0]} pairs  |  switch rate: {y.mean() * 100:.1f}%")

    groups = np.array([int(m["problem_id"]) for m in meta])

    log.append(f"Cross-validating [{model_name}] with {n_splits} folds...")
    cv = cross_validate_model(
        MODEL_REGISTRY[model_name](), X, y, groups, n_splits)
    log.append(
        f"  CV F1: {cv['test_f1_mean']:.4f} ± {cv['test_f1_std']:.4f}"
    )

    log.append("Training final model on all data...")
    # Hold out one fold for held-out evaluation metrics
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(splitter.split(X, y, groups))
    eval_model = MODEL_REGISTRY[model_name]()
    eval_model.fit(X[train_idx], y[train_idx])
    metrics = evaluate_model(eval_model, X[val_idx], y[val_idx], model_name)

    final_model = train_final_model(model_name, X, y)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_pickle(fe, os.path.join(RESULTS_DIR, "feature_extractor.pkl"))
    save_pickle(final_model, os.path.join(RESULTS_DIR, "model.pkl"))
    save_json(cv, os.path.join(RESULTS_DIR, "cv_results.json"))
    save_json(
        {k: float(v) for k, v in metrics.items()
         if isinstance(v, (float, np.floating))},
        os.path.join(RESULTS_DIR, "eval_metrics.json"),
    )

    log.append(f"Done — held-out F1: {metrics['f1']:.4f}")
    log.append(f"Model and metrics saved to '{RESULTS_DIR}/'")
    return log


# Making Predictions

@app.route("/predict", methods=["GET", "POST"])
def predict():
    trained = model_is_trained()
    result = None
    input_text = ""

    if request.method == "POST":
        text, error = text_from_request()
        if error:
            flash(error, "warning")
        else:
            input_text = text
            sentences = split_sentences(text)
            if len(sentences) < 2:
                flash("Please provide at least two sentences.", "warning")
            else:
                threshold = float(request.form.get("threshold", 0.5))
                try:
                    if trained:
                        result = _predict_with_model(sentences, threshold)
                    else:
                        result = _predict_random(sentences, threshold)
                except Exception:
                    flash("Prediction failed: " +
                          traceback.format_exc(), "danger")

    return render_template("predict.html", result=result, input_text=input_text,
                           trained=trained)


def _predict_with_model(sentences, threshold):
    """Run the trained classifier on consecutive sentence pairs."""
    fe = load_pickle(os.path.join(RESULTS_DIR, "feature_extractor.pkl"))
    model = load_pickle(os.path.join(RESULTS_DIR, "model.pkl"))

    vecs = fe.transform_document(sentences)
    pairs = []

    for i in range(len(sentences) - 1):
        diff = np.abs(vecs[i] - vecs[i + 1]).reshape(1, -1)
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(diff)[0, 1])
            label = int(prob >= threshold)
        else:
            prob = None
            label = int(model.predict(diff)[0])

        pairs.append({
            "index": i,
            "sentence_a": sentences[i],
            "sentence_b": sentences[i + 1],
            "switch": label == 1,
            "probability": round(prob, 3) if prob is not None else None,
        })

    return _build_result(pairs, sentences, mode="model")


def _predict_random(sentences, threshold):
    """
    Fallback when no model is trained.
    Assigns a random probability to each pair and labels it using the threshold.
    Useful for exploring the UI or establishing a random baseline to beat.
    """
    import random
    rng = random.Random()   # unseeded — different result every run
    pairs = []

    for i in range(len(sentences) - 1):
        prob = round(rng.random(), 3)
        pairs.append({
            "index": i,
            "sentence_a": sentences[i],
            "sentence_b": sentences[i + 1],
            "switch": prob >= threshold,
            "probability": prob,
        })

    return _build_result(pairs, sentences, mode="random")


def _build_result(pairs, sentences, mode):
    """Shared result dict for both prediction modes."""
    n_switches = sum(1 for p in pairs if p["switch"])
    return {
        "pairs": pairs,
        "n_switches": n_switches,
        "n_pairs": len(pairs),
        "n_sentences": len(sentences),
        "mode": mode,
    }


# View Results

@app.route("/results")
def results():
    if not model_is_trained():
        flash("No trained model found. Please train a model first.", "warning")
        return redirect(url_for("train"))

    metrics = None
    cv = None
    metrics_path = os.path.join(RESULTS_DIR, "eval_metrics.json")
    cv_path = os.path.join(RESULTS_DIR, "cv_results.json")

    if os.path.exists(metrics_path):
        metrics = load_json(metrics_path)
    if os.path.exists(cv_path):
        cv = load_json(cv_path)

    return render_template("results.html", metrics=metrics, cv=cv)


# Application Entry point
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
