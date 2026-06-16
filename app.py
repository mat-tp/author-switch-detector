"""
    app.py — Flask entry point for the Author Switch Detector.

Routes
------
    /               Home — model status and quick links
    /login          Login page
    /register       New-account registration
    /logout         End session
    /inspect        Paste or upload text, view stats and sentences  [login required]
    /explore        Dataset statistics                              [login required]
    /train          Train a classifier on the dataset              [login required]
    /predict        Run author-switch detection                    [login required]
    /results        View saved evaluation metrics                  [login required]
    /admin/users    List all registered users                      [admin required]
    /settings       Configure dataset path and model params        [login required]
"""

import os
import sys
import traceback

# Allow "from src.xxx import ..." to work when app.py is the entry point
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from flask import (
    Flask, flash, g, redirect, render_template,
    request, session, url_for,
)

from src.auth.auth import (
    admin_required, get_all_users, get_current_user,
    login, login_required, logout, register,
)
from src.data.loader import dataset_stats, load_all, load_split
from src.features.extractor import FeatureExtractor, extract
from src.models.classifiers import MODEL_REGISTRY, ModelService, train_final_model
from src.models.evaluation import cross_validate_model, evaluate_model
from src.utils.save_load_model import load_json, load_pickle, save_json, save_pickle
from src.utils.text_helpers import (
    build_pairwise_dataset, sentence_details,
    split_sentences, text_from_request, text_stats,
)

# ── App setup ──────────────────────────────────────────────────────────────────

app = Flask(__name__)
# Change this to a real secret before any public deployment.
app.secret_key = os.environ.get("SECRET_KEY", "author-switch-dev-key")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

# ── Default config (overridable via environment or /settings) ──────────────────

DEFAULT_DATASET_ROOT = os.environ.get("DATASET_ROOT", "dataset/mawsa26-pan-zenodo-DATA")
RESULTS_DIR          = "results"
DIFFICULTIES         = ["easy", "medium", "hard"]

DEFAULT_HYPERPARAMS = {
    "n_splits":   3,
    "model_name": "logistic_regression",
}


# ── Request lifecycle ──────────────────────────────────────────────────────────

@app.before_request
def _load_user():
    g.user = get_current_user()


@app.context_processor
def _inject_user():
    return {"current_user": g.user}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _dataset_root() -> str:
    return session.get("dataset_root", DEFAULT_DATASET_ROOT)


def _hyperparams() -> dict:
    params = dict(DEFAULT_HYPERPARAMS)
    params.update(session.get("hyperparams", {}))
    return params


def model_is_trained() -> bool:
    return os.path.exists(os.path.join(RESULTS_DIR, "model.pkl"))


def _allowed_upload(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {"txt"}


# ── Auth routes ────────────────────────────────────────────────────────────────

@app.route("/login", methods=["GET", "POST"])
def auth_login():
    if session.get("username"):
        return redirect(url_for("index"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        ok, err  = login(username, password)
        if ok:
            flash(f"Welcome back, {username}!", "success")
            next_url = request.args.get("next") or url_for("index")
            return redirect(next_url)
        flash(err, "danger")
    return render_template("auth/login.html")


@app.route("/register", methods=["GET", "POST"])
def auth_register():
    if session.get("username"):
        return redirect(url_for("index"))
    if request.method == "POST":
        username  = request.form.get("username", "").strip()
        password  = request.form.get("password", "")
        password2 = request.form.get("password2", "")
        errors = []
        if password != password2:
            errors.append("Passwords do not match.")
        if not errors:
            ok, err = register(username, password, role="user")
            if ok:
                login(username, password)
                flash(f"Account created. Welcome, {username}!", "success")
                return redirect(url_for("index"))
            errors.append(err)
        for e in errors:
            flash(e, "danger")
    return render_template("auth/register.html")


@app.route("/logout")
def auth_logout():
    username = session.get("username", "")
    logout()
    flash(f"Goodbye, {username}." if username else "Logged out.", "success")
    return redirect(url_for("auth_login"))


# ── Admin ──────────────────────────────────────────────────────────────────────

@app.route("/admin/users")
@admin_required
def admin_users():
    return render_template("admin/users.html", users=get_all_users())


# ── Settings ───────────────────────────────────────────────────────────────────

@app.route("/settings", methods=["GET", "POST"])
@login_required
def settings():
    if request.method == "POST":
        new_root = request.form.get("dataset_root", "").strip()
        if new_root:
            session["dataset_root"] = new_root
            flash("Dataset path updated.", "success")
        hp = {}
        raw_splits = request.form.get("n_splits", "").strip()
        if raw_splits.isdigit() and int(raw_splits) >= 2:
            hp["n_splits"] = int(raw_splits)
        elif raw_splits:
            flash("n_splits must be an integer >= 2.", "warning")
        raw_model = request.form.get("model_name", "").strip()
        if raw_model in MODEL_REGISTRY:
            hp["model_name"] = raw_model
        elif raw_model:
            flash(f"Unknown model '{raw_model}'.", "warning")
        session["hyperparams"] = hp
        flash("Settings saved.", "success")
        return redirect(url_for("settings"))
    return render_template(
        "settings.html",
        dataset_root=_dataset_root(),
        hyperparams=_hyperparams(),
        model_names=list(MODEL_REGISTRY.keys()),
    )


# ── Home ───────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    trained = model_is_trained()
    metrics = None
    if trained:
        p = os.path.join(RESULTS_DIR, "eval_metrics.json")
        try:
            if os.path.exists(p):
                metrics = load_json(p)
        except Exception:
            pass
    return render_template("index.html", trained=trained, metrics=metrics)


# ── Inspect ────────────────────────────────────────────────────────────────────

@app.route("/inspect", methods=["GET", "POST"])
@login_required
def inspect():
    if request.method == "GET":
        return render_template("inspect.html")
    text, error = text_from_request()
    if error:
        flash(error, "warning")
        return render_template("inspect.html")
    sentences = split_sentences(text)
    if not sentences:
        flash("No sentences found — make sure your text has at least one non-empty line.", "warning")
        return render_template("inspect.html", input_text=text)
    stats              = text_stats(sentences)
    details, truncated = sentence_details(sentences)
    return render_template(
        "inspect.html",
        input_text=text,
        stats=stats,
        details=details,
        truncated=truncated,
    )


# ── Explore ────────────────────────────────────────────────────────────────────

@app.route("/explore")
@login_required
def explore():
    dataset_root  = _dataset_root()
    stats_by_diff = {}
    errors        = []
    for diff in DIFFICULTIES:
        path = os.path.join(dataset_root, diff, "train")
        try:
            problems = load_split(path, diff)
            if not problems:
                errors.append(
                    f"'{diff}': directory found but no valid problems loaded — "
                    "check that problem-N.txt and truth-problem-N.json files are present."
                )
                stats_by_diff[diff] = None
                continue
            stats_by_diff[diff]             = dataset_stats(problems)
            stats_by_diff[diff]["examples"] = _switch_examples(problems, n=2)
        except Exception as exc:
            errors.append(f"'{diff}': {exc}")
            stats_by_diff[diff] = None
    return render_template(
        "explore.html",
        stats=stats_by_diff,
        dataset_root=dataset_root,
        errors=errors,
    )


def _switch_examples(problems, n=2):
    switches, same = [], []
    for p in problems:
        if len(switches) >= n and len(same) >= n:
            break
        for i, change in enumerate(p["changes"]):
            pair = {
                "problem_id": p["problem_id"],
                "pair_index": i,
                "before":     p["sentences"][i][:220],
                "after":      p["sentences"][i + 1][:220],
            }
            if change == 1 and len(switches) < n:
                switches.append(pair)
            elif change == 0 and len(same) < n:
                same.append(pair)
    return {"switch": switches, "same_author": same}


# ── Train ──────────────────────────────────────────────────────────────────────

@app.route("/train", methods=["GET", "POST"])
@login_required
def train():
    model_names = list(MODEL_REGISTRY.keys())
    hp          = _hyperparams()
    if request.method == "POST":
        model_name = request.form.get("model_name", hp["model_name"])
        raw_splits = request.form.get("n_splits", str(hp["n_splits"])).strip()
        if not raw_splits.isdigit() or int(raw_splits) < 2:
            flash("n_splits must be an integer >= 2.", "warning")
            return render_template("train.html", models=model_names, log=None, done=False, hp=hp)
        if model_name not in MODEL_REGISTRY:
            flash(f"Unknown model '{model_name}'.", "warning")
            return render_template("train.html", models=model_names, log=None, done=False, hp=hp)
        try:
            log = _run_training(model_name, int(raw_splits))
            flash("Training complete!", "success")
            return render_template("train.html", models=model_names, log=log, done=True, hp=hp)
        except ValueError as exc:
            flash(str(exc), "danger")
            return render_template("train.html", models=model_names,
                                   log=[str(exc)], done=False, hp=hp)
        except Exception:
            tb = traceback.format_exc()
            flash("Training failed — see log for details.", "danger")
            return render_template("train.html", models=model_names,
                                   log=tb.splitlines(), done=False, hp=hp)
    return render_template("train.html", models=model_names, log=None, done=False, hp=hp)


def _run_training(model_name: str, n_splits: int) -> list:
    from sklearn.model_selection import StratifiedGroupKFold
    log          = []
    dataset_root = _dataset_root()
    log.append(f"Dataset root: {dataset_root}")
    log.append("Loading dataset...")
    data         = load_all(dataset_root, difficulties=tuple(DIFFICULTIES), splits=("train",))
    all_problems = []
    for key, problems in data.items():
        log.append(f"  {key}: {len(problems)} problems")
        all_problems.extend(problems)
    log.append(f"  Total: {len(all_problems)} problems loaded")
    if not all_problems:
        raise ValueError(
            f"No problems found at '{dataset_root}'. "
            "Update the path in Settings or set the DATASET_ROOT environment variable."
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
    cv_res = cross_validate_model(MODEL_REGISTRY[model_name](), X, y, groups, n_splits)
    log.append(f"  CV F1: {cv_res['test_f1_mean']:.4f} +/- {cv_res['test_f1_std']:.4f}")
    log.append("Training final model on all data...")
    splitter           = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(splitter.split(X, y, groups))
    eval_model         = MODEL_REGISTRY[model_name]()
    eval_model.fit(X[train_idx], y[train_idx])
    metrics            = evaluate_model(eval_model, X[val_idx], y[val_idx], model_name)
    final_model        = train_final_model(model_name, X, y)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_pickle(fe,          os.path.join(RESULTS_DIR, "feature_extractor.pkl"))
    save_pickle(final_model, os.path.join(RESULTS_DIR, "model.pkl"))
    save_json(cv_res,        os.path.join(RESULTS_DIR, "cv_results.json"))
    save_json(
        {k: float(v) for k, v in metrics.items() if isinstance(v, (float, np.floating))},
        os.path.join(RESULTS_DIR, "eval_metrics.json"),
    )
    log.append(f"Done -- held-out F1: {metrics['f1']:.4f}")
    log.append(f"Model and metrics saved to '{RESULTS_DIR}/'")
    return log


# ── Predict ────────────────────────────────────────────────────────────────────

@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    trained    = model_is_trained()
    result     = None
    input_text = ""
    if request.method == "POST":
        uploaded = request.files.get("file")
        if uploaded and uploaded.filename:
            if not _allowed_upload(uploaded.filename):
                ext = uploaded.filename.rsplit(".", 1)[-1].lower() if "." in uploaded.filename else "(none)"
                flash(
                    f"Unsupported file type '.{ext}'. Only plain .txt files are accepted.",
                    "warning",
                )
                return render_template("predict.html", result=None, input_text="", trained=trained)
            try:
                raw = uploaded.read()
                try:
                    input_text = raw.decode("utf-8")
                except UnicodeDecodeError:
                    input_text = raw.decode("latin-1")
                    flash("File decoded with latin-1 fallback — some characters may look odd.", "warning")
            except Exception as exc:
                flash(f"Could not read the uploaded file: {exc}", "danger")
                return render_template("predict.html", result=None, input_text="", trained=trained)
        else:
            text, error = text_from_request()
            if error:
                flash(error, "warning")
                return render_template("predict.html", result=None, input_text="", trained=trained)
            input_text = text

        sentences = split_sentences(input_text)
        if len(sentences) < 2:
            flash("Please provide at least two sentences (one per line).", "warning")
            return render_template("predict.html", result=None, input_text=input_text, trained=trained)

        raw_threshold = request.form.get("threshold", "0.5").strip()
        try:
            threshold = float(raw_threshold)
            if not (0.0 <= threshold <= 1.0):
                raise ValueError
        except ValueError:
            flash("Threshold must be a number between 0.0 and 1.0.", "warning")
            threshold = 0.5

        try:
            result = (_predict_with_model(sentences, threshold)
                      if trained else _predict_random(sentences, threshold))
        except Exception as exc:
            flash(f"Prediction failed: {exc}", "danger")
            app.logger.exception("Prediction error")

    return render_template("predict.html", result=result, input_text=input_text, trained=trained)


def _predict_with_model(sentences, threshold):
    svc = ModelService(results_dir=RESULTS_DIR)
    if not svc.load():
        raise RuntimeError("Trained model artefacts not found in results/.")
    pairs = svc.predict(sentences, threshold=threshold)
    return _build_result(pairs, sentences, mode="model")


def _predict_random(sentences, threshold):
    import random
    rng   = random.Random()
    pairs = []
    for i in range(len(sentences) - 1):
        prob = round(rng.random(), 3)
        pairs.append({
            "index":       i,
            "sentence_a":  sentences[i],
            "sentence_b":  sentences[i + 1],
            "switch":      prob >= threshold,
            "probability": prob,
        })
    return _build_result(pairs, sentences, mode="random")


def _build_result(pairs, sentences, mode):
    return {
        "pairs":       pairs,
        "n_switches":  sum(1 for p in pairs if p["switch"]),
        "n_pairs":     len(pairs),
        "n_sentences": len(sentences),
        "mode":        mode,
    }


# ── Results ────────────────────────────────────────────────────────────────────

@app.route("/results")
@login_required
def results():
    if not model_is_trained():
        flash("No trained model found. Train a model first.", "warning")
        return redirect(url_for("train"))
    metrics, cv = None, None
    try:
        p = os.path.join(RESULTS_DIR, "eval_metrics.json")
        if os.path.exists(p):
            metrics = load_json(p)
    except Exception:
        flash("Could not load evaluation metrics — the file may be corrupted.", "warning")
    try:
        p = os.path.join(RESULTS_DIR, "cv_results.json")
        if os.path.exists(p):
            cv = load_json(p)
    except Exception:
        flash("Could not load cross-validation results.", "warning")
    return render_template("results.html", metrics=metrics, cv=cv)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
