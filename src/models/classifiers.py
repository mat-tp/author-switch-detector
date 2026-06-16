"""
    sl_classifiers.py — scikit-learn classifiers wrapped in a StandardScaler Pipeline.

    Each factory function returns a fresh, unfitted sklearn Pipeline.
    MODEL_REGISTRY maps string names to factory functions for use by the
    training and evaluation modules.

    ModelService
    ------------
    Responsible for loading a trained model from disk and running predictions.
    Acts as the bridge between serialised artefacts (produced by training)
    and the web app / evaluation pipeline.
"""

import os

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.utils.save_load_model import load_pickle, load_json


# ── Classifier factories ──────────────────────────────────────────────────────

def make_logistic_regression() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        )),
    ])


def make_svm() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf",
            class_weight="balanced",
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=42,
        )),
    ])


def make_knn() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=5, metric="euclidean")),
    ])


def make_mlp() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-3,
            learning_rate="adaptive",
            learning_rate_init=1e-3,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            batch_size=64,
            random_state=42,
            verbose=False,
        )),
    ])


MODEL_REGISTRY: dict = {
    "logistic_regression": make_logistic_regression,
    "svm":                 make_svm,
    "knn":                 make_knn,
    "mlp":                 make_mlp,
}


def train_final_model(model_name: str, X_train, y_train):
    """Instantiate, fit, and return the named model on the full training set."""
    if model_name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {list(MODEL_REGISTRY)}"
        )
    model = MODEL_REGISTRY[model_name]()
    model.fit(X_train, y_train)
    print(f"[classifiers] Trained '{model_name}' on {len(y_train)} samples.")
    return model


# ── ModelService ──────────────────────────────────────────────────────────────

class ModelService:
    """
    Responsible for:
        - loading a trained model (and its feature extractor) from disk
        - running author-switch predictions on new text
        - exposing the most recent evaluation metrics

    Usage
    -----
        svc = ModelService(results_dir="results")
        if svc.is_ready():
            pairs = svc.predict(sentences, threshold=0.5)
    """

    def __init__(self, results_dir: str = "results"):
        self.results_dir    = results_dir
        self._model         = None
        self._extractor     = None
        self._metrics: dict = {}
        self._model_name: str = ""

    # ── loading ───────────────────────────────────────────────────

    def load(self) -> bool:
        """
        Load model and feature extractor from *results_dir*.
        Returns True on success, False if artefacts are missing.
        """
        model_path     = os.path.join(self.results_dir, "model.pkl")
        extractor_path = os.path.join(self.results_dir, "feature_extractor.pkl")

        if not os.path.exists(model_path):
            print(f"[ModelService] No model found at {model_path}")
            return False

        self._model     = load_pickle(model_path)
        self._extractor = load_pickle(extractor_path) if os.path.exists(extractor_path) else None

        # load metrics if present
        metrics_path = os.path.join(self.results_dir, "eval_metrics.json")
        if os.path.exists(metrics_path):
            self._metrics = load_json(metrics_path)

        print(f"[ModelService] Model loaded from {model_path}")
        return True

    def is_ready(self) -> bool:
        """Return True iff a model has been loaded successfully."""
        return self._model is not None

    # ── prediction ────────────────────────────────────────────────

    def predict(self, sentences: list, threshold: float = 0.5) -> list:
        """
        Predict author-switch labels for each consecutive sentence pair.

        Parameters
        ----------
        sentences : List[str]   – one sentence per element (at least 2)
        threshold : float       – probability cutoff for labelling a pair as a switch

        Returns
        -------
        List[dict] with keys:
            index, sentence_a, sentence_b, switch (bool), probability (float | None)
        """
        if not self.is_ready():
            raise RuntimeError("ModelService is not ready. Call load() first.")

        import numpy as np

        fe   = self._extractor
        vecs = fe.transform_document(sentences)
        pairs = []

        for i in range(len(sentences) - 1):
            diff = abs(vecs[i] - vecs[i + 1]).reshape(1, -1)

            if hasattr(self._model, "predict_proba"):
                prob  = float(self._model.predict_proba(diff)[0, 1])
                label = int(prob >= threshold)
            else:
                prob  = None
                label = int(self._model.predict(diff)[0])

            pairs.append({
                "index":      i,
                "sentence_a": sentences[i],
                "sentence_b": sentences[i + 1],
                "switch":     label == 1,
                "probability": round(prob, 3) if prob is not None else None,
            })

        return pairs

    # ── inspection ────────────────────────────────────────────────

    def get_metrics(self) -> dict:
        """Return the most recently saved evaluation metrics."""
        return dict(self._metrics)

    def __repr__(self) -> str:
        status = "ready" if self.is_ready() else "not loaded"
        return f"ModelService(results_dir='{self.results_dir}', status={status})"
