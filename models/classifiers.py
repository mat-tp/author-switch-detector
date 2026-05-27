"""
    Classifiers wrapped in a sklearn Pipeline with StandardScaler.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def make_logistic_regression():
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


def make_svm():
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


def make_knn():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=5, metric="euclidean")),
    ])


def make_mlp():
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


MODEL_REGISTRY = {
    "logistic_regression": make_logistic_regression,
    "svm": make_svm,
    "knn": make_knn,
    "mlp": make_mlp,
}


def train_final_model(model_name, X_train, y_train):
    """Instantiate, fit, and return the named model on the full training set."""
    if model_name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{model_name}'. Choose from: {list(MODEL_REGISTRY)}"
        )
    model = MODEL_REGISTRY[model_name]()
    model.fit(X_train, y_train)
    print(f"[classifiers] Trained '{model_name}' on {len(y_train)} samples.")
    return model
