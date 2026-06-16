"""
Cross-validation, held-out evaluation, and permutation feature importance.
"""

# TODO: All evaluations will be replaced with my definitions.

import numpy as np
from sklearn.inspection import permutation_importance as _perm_imp
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_validate

from src.models.classifiers import MODEL_REGISTRY


def cross_validate_model(model, X, y, groups, n_splits=5):
    """
    Stratified group k-fold CV scored by F1.
    Groups keep sentences from the same problem in the same fold,
    which prevents data leakage between train and validation sets.
    """
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_validate(
        model, X, y,
        groups=groups,
        cv=cv,
        scoring="f1",
        return_train_score=True,
        n_jobs=-1,
    )
    return {
        "test_f1_mean": float(np.mean(scores["test_score"])),
        "test_f1_std": float(np.std(scores["test_score"])),
        "train_f1_mean": float(np.mean(scores["train_score"])),
        "train_f1_std": float(np.std(scores["train_score"])),
    }


def compare_all_models(X, y, groups, n_splits=5):
    """Run cross-validation for every model in MODEL_REGISTRY."""
    results = {}
    for name, factory in MODEL_REGISTRY.items():
        print(f"[evaluation] Cross-validating: {name} ...")
        res = cross_validate_model(factory(), X, y, groups, n_splits=n_splits)
        results[name] = res
        print(f"  F1 = {res['test_f1_mean']:.4f} ± {res['test_f1_std']:.4f}")
    return results


def evaluate_model(model, X_test, y_test, model_name="model"):
    """Return and print precision / recall / F1 and confusion matrix."""
    y_pred = model.predict(X_test)

    f1 = f1_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=["same_author", "switch"],
        zero_division=0,
    )

    print(f"\n{'=' * 55}\n  {model_name}\n{'=' * 55}")
    print(f"  F1        : {f1:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"\n{report}")
    print(f"Confusion matrix:\n{cm}\n")

    return {
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "confusion_matrix": cm,
        "report": report,
    }


def permutation_feature_importance(model, X_val, y_val, feature_names=None,
                                   n_repeats=10, top_k=20):
    """
    Rank features by how much F1 drops when each is randomly shuffled.
    A large drop means the feature matters; near zero means the model ignores it.
    """
    result = _perm_imp(
        model, X_val, y_val,
        n_repeats=n_repeats,
        scoring="f1",
        random_state=42,
        n_jobs=-1,
    )

    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(
            len(result.importances_mean))]

    ranked = sorted(
        zip(feature_names, result.importances_mean, result.importances_std),
        key=lambda x: -x[1],
    )[:top_k]

    print(f"\nTop-{top_k} features by permutation importance:")
    for name, imp, std in ranked:
        print(f"  {name:<33} {imp:>10.4f} ± {std:.4f}")

    return [
        {"name": n, "importance": float(imp), "std": float(std)}
        for n, imp, std in ranked
    ]
