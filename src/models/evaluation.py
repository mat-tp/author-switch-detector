"""
Cross-validation, held-out evaluation, and permutation feature importance.

"""

import numpy as np

from src.models.classifiers import MODEL_REGISTRY


# ── Metrics ────────────────────────────────────────────────────────────────────

def confusion_counts(y_true, y_pred):
    """Return (tp, fp, tn, fn) for binary labels 0/1."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return tp, fp, tn, fn


def confusion_matrix(y_true, y_pred):
    """Return a 2x2 array [[tn, fp], [fn, tp]] (row=true, col=pred)."""
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)
    return np.array([[tn, fp], [fn, tp]])


def precision_score(y_true, y_pred):
    """precision = tp / (tp + fp), 0.0 if undefined."""
    tp, fp, _, _ = confusion_counts(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) else 0.0


def recall_score(y_true, y_pred):
    """recall = tp / (tp + fn), 0.0 if undefined."""
    tp, _, _, fn = confusion_counts(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) else 0.0


def f1_score(y_true, y_pred):
    """harmonic mean of precision and recall, 0.0 if undefined."""
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def accuracy_score(y_true, y_pred):
    """fraction of correct predictions."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred)) if len(y_true) else 0.0


def classification_report(y_true, y_pred, target_names=("same_author", "switch")):
    """Plain-text precision/recall/F1 per class, in the style of sklearn's report."""
    lines = [f"{'':<14}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>10}"]
    for cls, name in enumerate(target_names):
        y_true_bin = (np.asarray(y_true) == cls).astype(int)
        y_pred_bin = (np.asarray(y_pred) == cls).astype(int)
        p = precision_score(y_true_bin, y_pred_bin)
        r = recall_score(y_true_bin, y_pred_bin)
        f1 = f1_score(y_true_bin, y_pred_bin)
        support = int(np.sum(np.asarray(y_true) == cls))
        lines.append(f"{name:<14}{p:>10.2f}{r:>10.2f}{f1:>10.2f}{support:>10}")
    acc = accuracy_score(y_true, y_pred)
    lines.append("")
    lines.append(f"{'accuracy':<14}{'':>10}{'':>10}{acc:>10.2f}{len(y_true):>10}")
    return "\n".join(lines)


# ── Custom k-fold splitter (stratified + grouped) ───────────────────────────────

def stratified_group_kfold(y, groups, n_splits, shuffle=True, random_state=42):
    """
    Yield (train_idx, val_idx) pairs, n_splits folds.

    Keeps every sample from the same group (problem_id) in a single fold, and
    tries to keep each fold's positive-label rate close to the overall rate —
    a from-scratch replacement for sklearn's StratifiedGroupKFold.

    Greedy assignment: groups are sorted by size (largest first) and each one
    goes into whichever fold currently has the lowest "deficit" — the
    difference between its running positive-rate and the dataset's overall
    positive-rate, weighted by fold size. This keeps folds both size-balanced
    and label-balanced without needing an external library.
    """
    y = np.asarray(y)
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)

    rng = np.random.default_rng(random_state)
    if shuffle:
        rng.shuffle(unique_groups)

    # Positive rate per group, and group sizes
    group_pos_rate = {}
    group_size = {}
    for g in unique_groups:
        mask = groups == g
        group_size[g] = int(np.sum(mask))
        group_pos_rate[g] = float(np.mean(y[mask]))

    # Sort groups largest-first so big groups get placed before small ones
    ordered = sorted(unique_groups, key=lambda g: -group_size[g])

    overall_pos_rate = float(np.mean(y))
    fold_size  = [0] * n_splits
    fold_pos   = [0] * n_splits
    fold_groups = [[] for _ in range(n_splits)]

    for g in ordered:
        best_fold, best_score = 0, None
        for f in range(n_splits):
            new_size = fold_size[f] + group_size[g]
            new_pos  = fold_pos[f] + group_pos_rate[g] * group_size[g]
            new_rate = new_pos / new_size if new_size else 0.0
            # Penalise both label-rate drift and size imbalance
            score = abs(new_rate - overall_pos_rate) + 0.01 * new_size
            if best_score is None or score < best_score:
                best_score, best_fold = score, f
        fold_groups[best_fold].append(g)
        fold_size[best_fold] += group_size[g]
        fold_pos[best_fold]  += group_pos_rate[g] * group_size[g]

    for f in range(n_splits):
        val_groups = set(fold_groups[f])
        val_idx   = np.where(np.isin(groups, list(val_groups)))[0]
        train_idx = np.where(~np.isin(groups, list(val_groups)))[0]
        yield train_idx, val_idx


# ── Cross-validation ──────────────────────────────────────────────────────────

def cross_validate_model(model_factory, X, y, groups, n_splits=5):
    """
    Run our own stratified-group k-fold CV, scored by F1.

    model_factory : callable that returns a fresh, unfitted model each call
                    (e.g. MODEL_REGISTRY["logistic_regression"])
    """
    test_f1, train_f1 = [], []

    for train_idx, val_idx in stratified_group_kfold(y, groups, n_splits):
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])

        train_pred = model.predict(X[train_idx])
        val_pred   = model.predict(X[val_idx])

        train_f1.append(f1_score(y[train_idx], train_pred))
        test_f1.append(f1_score(y[val_idx], val_pred))

    return {
        "test_f1_mean":  float(np.mean(test_f1)),
        "test_f1_std":   float(np.std(test_f1)),
        "train_f1_mean": float(np.mean(train_f1)),
        "train_f1_std":  float(np.std(train_f1)),
    }


def compare_all_models(X, y, groups, n_splits=5):
    """Run cross-validation for every model in MODEL_REGISTRY."""
    results = {}
    for name, factory in MODEL_REGISTRY.items():
        print(f"[evaluation] Cross-validating: {name} ...")
        res = cross_validate_model(factory, X, y, groups, n_splits=n_splits)
        results[name] = res
        print(f"  F1 = {res['test_f1_mean']:.4f} +/- {res['test_f1_std']:.4f}")
    return results


# ── Held-out evaluation ──────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name="model"):
    """Return and print precision / recall / F1 and confusion matrix."""
    y_pred = model.predict(X_test)

    f1   = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    cm   = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["same_author", "switch"])

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


# ── Permutation feature importance ──────────────────────────────────────────────

def permutation_feature_importance(model, X_val, y_val, feature_names=None,
                                    n_repeats=10, top_k=20, random_state=42):
    """
    Rank features by how much F1 drops when each column is randomly shuffled.

    For each feature: shuffle that column n_repeats times, re-score F1 each
    time, and report the average drop from the baseline. A large drop means
    the model relies heavily on that feature; near zero means it's ignored.
    """
    rng = np.random.default_rng(random_state)
    X_val = np.asarray(X_val)
    n_features = X_val.shape[1]

    baseline_pred = model.predict(X_val)
    baseline_f1   = f1_score(y_val, baseline_pred)

    means = np.zeros(n_features)
    stds  = np.zeros(n_features)

    for col in range(n_features):
        drops = np.empty(n_repeats)
        for r in range(n_repeats):
            X_shuffled = X_val.copy()
            perm = rng.permutation(len(X_shuffled))
            X_shuffled[:, col] = X_shuffled[perm, col]
            shuffled_pred = model.predict(X_shuffled)
            shuffled_f1   = f1_score(y_val, shuffled_pred)
            drops[r] = baseline_f1 - shuffled_f1
        means[col] = drops.mean()
        stds[col]  = drops.std()

    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(n_features)]

    ranked = sorted(zip(feature_names, means, stds), key=lambda x: -x[1])[:top_k]

    print(f"\nTop-{top_k} features by permutation importance:")
    for name, imp, std in ranked:
        print(f"  {name:<33} {imp:>10.4f} +/- {std:.4f}")

    return [
        {"name": n, "importance": float(imp), "std": float(std)}
        for n, imp, std in ranked
    ]


# ── Entry-point example ───────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_samples, n_features = 200, 6
    X = rng.normal(size=(n_samples, n_features)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] * 0.5 > 0).astype(int)
    groups = rng.integers(0, 40, size=n_samples)   # 40 "problems"

    factory = MODEL_REGISTRY["logistic_regression"]
    cv = cross_validate_model(factory, X, y, groups, n_splits=4)
    print("CV results:", cv)

    model = factory()
    model.fit(X, y)
    metrics = evaluate_model(model, X, y, "demo_model")

    names = [f"f{i}" for i in range(n_features)]
    permutation_feature_importance(model, X, y, feature_names=names, n_repeats=5, top_k=6)
