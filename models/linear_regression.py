"""
    linear_regression.py — Custom Linear Regression implementation.

    Implements the model:  y = mX + c

    Class
    -----
    LinearRegression
        fit(X, y)            — learn weights m and bias c using the normal equation
        predict(X)           — return continuous predictions
        predict_binary(X)    — threshold predictions at 0.5 → 0 / 1 labels
        score(X, y)          — R² on a held-out set
        get_params()         — return {'m': ..., 'c': ...}
"""

import numpy as np


class LinearRegression:
    """
    Simple closed-form linear regression: y = mX + c

    Solved via the normal equation:
        [m, c] = (X_aug^T X_aug)^{-1}  X_aug^T  y
    where X_aug is X with a column of ones appended for the bias term.

    Parameters
    ----------
    regularization : float
        L2 (ridge) regularisation strength added to the diagonal of
        (X_aug^T X_aug) to keep the solution numerically stable.
        Default: 1e-4.
    """

    def __init__(self, regularization: float = 1e-4):
        self.regularization = regularization
        self.m: np.ndarray | None = None   # weight vector (n_features,)
        self.c: float | None      = None   # bias term

    # ── training ──────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """
        Learn weights m and bias c from training data.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
        y : array-like, shape [n_samples]

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        n_samples, n_features = X.shape

        # Augment X with a bias column of ones  →  [X | 1]
        ones  = np.ones((n_samples, 1), dtype=np.float64)
        X_aug = np.hstack([X, ones])                         # (n, d+1)

        # Normal equation: w = (X_aug^T X_aug + λI)^{-1} X_aug^T y
        A      = X_aug.T @ X_aug
        A     += self.regularization * np.eye(A.shape[0])
        w      = np.linalg.solve(A, X_aug.T @ y)            # (d+1,)

        self.m = w[:-1]   # feature weights
        self.c = float(w[-1])   # bias

        return self

    # ── inference ─────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return continuous predictions  ŷ = Xm + c.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]

        Returns
        -------
        np.ndarray, shape [n_samples]
        """
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        return X @ self.m + self.c

    def predict_binary(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Threshold continuous predictions to produce 0 / 1 labels.

        Parameters
        ----------
        X         : array-like, shape [n_samples, n_features]
        threshold : float  (default 0.5)

        Returns
        -------
        np.ndarray of int32, shape [n_samples]
        """
        return (self.predict(X) >= threshold).astype(np.int32)

    # ── evaluation ────────────────────────────────────────────────

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the coefficient of determination R² on a held-out set.

        R² = 1 - SS_res / SS_tot
        """
        self._check_fitted()
        y     = np.asarray(y, dtype=np.float64)
        y_hat = self.predict(X)
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return float(1.0 - ss_res / ss_tot)

    # ── inspection ────────────────────────────────────────────────

    def get_params(self) -> dict:
        """Return the learned parameters as a dict."""
        self._check_fitted()
        return {"m": self.m.tolist(), "c": self.c}

    def __repr__(self) -> str:
        fitted = self.m is not None
        return (
            f"LinearRegression(regularization={self.regularization}, "
            f"fitted={fitted})"
        )

    # ── internal ──────────────────────────────────────────────────

    def _check_fitted(self):
        if self.m is None:
            raise RuntimeError(
                "LinearRegression is not fitted yet. Call fit(X, y) first."
            )


# ── Entry-point example ───────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    X   = rng.normal(size=(200, 4)).astype(np.float32)
    y   = (X[:, 0] * 2.0 + X[:, 1] * -1.5 + 0.3 + rng.normal(scale=0.1, size=200))

    lr = LinearRegression(regularization=1e-4)
    lr.fit(X, y)

    print(lr)
    print("Params:", lr.get_params())
    print("R²    :", round(lr.score(X, y), 4))

    # Binary prediction demo (e.g. switch / no-switch)
    y_bin  = (y > y.mean()).astype(int)
    lr_bin = LinearRegression()
    lr_bin.fit(X, y_bin)
    preds  = lr_bin.predict_binary(X)
    acc    = (preds == y_bin).mean()
    print(f"Binary accuracy: {acc:.2%}")
