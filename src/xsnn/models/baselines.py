"""Linear and tree baselines fit on cross-sectionally ranked features."""

from __future__ import annotations

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

from xsnn.features import FEATURE_COLUMNS


class MomentumReversalScore:
    """Equal-weight 12-1 / 6-1 / 3-1 momentum minus one-month reversal.

    Inputs are already rank-normalized, so the score is itself a rank composite.
    There is nothing to fit; ``fit`` exists so the walk-forward loop is uniform.
    """

    def __init__(self, feature_names: list[str] | None = None) -> None:
        names = list(feature_names or FEATURE_COLUMNS)
        self.feature_names = names
        self._mom = [names.index(name) for name in ("mom_12_1", "mom_6_1", "mom_3_1")]
        self._rev = names.index("rev_1m")

    def fit(self, features: np.ndarray, target: np.ndarray, dates: np.ndarray) -> MomentumReversalScore:
        del features, target, dates
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        momentum = features[:, self._mom].mean(axis=1)
        return momentum - features[:, self._rev]


class RidgeRanker:
    """Ridge regression of the ranked next-month return on ranked features."""

    def __init__(self, alpha: float) -> None:
        self.alpha = float(alpha)
        self.model = Ridge(alpha=self.alpha)

    def fit(self, features: np.ndarray, target: np.ndarray, dates: np.ndarray) -> RidgeRanker:
        del dates
        self.model.fit(features, target)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(features), dtype=float)


class LightGBMRanker:
    """Gradient-boosted trees on the same ranked target."""

    def __init__(
        self,
        *,
        n_estimators: int,
        learning_rate: float,
        num_leaves: int,
        min_child_samples: int,
        subsample: float,
        colsample_bytree: float,
        seed: int,
    ) -> None:
        self.params = {
            "n_estimators": int(n_estimators),
            "learning_rate": float(learning_rate),
            "num_leaves": int(num_leaves),
            "min_child_samples": int(min_child_samples),
            "subsample": float(subsample),
            "colsample_bytree": float(colsample_bytree),
            "random_state": int(seed),
        }
        self.model = LGBMRegressor(
            objective="regression",
            n_jobs=1,
            verbosity=-1,
            force_row_wise=True,
            **self.params,
        )

    def fit(self, features: np.ndarray, target: np.ndarray, dates: np.ndarray) -> LightGBMRanker:
        del dates
        self.model.fit(features, target)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(features), dtype=float)
