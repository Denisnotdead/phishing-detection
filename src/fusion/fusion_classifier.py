"""LightGBM meta-classifier combining XGBoost, DistilBERT and OCR signals."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_COLS = ["xgb_prob", "bert_prob", "ocr_confidence", "is_image_input"]

# Fallback weights for weighted-average when LightGBM model is not yet trained.
# BERT gets a slightly higher weight because it operates on raw semantics.
_FALLBACK_WEIGHTS = {
    "xgb_prob":       0.40,
    "bert_prob":       0.45,
    "ocr_confidence":  0.15,
}


def confidence_tier(prob: float) -> str:
    """Map a phishing probability to a human-readable confidence label.

    Distance from 0.5 is used so both sides of the boundary are treated symmetrically:
    LOW (<0.10 from 0.5), MEDIUM (0.10–0.35), HIGH (>0.35).
    """
    distance = abs(prob - 0.5)
    if distance < 0.10:
        return "LOW"
    if distance < 0.35:
        return "MEDIUM"
    return "HIGH"


class FusionClassifier:
    """LightGBM meta-classifier fusing XGBoost, DistilBERT, and image signals.

    Falls back to a weighted average of input signals until a model is trained and saved.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 4,  # kept shallow — only four input features
        models_dir: str | Path = "models",
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models_dir = Path(models_dir)
        self._model = None

    @property
    def is_trained(self) -> bool:
        """True if a LightGBM model has been fitted or loaded."""
        return self._model is not None

    def fit(self, X: pd.DataFrame, y) -> "FusionClassifier":
        """Train the LightGBM meta-classifier on pre-computed signal features."""
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError(
                "lightgbm is required for FusionClassifier. "
                "Install it with: pip install lightgbm"
            ) from exc

        missing = set(FEATURE_COLS) - set(X.columns)
        if missing:
            raise ValueError(
                f"Input DataFrame is missing required columns: {missing}. "
                f"Expected: {FEATURE_COLS}"
            )

        y = np.asarray(y).astype(int)
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos

        self._model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            scale_pos_weight=n_neg / max(n_pos, 1),
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

        logger.info(
            "Training FusionClassifier on %d samples (%d phishing, %d legitimate)...",
            len(y), n_pos, n_neg,
        )
        self._model.fit(X[FEATURE_COLS], y)
        logger.info("FusionClassifier training complete.")
        return self

    def predict_proba(self, signals: dict | pd.DataFrame) -> float:
        """Return the phishing probability for one set of input signals.

        Falls back to a weighted average with a warning if the model is not trained.
        """
        if isinstance(signals, dict):
            row = pd.DataFrame([signals])[FEATURE_COLS]
        else:
            row = signals[FEATURE_COLS].head(1)

        if not self.is_trained:
            logger.warning(
                "FusionClassifier has not been trained. "
                "Falling back to weighted average of input signals. "
                "Run FusionClassifier.fit() to enable the meta-learner."
            )
            return self._weighted_average_fallback(row.iloc[0].to_dict())

        prob = float(self._model.predict_proba(row)[0, 1])
        return prob

    def _weighted_average_fallback(self, signals: dict) -> float:
        """Compute a weighted average when no trained model is available.

        OCR confidence is excluded for text-only inputs to avoid diluting the text scores.
        """
        if signals.get("is_image_input", 0):
            total_weight = sum(_FALLBACK_WEIGHTS.values())
            prob = sum(
                _FALLBACK_WEIGHTS[k] * signals.get(k, 0.5)
                for k in _FALLBACK_WEIGHTS
            ) / total_weight
        else:
            w_xgb  = _FALLBACK_WEIGHTS["xgb_prob"]
            w_bert = _FALLBACK_WEIGHTS["bert_prob"]
            prob = (
                w_xgb  * signals.get("xgb_prob",  0.5) +
                w_bert * signals.get("bert_prob", 0.5)
            ) / (w_xgb + w_bert)

        return float(np.clip(prob, 0.0, 1.0))

    def save(self, name: str = "fusion_classifier") -> Path:
        """Save the trained LightGBM model to models_dir/<name>.pkl."""
        if not self.is_trained:
            raise RuntimeError("Nothing to save — model has not been fitted yet.")

        self.models_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.models_dir / f"{name}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(self._model, f)
        logger.info("FusionClassifier saved to %s", save_path)
        return save_path

    def load(self, name: str = "fusion_classifier") -> "FusionClassifier":
        """Load a previously saved model from models_dir/<name>.pkl."""
        load_path = self.models_dir / f"{name}.pkl"
        with open(load_path, "rb") as f:
            self._model = pickle.load(f)
        logger.info("FusionClassifier loaded from %s", load_path)
        return self
