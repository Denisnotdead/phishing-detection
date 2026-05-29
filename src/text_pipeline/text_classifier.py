"""
text_classifier.py
XGBoost-based phishing classifier using hand-crafted features from feature_extractor.py.
Handles feature extraction, scaling, SHAP explanation, and save/load internally.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from src.text_pipeline.feature_extractor import (
    StructuralFeatureTransformer,
    TextStatFeatureTransformer,
    URLFeatureTransformer,
)

logger = logging.getLogger(__name__)


class PhishingXGBClassifier:
    """XGBoost classifier for phishing detection with built-in feature extraction and SHAP support."""

    def __init__(
        self,
        n_estimators: int = 400,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        scale_pos_weight: Optional[float] = None,
        random_state: int = 42,
        models_dir: str | Path = "models",
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state
        self.models_dir = Path(models_dir)

        self._model: Optional[xgb.XGBClassifier] = None
        self._scaler: Optional[StandardScaler] = None
        self._feature_names: Optional[list[str]] = None
        self._explainer: Optional[shap.TreeExplainer] = None

        # Stateless transformers created once and reused
        self._url_transformer = URLFeatureTransformer()
        self._text_transformer = TextStatFeatureTransformer()
        self._struct_transformer = StructuralFeatureTransformer()

    def _build_features(self, df: pd.DataFrame) -> np.ndarray:
        """Run all three feature transformers and concatenate into a dense matrix."""
        if "urls" not in df.columns:
            df = df.copy()
            df["urls"] = [[] for _ in range(len(df))]

        url_feats    = self._url_transformer.fit_transform(df)
        text_feats   = self._text_transformer.fit_transform(df)
        struct_feats = self._struct_transformer.fit_transform(df)

        return np.hstack([url_feats, text_feats, struct_feats])

    def _get_feature_names(self) -> list[str]:
        """Collect feature names from all transformers in stack order."""
        url_names    = list(self._url_transformer.get_feature_names_out())
        text_names   = list(self._text_transformer.get_feature_names_out())
        struct_names = list(self._struct_transformer.get_feature_names_out())
        return url_names + text_names + struct_names

    def fit(self, df: pd.DataFrame, label_col: str = "label") -> "PhishingXGBClassifier":
        """Extract features, fit scaler and XGBoost model, then build SHAP explainer."""
        logger.info("Building feature matrix for XGBoost training...")
        X = self._build_features(df)
        y = df[label_col].values.astype(int)

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        self._feature_names = self._get_feature_names()

        # Auto-compute class weight if not supplied
        spw = self.scale_pos_weight
        if spw is None:
            n_neg = int((y == 0).sum())
            n_pos = int((y == 1).sum())
            spw = n_neg / max(n_pos, 1)
            logger.info(
                "Class balance: %d legit, %d phishing  (scale_pos_weight=%.2f)",
                n_neg, n_pos, spw,
            )

        self._model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            scale_pos_weight=spw,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=self.random_state,
            n_jobs=-1,
        )

        logger.info("Training XGBoost on %d samples, %d features...", *X_scaled.shape)
        self._model.fit(X_scaled, y)

        logger.info("Building SHAP TreeExplainer...")
        self._explainer = shap.TreeExplainer(self._model)

        logger.info("XGBoost training complete.")
        return self

    def _transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply the fitted scaler to features built from df."""
        if self._scaler is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        X = self._build_features(df)
        return self._scaler.transform(X)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict binary labels (0 or 1) for each row of df."""
        X = self._transform(df)
        return self._model.predict(X)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return class probability estimates; column 1 is the phishing probability."""
        X = self._transform(df)
        return self._model.predict_proba(X)

    def evaluate(self, df: pd.DataFrame, label_col: str = "label") -> dict:
        """Compute classification metrics on a labelled test DataFrame."""
        y_true = df[label_col].values.astype(int)
        y_pred = self.predict(df)
        y_prob = self.predict_proba(df)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
            "classification_report": classification_report(
                y_true, y_pred, target_names=["Legitimate", "Phishing"]
            ),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }

        logger.info(
            "XGBoost evaluation  acc=%.4f  f1=%.4f  roc_auc=%.4f",
            metrics["accuracy"], metrics["f1"], metrics["roc_auc"],
        )
        return metrics

    def explain_prediction(
        self,
        df_row: pd.DataFrame,
        top_n: int = 10,
    ) -> dict:
        """Return top SHAP features driving the prediction for a single sample.

        Positive SHAP values push toward phishing; negative toward legitimate.
        """
        if self._explainer is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        X = self._transform(df_row)
        shap_values = self._explainer.shap_values(X)

        # shap_values shape: (n_samples, n_features); take row 0
        row_shap = shap_values[0] if shap_values.ndim == 2 else shap_values

        feature_names = self._feature_names or [f"f{i}" for i in range(len(row_shap))]
        pairs = sorted(
            zip(feature_names, row_shap.tolist()),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        return {
            "predicted_label": int(self._model.predict(X)[0]),
            "predicted_prob": float(self._model.predict_proba(X)[0, 1]),
            "top_features": [
                {"feature": name, "shap_value": value}
                for name, value in pairs[:top_n]
            ],
        }

    def save(self, name: str = "xgb_phishing") -> Path:
        """Save scaler + feature names to <name>.pkl and model to <name>.json."""
        if self._model is None:
            raise RuntimeError("Nothing to save — model has not been fitted yet.")

        self.models_dir.mkdir(parents=True, exist_ok=True)

        pkl_path = self.models_dir / f"{name}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(
                {"scaler": self._scaler, "feature_names": self._feature_names}, f
            )

        model_path = self.models_dir / f"{name}.json"
        self._model.save_model(str(model_path))

        logger.info("XGBoost model saved to %s and %s", pkl_path, model_path)
        return model_path

    def load(self, name: str = "xgb_phishing") -> "PhishingXGBClassifier":
        """Load a previously saved model from models_dir."""
        pkl_path = self.models_dir / f"{name}.pkl"
        with open(pkl_path, "rb") as f:
            bundle = pickle.load(f)
        self._scaler = bundle["scaler"]
        self._feature_names = bundle["feature_names"]

        model_path = self.models_dir / f"{name}.json"
        self._model = xgb.XGBClassifier()
        self._model.load_model(str(model_path))

        self._explainer = shap.TreeExplainer(self._model)

        logger.info("XGBoost model loaded from %s", model_path)
        return self
