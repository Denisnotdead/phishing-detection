"""Evaluates all models and saves results to reports/."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

DEFAULT_MODELS_DIR  = PROJECT_ROOT / "models"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_PROCESSED   = PROJECT_ROOT / "data" / "processed" / "combined_dataset.csv"

# BERT inference is slow, so evaluation is capped at this many test samples.
# XGBoost evaluates on the full test set.
DEFAULT_BERT_EVAL_SAMPLES = 5_000


class PhishingEvaluator:
    """Loads the held-out test split and evaluates XGBoost, DistilBERT, and the fusion classifier."""

    def __init__(
        self,
        models_dir: str | Path = DEFAULT_MODELS_DIR,
        reports_dir: str | Path = DEFAULT_REPORTS_DIR,
        processed_path: str | Path = DEFAULT_PROCESSED,
        bert_eval_samples: int = DEFAULT_BERT_EVAL_SAMPLES,
        test_size: float = 0.15,
        random_state: int = 42,
    ):
        self.models_dir        = Path(models_dir)
        self.reports_dir       = Path(reports_dir)
        self.processed_path    = Path(processed_path)
        self.bert_eval_samples = bert_eval_samples
        self.test_size         = test_size
        self.random_state      = random_state

    def load_test_data(self) -> pd.DataFrame:
        """Load the test split, regenerating from raw data if the processed CSV is missing.

        The split parameters must match those used in trainer.py to get identical test rows.
        """
        if self.processed_path.exists():
            logger.info("Loading processed dataset from %s ...", self.processed_path)
            df = pd.read_csv(self.processed_path, low_memory=False)
            # urls are saved as pipe-delimited strings in the EDA notebook; restore lists
            if "urls" in df.columns:
                df["urls"] = df["urls"].fillna("").apply(
                    lambda s: s.split("|") if isinstance(s, str) and s else []
                )
            else:
                df["urls"] = [[] for _ in range(len(df))]
        else:
            logger.info(
                "Processed CSV not found at %s. Regenerating from raw data...",
                self.processed_path,
            )
            from src.text_pipeline.data_loader import load_all
            df = load_all()

        logger.info("Full dataset: %d rows", len(df))

        # Reproduce the exact same test split as trainer.py
        _, df_test = train_test_split(
            df,
            test_size=self.test_size,
            stratify=df["label"],
            random_state=self.random_state,
        )
        df_test = df_test.reset_index(drop=True)
        logger.info(
            "Test split: %d rows (%d phishing, %d legitimate)",
            len(df_test),
            int(df_test["label"].sum()),
            int((df_test["label"] == 0).sum()),
        )
        return df_test

    @staticmethod
    def _compute_metrics(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        model_name: str,
        threshold: float = 0.5,
    ) -> dict:
        """Compute scalar metrics, ROC curve, and PR curve for one model."""
        y_pred = (y_prob >= threshold).astype(int)

        fpr, tpr, roc_thresh = roc_curve(y_true, y_prob)
        prec_arr, rec_arr, pr_thresh = precision_recall_curve(y_true, y_prob)

        metrics = {
            "model":         model_name,
            "n_samples":     int(len(y_true)),
            "accuracy":      float(accuracy_score(y_true, y_pred)),
            "precision":     float(precision_score(y_true, y_pred, zero_division=0)),
            "recall":        float(recall_score(y_true, y_pred, zero_division=0)),
            "f1":            float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc":       float(roc_auc_score(y_true, y_prob)),
            "avg_precision": float(average_precision_score(y_true, y_prob)),
            "classification_report": classification_report(
                y_true, y_pred, target_names=["Legitimate", "Phishing"]
            ),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "roc_curve": {
                "fpr":        fpr.tolist(),
                "tpr":        tpr.tolist(),
                "thresholds": roc_thresh.tolist(),
            },
            "pr_curve": {
                "precision":  prec_arr.tolist(),
                "recall":     rec_arr.tolist(),
                "thresholds": pr_thresh.tolist(),
            },
        }

        logger.info(
            "%s  acc=%.4f  f1=%.4f  roc_auc=%.4f  avg_precision=%.4f",
            model_name.ljust(12),
            metrics["accuracy"],
            metrics["f1"],
            metrics["roc_auc"],
            metrics["avg_precision"],
        )
        return metrics

    def evaluate_xgb(self, df_test: pd.DataFrame) -> dict:
        """Load and evaluate PhishingXGBClassifier on the full test set, including SHAP values."""
        from src.text_pipeline.text_classifier import PhishingXGBClassifier

        logger.info("Evaluating XGBoost...")
        clf = PhishingXGBClassifier(models_dir=self.models_dir).load("xgb_phishing")

        y_true  = df_test["label"].values.astype(int)
        y_prob  = clf.predict_proba(df_test)[:, 1]
        metrics = self._compute_metrics(y_true, y_prob, "XGBoost")

        # Compute SHAP values; _transform applies _build_features AND the fitted scaler
        # which matches exactly what the model saw during training
        try:
            X_scaled    = clf._transform(df_test)
            shap_values = clf._explainer.shap_values(X_scaled)
            mean_abs    = np.abs(shap_values).mean(axis=0)
            feature_names = clf._feature_names or [f"f{i}" for i in range(len(mean_abs))]
            importance = dict(
                sorted(
                    zip(feature_names, mean_abs.tolist()),
                    key=lambda x: x[1],
                    reverse=True,
                )
            )
            top20 = dict(list(importance.items())[:20])
            metrics["shap_feature_importances"] = top20
            logger.info("SHAP importances computed for %d features.", len(importance))
        except Exception as exc:
            logger.warning("SHAP computation failed: %s", exc)
            metrics["shap_feature_importances"] = {}

        # Attach for use by explainability.py (not serialised to JSON)
        metrics["_clf"]    = clf
        metrics["_y_prob"] = y_prob

        return metrics

    def evaluate_bert(self, df_test: pd.DataFrame) -> dict:
        """Load and evaluate PhishingBERTClassifier on a stratified sample of the test set.

        Uses a sample of bert_eval_samples rows because BERT inference is slow on a 4 GB GPU.
        """
        from src.text_pipeline.bert_classifier import PhishingBERTClassifier

        bert_dir = self.models_dir / "bert_phishing"
        if not bert_dir.exists():
            logger.warning("DistilBERT checkpoint not found at %s, skipping.", bert_dir)
            return {"model": "DistilBERT", "skipped": True}

        if self.bert_eval_samples and len(df_test) > self.bert_eval_samples:
            logger.info(
                "Sampling %d of %d test rows for BERT evaluation.",
                self.bert_eval_samples, len(df_test),
            )
            # Sample each class separately to preserve index alignment with df_test.
            # Do NOT call reset_index — evaluate_fusion relies on the original df_test
            # row positions to align XGBoost and BERT probabilities correctly.
            per_class = self.bert_eval_samples // 2
            phishing_rows = df_test[df_test["label"] == 1].sample(
                min(int((df_test["label"] == 1).sum()), per_class),
                random_state=42,
            )
            legit_rows = df_test[df_test["label"] == 0].sample(
                min(int((df_test["label"] == 0).sum()), per_class),
                random_state=42,
            )
            df_eval = pd.concat([phishing_rows, legit_rows])
        else:
            df_eval = df_test

        logger.info("df_eval columns: %s", list(df_eval.columns))
        logger.info("Evaluating DistilBERT on %d samples...", len(df_eval))
        clf = PhishingBERTClassifier(models_dir=str(self.models_dir)).load("bert_phishing")

        y_true  = df_eval["label"].values.astype(int)
        y_prob  = clf.predict_proba(df_eval)[:, 1]
        metrics = self._compute_metrics(y_true, y_prob, "DistilBERT")

        metrics["_clf"]     = clf
        metrics["_y_prob"]  = y_prob
        metrics["_df_eval"] = df_eval

        return metrics

    def evaluate_fusion(
        self,
        df_test: pd.DataFrame,
        xgb_proba: np.ndarray,
        bert_proba: np.ndarray,
        bert_df: pd.DataFrame,
    ) -> dict:
        """Evaluate the fusion classifier using pre-computed XGBoost and BERT probabilities.

        ocr_confidence is set to 0.5 (neutral) and is_image_input to 0 for the text-only test set.
        """
        from src.fusion.fusion_classifier import FusionClassifier, FEATURE_COLS

        # Log file existence for debugging
        xgb_json   = self.models_dir / "xgb_phishing.json"
        xgb_pkl    = self.models_dir / "xgb_phishing.pkl"
        bert_dir   = self.models_dir / "bert_phishing"
        fusion_file = self.models_dir / "fusion_classifier.pkl"

        logger.info("evaluate_fusion — models_dir  : %s", self.models_dir)
        logger.info("  xgb_phishing.json  exists   : %s  (%s)", xgb_json.exists(),  xgb_json)
        logger.info("  xgb_phishing.pkl   exists   : %s  (%s)", xgb_pkl.exists(),   xgb_pkl)
        logger.info("  bert_phishing/     exists   : %s  (%s)", bert_dir.exists(),  bert_dir)
        logger.info("  fusion_classifier.pkl exists: %s  (%s)", fusion_file.exists(), fusion_file)

        # Align on index — bert_df carries original df_test row positions (no reset_index
        # was called in evaluate_bert) so xgb_proba[bert_index] pulls the matching XGBoost score
        bert_index = bert_df.index
        xgb_sub    = xgb_proba[bert_index]
        y_true     = df_test["label"].values[bert_index].astype(int)

        logger.info(
            "Input probabilities — xgb : min=%.3f  mean=%.3f  max=%.3f  (n=%d)",
            float(xgb_sub.min()), float(xgb_sub.mean()), float(xgb_sub.max()), len(xgb_sub),
        )
        logger.info(
            "Input probabilities — bert: min=%.3f  mean=%.3f  max=%.3f  (n=%d)",
            float(bert_proba.min()), float(bert_proba.mean()), float(bert_proba.max()), len(bert_proba),
        )
        logger.info(
            "Ground-truth labels       : %d phishing, %d legitimate",
            int(y_true.sum()), int((y_true == 0).sum()),
        )

        signals_df = pd.DataFrame({
            "xgb_prob":       xgb_sub,
            "bert_prob":      bert_proba,
            "ocr_confidence": 0.5,
            "is_image_input": 0,
        })

        fusion = FusionClassifier(models_dir=self.models_dir)
        if fusion_file.exists():
            try:
                fusion.load("fusion_classifier")
                logger.info("Fusion model loaded successfully — using LightGBM meta-learner.")
                fusion_proba = fusion._model.predict_proba(signals_df[FEATURE_COLS])[:, 1]
            except Exception as exc:
                logger.error("Fusion model failed to load: %s — falling back to weighted average.", exc)
                fusion_proba = np.array([
                    fusion._weighted_average_fallback(row.to_dict())
                    for _, row in signals_df.iterrows()
                ])
        else:
            logger.warning(
                "fusion_classifier.pkl not found at %s — using weighted-average fallback. "
                "Train and save the fusion model first for proper fusion evaluation.",
                fusion_file,
            )
            fusion_proba = np.array([
                fusion._weighted_average_fallback(row.to_dict())
                for _, row in signals_df.iterrows()
            ])

        logger.info(
            "Fusion probabilities      : min=%.3f  mean=%.3f  max=%.3f",
            float(fusion_proba.min()), float(fusion_proba.mean()), float(fusion_proba.max()),
        )

        metrics = self._compute_metrics(y_true, fusion_proba, "Fusion (Ensemble)")
        return metrics

    def print_comparison_table(self, report: dict) -> None:
        """Print a side-by-side comparison table of all evaluated models."""
        model_keys = [k for k in ("xgboost", "bert", "fusion") if k in report
                      and not report[k].get("skipped")]

        col_width    = 13
        metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc", "avg_precision"]
        header_map   = {
            "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
            "f1": "F1", "roc_auc": "ROC-AUC", "avg_precision": "Avg Precision",
        }

        header = f"{'Metric':<18}" + "".join(
            report[k]["model"][:col_width].ljust(col_width) for k in model_keys
        )
        print()
        print(header)
        print("-" * len(header))
        for m in metric_names:
            row = f"{header_map[m]:<18}"
            for k in model_keys:
                val = report[k].get(m)
                row += f"{val:.4f}".ljust(col_width) if val is not None else "N/A".ljust(col_width)
            print(row)
        print()

    def save_report(self, report: dict) -> Path:
        """Write evaluation results to reports_dir/evaluation_report.json.

        Strips classification_report strings and private _clf/_y_prob/_df_eval keys.
        """
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        def clean(d: dict) -> dict:
            skip = {"_clf", "_y_prob", "_df_eval", "classification_report"}
            return {k: v for k, v in d.items() if k not in skip}

        serialisable = {}
        for key, val in report.items():
            if isinstance(val, dict):
                serialisable[key] = clean(val)
            else:
                serialisable[key] = val

        out_path = self.reports_dir / "evaluation_report.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(serialisable, f, indent=2)

        logger.info("Evaluation report saved to %s", out_path)
        return out_path

    def run_evaluation(self) -> dict:
        """Run the full evaluation pipeline: load data, evaluate all models, save report and plots."""
        df_test = self.load_test_data()

        report: dict = {
            "dataset_info": {
                "n_test":       int(len(df_test)),
                "n_phishing":   int(df_test["label"].sum()),
                "n_legitimate": int((df_test["label"] == 0).sum()),
                "sources":      df_test["source"].value_counts().to_dict()
                                if "source" in df_test.columns else {},
            }
        }

        xgb_metrics = self.evaluate_xgb(df_test)
        report["xgboost"] = xgb_metrics
        report["shap_feature_importances"] = xgb_metrics.get("shap_feature_importances", {})

        bert_metrics = self.evaluate_bert(df_test)
        report["bert"] = bert_metrics

        if not bert_metrics.get("skipped"):
            fusion_metrics = self.evaluate_fusion(
                df_test=df_test,
                xgb_proba=xgb_metrics["_y_prob"],
                bert_proba=bert_metrics["_y_prob"],
                bert_df=bert_metrics.get("_df_eval", df_test),
            )
            report["fusion"] = fusion_metrics
        else:
            logger.warning("Skipping fusion evaluation because BERT was not available.")

        self.print_comparison_table(report)
        self.save_report(report)

        try:
            from src.evaluation.explainability import XGBExplainer, BERTAttentionVisualizer

            xgb_clf   = report["xgboost"].get("_clf")
            xgb_yprob = report["xgboost"].get("_y_prob")

            if xgb_clf is not None and xgb_yprob is not None:
                logger.info("Generating SHAP plots...")
                xgb_exp = XGBExplainer(xgb_clf, reports_dir=self.reports_dir)
                xgb_exp.explain_all(
                    df=df_test,
                    y_true=df_test["label"],
                    y_prob=xgb_yprob,
                    top_n=20,
                )

            bert_clf   = report.get("bert", {}).get("_clf")
            bert_yprob = report.get("bert", {}).get("_y_prob")
            bert_df    = report.get("bert", {}).get("_df_eval", df_test)

            if bert_clf is not None and bert_yprob is not None:
                logger.info("Generating DistilBERT attention plots...")
                bert_viz = BERTAttentionVisualizer(bert_clf, reports_dir=self.reports_dir)
                bert_viz.explain_both(
                    df=bert_df,
                    y_true=bert_df["label"],
                    y_prob=bert_yprob,
                    bert_clf=bert_clf,
                )

        except Exception as exc:
            logger.warning(
                "Explainability plots could not be generated: %s. "
                "Evaluation report and metrics were still saved.",
                exc,
            )

        return report


def load_test_data(
    processed_path: Path = DEFAULT_PROCESSED,
    test_size: float = 0.15,
    random_state: int = 42,
) -> pd.DataFrame:
    """Module-level wrapper around PhishingEvaluator.load_test_data() for notebook convenience."""
    return PhishingEvaluator(
        processed_path=processed_path,
        test_size=test_size,
        random_state=random_state,
    ).load_test_data()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    PhishingEvaluator().run_evaluation()
