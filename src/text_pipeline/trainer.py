"""
trainer.py
End-to-end training script: loads data, trains XGBoost + DistilBERT, evaluates the ensemble.

Run from the project root:
    python -m src.text_pipeline.trainer
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, works without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.text_pipeline.bert_classifier import PhishingBERTClassifier
from src.text_pipeline.data_loader import load_all
from src.text_pipeline.text_classifier import PhishingXGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def split_dataset(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    label_col: str = "label",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train/val/test split preserving the phishing/legitimate ratio."""
    # Carve out the test set first, then split the remainder into train/val
    df_train_val, df_test = train_test_split(
        df,
        test_size=test_size,
        stratify=df[label_col],
        random_state=random_state,
    )

    # Adjust val fraction to be relative to the remaining train_val pool
    val_fraction_of_remainder = val_size / (1.0 - test_size)
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=val_fraction_of_remainder,
        stratify=df_train_val[label_col],
        random_state=random_state,
    )

    logger.info(
        "Split sizes  train=%d  val=%d  test=%d",
        len(df_train), len(df_val), len(df_test),
    )
    return df_train, df_val, df_test


def evaluate_ensemble(
    df: pd.DataFrame,
    xgb_clf: PhishingXGBClassifier,
    bert_clf: PhishingBERTClassifier,
    xgb_weight: float = 0.5,
    bert_weight: float = 0.5,
    label_col: str = "label",
    threshold: float = 0.5,
) -> dict:
    """Combine XGBoost and BERT probabilities via weighted soft voting and compute metrics.

    Weights are normalised so the threshold interpretation is consistent regardless of values passed.
    """
    total  = xgb_weight + bert_weight
    w_xgb  = xgb_weight / total
    w_bert = bert_weight / total

    xgb_proba      = xgb_clf.predict_proba(df)[:, 1]
    bert_proba     = bert_clf.predict_proba(df)[:, 1]
    ensemble_proba = w_xgb * xgb_proba + w_bert * bert_proba

    y_true          = df[label_col].values.astype(int)
    y_pred_ensemble = (ensemble_proba >= threshold).astype(int)

    def compute_metrics(y_true, y_pred, y_prob, model_name):
        return {
            "model": model_name,
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

    results = {
        "ensemble": compute_metrics(
            y_true, y_pred_ensemble, ensemble_proba, "Ensemble"
        ),
        "xgboost": compute_metrics(
            y_true,
            (xgb_proba >= threshold).astype(int),
            xgb_proba,
            "XGBoost",
        ),
        "bert": compute_metrics(
            y_true,
            (bert_proba >= threshold).astype(int),
            bert_proba,
            "DistilBERT",
        ),
        "ensemble_config": {
            "xgb_weight": xgb_weight,
            "bert_weight": bert_weight,
            "threshold": threshold,
        },
    }

    for key in ("ensemble", "xgboost", "bert"):
        m = results[key]
        logger.info(
            "%s  acc=%.4f  f1=%.4f  roc_auc=%.4f",
            m["model"].ljust(12), m["accuracy"], m["f1"], m["roc_auc"],
        )

    return results


def save_confusion_matrix(
    cm: list[list[int]],
    title: str,
    save_path: Path,
) -> None:
    """Render a confusion matrix as a PNG and write it to save_path."""
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=np.array(cm),
        display_labels=["Legitimate", "Phishing"],
    )
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", save_path)


def save_report(results: dict, save_path: Path) -> None:
    """Write the evaluation results dictionary to a JSON file, stripping classification_report strings."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    clean = {}
    for model_key, model_metrics in results.items():
        if not isinstance(model_metrics, dict):
            clean[model_key] = model_metrics
            continue
        clean[model_key] = {
            k: v for k, v in model_metrics.items() if k != "classification_report"
        }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2)

    logger.info("Evaluation report saved to %s", save_path)


def train(
    email_dir: str | Path | None = None,
    sms_path: str | Path | None = None,
    url_path: str | Path | None = None,
    models_dir: str | Path = "models",
    reports_dir: str | Path = "reports",
    xgb_weight: float = 0.5,
    bert_weight: float = 0.5,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    skip_bert: bool = False,
) -> dict:
    """Full training run: load data, train both models, evaluate the ensemble.

    Set skip_bert=True for fast iteration when VRAM is unavailable.
    """
    models_dir  = Path(models_dir)
    reports_dir = Path(reports_dir)

    logger.info("Loading datasets...")
    load_kwargs = {}
    if email_dir is not None:
        load_kwargs["email_dir"] = email_dir
    if sms_path is not None:
        load_kwargs["sms_path"] = sms_path
    if url_path is not None:
        load_kwargs["url_path"] = url_path

    df = load_all(**load_kwargs)

    if len(df) == 0:
        raise RuntimeError(
            "No data was loaded. Make sure the raw data files exist in the "
            "expected directories (data/raw/emails/, data/raw/sms_spam/, "
            "data/raw/urls/).  See data_loader.py for the exact file paths."
        )

    logger.info(
        "Loaded %d samples  (phishing=%d, legitimate=%d)",
        len(df),
        int(df["label"].sum()),
        int((df["label"] == 0).sum()),
    )

    df_train, df_val, df_test = split_dataset(
        df, val_size=val_size, test_size=test_size, random_state=random_state
    )

    logger.info("Training XGBoost classifier...")
    xgb_clf = PhishingXGBClassifier(models_dir=models_dir, random_state=random_state)
    xgb_clf.fit(df_train)
    xgb_clf.save("xgb_phishing")

    if skip_bert:
        logger.info("skip_bert=True, evaluating XGBoost only.")
        xgb_metrics = xgb_clf.evaluate(df_test)
        report_path = reports_dir / "evaluation_xgb_only.json"
        reports_dir.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(
                {k: v for k, v in xgb_metrics.items() if k != "classification_report"},
                f, indent=2,
            )
        save_confusion_matrix(
            xgb_metrics["confusion_matrix"],
            "XGBoost Confusion Matrix",
            reports_dir / "confusion_matrix_xgb.png",
        )
        print(xgb_metrics["classification_report"])
        return {"xgboost": xgb_metrics}

    logger.info("Fine-tuning DistilBERT classifier...")
    bert_clf = PhishingBERTClassifier(models_dir=models_dir)
    bert_clf.fit(df_train, df_val)
    bert_clf.save("bert_phishing")

    logger.info("Evaluating ensemble on test set...")
    results = evaluate_ensemble(
        df_test,
        xgb_clf=xgb_clf,
        bert_clf=bert_clf,
        xgb_weight=xgb_weight,
        bert_weight=bert_weight,
    )

    print("\nEnsemble classification report:")
    print(results["ensemble"]["classification_report"])

    save_report(results, reports_dir / "evaluation.json")

    save_confusion_matrix(
        results["ensemble"]["confusion_matrix"],
        "Ensemble Confusion Matrix (XGBoost + DistilBERT)",
        reports_dir / "confusion_matrix_ensemble.png",
    )
    save_confusion_matrix(
        results["xgboost"]["confusion_matrix"],
        "XGBoost Confusion Matrix",
        reports_dir / "confusion_matrix_xgb.png",
    )
    save_confusion_matrix(
        results["bert"]["confusion_matrix"],
        "DistilBERT Confusion Matrix",
        reports_dir / "confusion_matrix_bert.png",
    )

    return results


if __name__ == "__main__":
    train()
