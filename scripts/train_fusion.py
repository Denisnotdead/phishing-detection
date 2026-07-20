"""Trains the LightGBM fusion meta-classifier via out-of-distribution stacking."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fusion.fusion_classifier import FEATURE_COLS, FusionClassifier
from src.text_pipeline.bert_classifier import PhishingBERTClassifier
from src.text_pipeline.data_loader import load_all
from src.text_pipeline.text_classifier import PhishingXGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# data and models live in the main repo, not in git worktrees
DEFAULT_MODELS_DIR = Path("D:/phishing-detection/models")
DEFAULT_DATA_DIR = Path("D:/phishing-detection/data/raw")

# neutral meta-features for a text-only corpus
TEXT_OCR_CONFIDENCE = 0.5
TEXT_IS_IMAGE_INPUT = 0

# cap DistilBERT fine-tuning size — full-corpus tokenisation exhausts memory
MAX_BERT_TRAIN_SAMPLES = 50_000


def three_way_split(
    df: pd.DataFrame,
    train_size: float = 0.60,
    fusion_size: float = 0.20,
    test_size: float = 0.20,
    random_state: int = 42,
    label_col: str = "label",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 60/20/20 split into (train, fusion, test) DataFrames."""
    if not np.isclose(train_size + fusion_size + test_size, 1.0):
        raise ValueError(
            f"Split sizes must sum to 1.0, got "
            f"{train_size} + {fusion_size} + {test_size} = "
            f"{train_size + fusion_size + test_size}"
        )

    df_remainder, df_test = train_test_split(
        df,
        test_size=test_size,
        stratify=df[label_col],
        random_state=random_state,
    )

    # rescale fusion_size to a fraction of the remaining pool
    fusion_fraction_of_remainder = fusion_size / (train_size + fusion_size)
    df_train, df_fusion = train_test_split(
        df_remainder,
        test_size=fusion_fraction_of_remainder,
        stratify=df_remainder[label_col],
        random_state=random_state,
    )

    logger.info(
        "Split sizes  train=%d (%.0f%%)  fusion=%d (%.0f%%)  test=%d (%.0f%%)",
        len(df_train), 100 * len(df_train) / len(df),
        len(df_fusion), 100 * len(df_fusion) / len(df),
        len(df_test), 100 * len(df_test) / len(df),
    )
    return df_train, df_fusion, df_test


def build_meta_features(
    df: pd.DataFrame,
    xgb_clf: PhishingXGBClassifier,
    bert_clf: PhishingBERTClassifier | None,
    label_col: str = "label",
) -> tuple[pd.DataFrame, np.ndarray]:
    """Run the base classifiers and assemble the meta-feature matrix."""
    logger.info("Generating XGBoost predictions on %d samples...", len(df))
    xgb_prob = xgb_clf.predict_proba(df)[:, 1]

    if bert_clf is not None:
        logger.info("Generating DistilBERT predictions on %d samples...", len(df))
        bert_prob = bert_clf.predict_proba(df)[:, 1]
    else:
        logger.warning("skip_bert=True: filling bert_prob with neutral 0.5.")
        bert_prob = np.full(len(df), 0.5)

    features = pd.DataFrame(
        {
            "xgb_prob": xgb_prob,
            "bert_prob": bert_prob,
            "ocr_confidence": TEXT_OCR_CONFIDENCE,
            "is_image_input": TEXT_IS_IMAGE_INPUT,
        },
        index=df.index,
    )[FEATURE_COLS]

    labels = df[label_col].to_numpy(dtype=int)
    return features, labels


def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute accuracy, F1 and ROC-AUC from phishing probabilities."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def evaluate_on_test(
    fusion: FusionClassifier,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> dict:
    """Score the test set with each base signal and the fusion meta-learner."""
    if not fusion.is_trained:
        raise RuntimeError("Fusion classifier is not trained; cannot evaluate.")

    fusion_prob = fusion._model.predict_proba(X_test[FEATURE_COLS])[:, 1]

    results = {
        "xgboost": _binary_metrics(y_test, X_test["xgb_prob"].to_numpy()),
        "distilbert": _binary_metrics(y_test, X_test["bert_prob"].to_numpy()),
        "fusion": _binary_metrics(y_test, fusion_prob),
    }
    return results


def _print_report(results: dict, fusion: FusionClassifier) -> None:
    """Pretty-print the test metrics and the learned meta-feature importances."""
    print("\n" + "=" * 62)
    print("  FUSION META-CLASSIFIER  --  TEST SET EVALUATION")
    print("=" * 62)
    print(f"  {'Model':<14}{'Accuracy':>12}{'F1':>12}{'ROC-AUC':>12}")
    print("  " + "-" * 58)
    for name, key in [("XGBoost", "xgboost"), ("DistilBERT", "distilbert"), ("Fusion (LGBM)", "fusion")]:
        m = results[key]
        print(f"  {name:<14}{m['accuracy']:>12.4f}{m['f1']:>12.4f}{m['roc_auc']:>12.4f}")
    print("=" * 62)

    importances = fusion._model.feature_importances_
    print("\n  Learned meta-feature importances (LightGBM split gain count):")
    for name, imp in sorted(
        zip(FEATURE_COLS, importances), key=lambda x: x[1], reverse=True
    ):
        print(f"    {name:<18}{int(imp):>8}")
    print()


def train_fusion(
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    random_state: int = 42,
    skip_bert: bool = False,
) -> dict:
    """Full stacking run: load data, fit base models, train and evaluate fusion."""
    data_dir = Path(data_dir)
    models_dir = Path(models_dir)

    logger.info("Loading datasets from %s ...", data_dir)
    df = load_all(
        email_dir=data_dir / "emails",
        sms_path=data_dir / "sms_spam" / "SMSSpamCollection",
        url_path=data_dir / "urls" / "malicious_phish.csv",
    )
    if len(df) == 0:
        raise RuntimeError(
            f"No data was loaded from {data_dir}. Check that the raw datasets "
            "exist (emails/, sms_spam/, urls/)."
        )
    logger.info(
        "Loaded %d samples  (phishing=%d, legitimate=%d)",
        len(df), int(df["label"].sum()), int((df["label"] == 0).sum()),
    )

    df_train, df_fusion, df_test = three_way_split(df, random_state=random_state)

    # --- base models: fit on the train split only (60%) ---
    logger.info("Training XGBoost base classifier on the train split...")
    xgb_clf = PhishingXGBClassifier(models_dir=models_dir, random_state=random_state)
    xgb_clf.fit(df_train)

    bert_clf: PhishingBERTClassifier | None = None
    if not skip_bert:
        # small validation slice for early stopping
        df_bert_train, df_bert_val = train_test_split(
            df_train,
            test_size=0.10,
            stratify=df_train["label"],
            random_state=random_state,
        )

        # cap the BERT training set (full split hangs during tokenisation)
        if len(df_bert_train) > MAX_BERT_TRAIN_SAMPLES:
            df_bert_train, _ = train_test_split(
                df_bert_train,
                train_size=MAX_BERT_TRAIN_SAMPLES,
                stratify=df_bert_train["label"],
                random_state=random_state,
            )

        logger.info(
            "Fine-tuning DistilBERT on %d samples (val=%d).",
            len(df_bert_train), len(df_bert_val),
        )
        bert_clf = PhishingBERTClassifier(models_dir=models_dir)
        bert_clf.fit(df_bert_train, df_bert_val)

    # --- Meta-learner: train on out-of-distribution fusion-split predictions
    logger.info("Building meta-features on the fusion split...")
    X_fusion, y_fusion = build_meta_features(df_fusion, xgb_clf, bert_clf)

    logger.info("Training LightGBM fusion meta-classifier...")
    fusion = FusionClassifier(models_dir=models_dir)
    fusion.fit(X_fusion, y_fusion)
    save_path = fusion.save("fusion_classifier")
    logger.info("Fusion meta-classifier saved to %s", save_path)

    # --- Evaluation on the untouched test split ---------------------------
    logger.info("Building meta-features on the test split...")
    X_test, y_test = build_meta_features(df_test, xgb_clf, bert_clf)
    results = evaluate_on_test(fusion, X_test, y_test)

    _print_report(results, fusion)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the LightGBM fusion meta-classifier via stacking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/train_fusion.py\n"
            "  python scripts/train_fusion.py --skip-bert\n"
            "  python scripts/train_fusion.py --models-dir D:/phishing-detection/models\n"
        ),
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help=f"Root directory of the raw datasets. Default: {DEFAULT_DATA_DIR}",
    )
    parser.add_argument(
        "--models-dir",
        default=str(DEFAULT_MODELS_DIR),
        help=f"Directory to save fusion_classifier.pkl. Default: {DEFAULT_MODELS_DIR}",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for the stratified splits and base models. Default: 42",
    )
    parser.add_argument(
        "--skip-bert",
        action="store_true",
        help=(
            "Skip DistilBERT fine-tuning and fill bert_prob with 0.5. "
            "Useful for a fast CPU-only smoke test of the stacking plumbing."
        ),
    )
    args = parser.parse_args()

    train_fusion(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        random_state=args.random_state,
        skip_bert=args.skip_bert,
    )


if __name__ == "__main__":
    main()
