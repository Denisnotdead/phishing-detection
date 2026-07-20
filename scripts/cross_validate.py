"""5-fold stratified cross-validation for the XGBoost phishing classifier."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.text_pipeline.data_loader import load_all
from src.text_pipeline.text_classifier import PhishingXGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# data lives in the main repo, not in git worktrees
DEFAULT_DATA_DIR = Path("D:/phishing-detection/data/raw")
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"

# metrics per fold, in display order
METRIC_NAMES = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def _fold_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute the five reported metrics for a single fold."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def cross_validate(
    df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
    label_col: str = "label",
) -> list[dict]:
    """Run stratified k-fold CV on XGBoost and return per-fold metric dicts."""
    y = df[label_col].to_numpy(dtype=int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_results: list[dict] = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, y), start=1):
        # keep transformers index-agnostic
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_val = df.iloc[val_idx].reset_index(drop=True)

        logger.info(
            "Fold %d/%d  train=%d  val=%d", fold, n_splits, len(df_train), len(df_val)
        )

        clf = PhishingXGBClassifier(random_state=random_state)
        clf.fit(df_train, label_col=label_col)

        y_true = df_val[label_col].to_numpy(dtype=int)
        y_pred = clf.predict(df_val)
        y_prob = clf.predict_proba(df_val)[:, 1]

        metrics = _fold_metrics(y_true, y_pred, y_prob)
        metrics["fold"] = fold
        fold_results.append(metrics)

        logger.info(
            "Fold %d  acc=%.4f  prec=%.4f  rec=%.4f  f1=%.4f  roc_auc=%.4f",
            fold, metrics["accuracy"], metrics["precision"],
            metrics["recall"], metrics["f1"], metrics["roc_auc"],
        )

    return fold_results


def summarise(fold_results: list[dict]) -> dict:
    """Return {metric: {"mean": .., "std": ..}} across folds for each metric."""
    summary = {}
    for name in METRIC_NAMES:
        values = [f[name] for f in fold_results]
        summary[name] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    return summary


def print_summary(summary: dict, n_splits: int) -> None:
    """Print the mean +/- std table to stdout."""
    print("\n" + "=" * 50)
    print(f"  XGBOOST  --  {n_splits}-FOLD CROSS-VALIDATION SUMMARY")
    print("=" * 50)
    print(f"  {'Metric':<12}{'Mean +/- Std':>22}")
    print("  " + "-" * 46)
    for name in METRIC_NAMES:
        stats = summary[name]
        print(f"  {name:<12}{stats['mean']:>12.4f}  +/- {stats['std']:.4f}")
    print("=" * 50 + "\n")


def save_results(
    fold_results: list[dict],
    summary: dict,
    n_splits: int,
    save_path: Path,
) -> None:
    """Write per-fold and summary metrics to a JSON file."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "XGBoost",
        "n_splits": n_splits,
        "metrics": METRIC_NAMES,
        "folds": fold_results,
        "summary": summary,
        # BERT CV omitted due to cost; single-split results reported instead
        "note": (
            "BERT CV omitted due to computational cost; single split results "
            "reported instead."
        ),
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Cross-validation results saved to %s", save_path)


def save_boxplot(fold_results: list[dict], save_path: Path) -> None:
    """Render a box plot of each metric's distribution across folds."""
    data = [[f[name] for f in fold_results] for name in METRIC_NAMES]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, labels=METRIC_NAMES, showmeans=True)
    # overlay individual fold scores
    for i, scores in enumerate(data, start=1):
        jitter = np.random.RandomState(0).normal(0, 0.04, size=len(scores))
        ax.scatter(np.full(len(scores), i) + jitter, scores, alpha=0.6, s=20, color="tab:blue")

    ax.set_ylabel("Score")
    ax.set_title(f"XGBoost {len(fold_results)}-Fold Cross-Validation Score Distribution")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Box plot saved to %s", save_path)


def run(
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """Full CV run: load data, cross-validate, summarise, and write artefacts."""
    data_dir = Path(data_dir)
    reports_dir = Path(reports_dir)

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

    fold_results = cross_validate(df, n_splits=n_splits, random_state=random_state)
    summary = summarise(fold_results)

    print_summary(summary, n_splits)
    save_results(fold_results, summary, n_splits, reports_dir / "cv_results.json")
    save_boxplot(fold_results, reports_dir / "cv_boxplot.png")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="5-fold stratified cross-validation for the XGBoost classifier.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/cross_validate.py\n"
            "  python scripts/cross_validate.py --n-splits 5 --random-state 42\n"
        ),
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help=f"Root directory of the raw datasets. Default: {DEFAULT_DATA_DIR}",
    )
    parser.add_argument(
        "--reports-dir",
        default=str(DEFAULT_REPORTS_DIR),
        help=f"Directory for cv_results.json and cv_boxplot.png. Default: {DEFAULT_REPORTS_DIR}",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of stratified folds. Default: 5",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for the fold shuffling and the model. Default: 42",
    )
    args = parser.parse_args()

    run(
        data_dir=args.data_dir,
        reports_dir=args.reports_dir,
        n_splits=args.n_splits,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
