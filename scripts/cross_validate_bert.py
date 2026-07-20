"""5-fold stratified cross-validation for the DistilBERT phishing classifier."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.text_pipeline.bert_classifier import PhishingBERTClassifier
from src.text_pipeline.data_loader import load_all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# data lives in the main repo, not in git worktrees
DEFAULT_DATA_DIR = Path("D:/phishing-detection/data/raw")
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"

# stratified subsample keeps transformer CV feasible
DEFAULT_SAMPLE_SIZE = 50_000
DEFAULT_N_SPLITS = 5
DEFAULT_MAX_EPOCHS = 5
DEFAULT_PATIENCE = 2
RANDOM_STATE = 42

# metrics per fold, in display order
METRIC_NAMES = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def stratified_sample(
    df: pd.DataFrame, sample_size: int, random_state: int, label_col: str = "label"
) -> pd.DataFrame:
    """Return a stratified subsample preserving the label ratio (or all rows if smaller)."""
    if len(df) <= sample_size:
        return df.reset_index(drop=True)
    df_sample, _ = train_test_split(
        df,
        train_size=sample_size,
        stratify=df[label_col],
        random_state=random_state,
    )
    return df_sample.reset_index(drop=True)


def cross_validate(
    df: pd.DataFrame,
    n_splits: int = DEFAULT_N_SPLITS,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    random_state: int = RANDOM_STATE,
    label_col: str = "label",
) -> list[dict]:
    """Run stratified k-fold CV on DistilBERT and return per-fold metric dicts."""
    y = df[label_col].to_numpy(dtype=int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_results: list[dict] = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, y), start=1):
        df_train_full = df.iloc[train_idx].reset_index(drop=True)
        df_holdout = df.iloc[val_idx].reset_index(drop=True)

        # small validation slice for early stopping
        df_fit, df_earlystop = train_test_split(
            df_train_full,
            test_size=0.10,
            stratify=df_train_full[label_col],
            random_state=random_state,
        )

        logger.info(
            "=== Fold %d/%d  fit=%d  earlystop=%d  holdout=%d ===",
            fold, n_splits, len(df_fit), len(df_earlystop), len(df_holdout),
        )

        start = time.perf_counter()
        clf = PhishingBERTClassifier(
            epochs=max_epochs, patience=patience, models_dir="models"
        )
        clf.fit(df_fit, df_earlystop, label_col=label_col)

        metrics = clf.evaluate(df_holdout, label_col=label_col)
        elapsed = time.perf_counter() - start

        fold_metrics = {name: metrics[name] for name in METRIC_NAMES}
        fold_metrics["fold"] = fold
        fold_metrics["seconds"] = round(elapsed, 1)
        fold_results.append(fold_metrics)

        logger.info(
            "Fold %d done in %.1fs  acc=%.4f  prec=%.4f  rec=%.4f  f1=%.4f  roc_auc=%.4f",
            fold, elapsed, fold_metrics["accuracy"], fold_metrics["precision"],
            fold_metrics["recall"], fold_metrics["f1"], fold_metrics["roc_auc"],
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
    print(f"  DISTILBERT  --  {n_splits}-FOLD CROSS-VALIDATION SUMMARY")
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
        "model": "DistilBERT",
        "n_splits": n_splits,
        "metrics": METRIC_NAMES,
        "folds": fold_results,
        "summary": summary,
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Cross-validation results saved to %s", save_path)


def save_boxplot(fold_results: list[dict], save_path: Path) -> None:
    """Render a box plot of each metric's distribution across folds."""
    data = [[f[name] for f in fold_results] for name in METRIC_NAMES]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, labels=METRIC_NAMES, showmeans=True)
    for i, scores in enumerate(data, start=1):
        jitter = np.random.RandomState(0).normal(0, 0.04, size=len(scores))
        ax.scatter(np.full(len(scores), i) + jitter, scores, alpha=0.6, s=20, color="tab:orange")

    ax.set_ylabel("Score")
    ax.set_title(f"DistilBERT {len(fold_results)}-Fold Cross-Validation Score Distribution")
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
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    n_splits: int = DEFAULT_N_SPLITS,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Full CV run: load data, subsample, cross-validate, summarise, write artefacts."""
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

    df = stratified_sample(df, sample_size, random_state)
    logger.info(
        "Cross-validating on %d samples  (phishing=%d, legitimate=%d)",
        len(df), int(df["label"].sum()), int((df["label"] == 0).sum()),
    )

    overall_start = time.perf_counter()
    fold_results = cross_validate(
        df,
        n_splits=n_splits,
        max_epochs=max_epochs,
        patience=patience,
        random_state=random_state,
    )
    logger.info(
        "All %d folds finished in %.1f min.",
        n_splits, (time.perf_counter() - overall_start) / 60,
    )

    summary = summarise(fold_results)
    print_summary(summary, n_splits)
    save_results(fold_results, summary, n_splits, reports_dir / "cv_results_bert.json")
    save_boxplot(fold_results, reports_dir / "cv_boxplot_bert.png")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="5-fold stratified cross-validation for the DistilBERT classifier.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/cross_validate_bert.py\n"
            "  python scripts/cross_validate_bert.py --sample-size 50000 --max-epochs 5\n"
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
        help=f"Directory for JSON and box plot. Default: {DEFAULT_REPORTS_DIR}",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Stratified subsample size for CV. Default: {DEFAULT_SAMPLE_SIZE}",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=DEFAULT_N_SPLITS,
        help=f"Number of stratified folds. Default: {DEFAULT_N_SPLITS}",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=DEFAULT_MAX_EPOCHS,
        help=f"Max fine-tuning epochs per fold. Default: {DEFAULT_MAX_EPOCHS}",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help=f"Early-stopping patience per fold. Default: {DEFAULT_PATIENCE}",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help=f"Random seed. Default: {RANDOM_STATE}",
    )
    args = parser.parse_args()

    run(
        data_dir=args.data_dir,
        reports_dir=args.reports_dir,
        sample_size=args.sample_size,
        n_splits=args.n_splits,
        max_epochs=args.max_epochs,
        patience=args.patience,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
