"""Optuna (TPE) hyperparameter tuning for the DistilBERT phishing classifier."""

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
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split

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
optuna.logging.set_verbosity(optuna.logging.WARNING)  # keep our own log readable

# data lives in the main repo, not in git worktrees
DEFAULT_DATA_DIR = Path("D:/phishing-detection/data/raw")
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"

# small fixed subsample keeps each trial tractable
DEFAULT_TRAIN_SIZE = 20_000
DEFAULT_VAL_SIZE = 5_000
DEFAULT_N_TRIALS = 12
DEFAULT_EPOCHS = 3
RANDOM_STATE = 42

# DistilBERT defaults used as the comparison baseline
DEFAULT_BERT_PARAMS = {
    "learning_rate": 2e-5,
    "batch_size": 16,
    "max_length": 128,
    "warmup_ratio": 0.1,
}


def sample_train_val(
    df: pd.DataFrame,
    train_size: int,
    val_size: int,
    random_state: int,
    label_col: str = "label",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Draw disjoint stratified train and validation subsamples from df."""
    total = train_size + val_size
    if len(df) < total:
        # scale both splits down to fit available data
        scale = len(df) / total
        train_size = int(train_size * scale)
        val_size = len(df) - train_size

    df_pool, _ = (df, None) if len(df) == total else train_test_split(
        df, train_size=total, stratify=df[label_col], random_state=random_state
    )
    df_train, df_val = train_test_split(
        df_pool,
        test_size=val_size / total,
        stratify=df_pool[label_col],
        random_state=random_state,
    )
    return df_train.reset_index(drop=True), df_val.reset_index(drop=True)


def make_objective(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    epochs: int,
):
    """Build the Optuna objective: sample params, fine-tune DistilBERT, return val F1."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
            "max_length": trial.suggest_categorical("max_length", [64, 128, 256]),
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
        }

        logger.info(
            "Trial %d starting  lr=%.2e  batch=%d  max_len=%d  warmup=%.3f",
            trial.number, params["learning_rate"], params["batch_size"],
            params["max_length"], params["warmup_ratio"],
        )
        start = time.perf_counter()

        clf = PhishingBERTClassifier(
            epochs=epochs,
            patience=epochs,  # disable early stopping
            models_dir="models",
            **params,
        )
        clf.fit(df_train, df_val)
        metrics = clf.evaluate(df_val)

        elapsed = time.perf_counter() - start
        logger.info(
            "Trial %d done in %.1fs  val_f1=%.4f", trial.number, elapsed, metrics["f1"]
        )
        return float(metrics["f1"])

    return objective


def evaluate_params(
    params: dict,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    epochs: int,
) -> dict:
    """Fine-tune DistilBERT with *params* and return accuracy, F1 and ROC-AUC on val."""
    clf = PhishingBERTClassifier(
        epochs=epochs,
        patience=epochs,
        models_dir="models",
        **params,
    )
    clf.fit(df_train, df_val)
    metrics = clf.evaluate(df_val)
    return {
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
    }


def save_plots(study: optuna.Study, reports_dir: Path) -> None:
    """Render the two Optuna diagnostic plots to PNG files (each guarded)."""
    from optuna.visualization.matplotlib import (
        plot_optimization_history,
        plot_param_importances,
    )

    plots = [
        ("optuna_optimization_history_bert.png", plot_optimization_history,
         "Optimization History"),
        ("optuna_param_importances_bert.png", plot_param_importances,
         "Hyperparameter Importances"),
    ]

    for filename, plot_fn, title in plots:
        try:
            ax = plot_fn(study)
            fig = ax.figure
            fig.set_size_inches(10, 6)
            ax.set_title(title)
            fig.tight_layout()
            save_path = reports_dir / filename
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
            logger.info("Saved %s", save_path)
        except Exception as exc:  # noqa: BLE001 - diagnostics shouldn't crash the run
            logger.warning("Could not generate %s: %s", filename, exc)


def save_results(
    study: optuna.Study,
    default_metrics: dict,
    tuned_metrics: dict,
    save_path: Path,
) -> None:
    """Write best params, every trial, and the default-vs-tuned comparison to JSON."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    trials = [
        {
            "number": t.number,
            "value": t.value,
            "state": t.state.name,
            "params": t.params,
        }
        for t in study.trials
    ]

    payload = {
        "model": "DistilBERT",
        "sampler": "TPESampler",
        "objective": "validation_f1",
        "n_trials": len(study.trials),
        "best_value": study.best_value,
        "best_params": study.best_params,
        "comparison": {
            "default_params": DEFAULT_BERT_PARAMS,
            "default_metrics": default_metrics,
            "tuned_metrics": tuned_metrics,
        },
        "trials": trials,
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Optuna results saved to %s", save_path)


def print_comparison(
    study: optuna.Study, default_metrics: dict, tuned_metrics: dict
) -> None:
    """Print the best params and a default-vs-tuned metric comparison table."""
    print("\n" + "=" * 60)
    print("  OPTUNA DISTILBERT TUNING  --  BEST RESULT")
    print("=" * 60)
    print(f"  Best validation F1: {study.best_value:.4f}")
    print("  Best parameters:")
    for k, v in study.best_params.items():
        shown = f"{v:.6g}" if isinstance(v, float) else str(v)
        print(f"    {k:<20}{shown}")

    print("\n  Default vs. tuned (validation set):")
    print(f"  {'Metric':<12}{'Default':>12}{'Tuned':>12}{'Delta':>12}")
    print("  " + "-" * 48)
    for name in ("accuracy", "f1", "roc_auc"):
        d, t = default_metrics[name], tuned_metrics[name]
        print(f"  {name:<12}{d:>12.4f}{t:>12.4f}{t - d:>+12.4f}")
    print("=" * 60 + "\n")


def run(
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
    train_size: int = DEFAULT_TRAIN_SIZE,
    val_size: int = DEFAULT_VAL_SIZE,
    n_trials: int = DEFAULT_N_TRIALS,
    epochs: int = DEFAULT_EPOCHS,
    random_state: int = RANDOM_STATE,
) -> optuna.Study:
    """Full tuning run: load, sample, optimise, evaluate, and persist."""
    data_dir = Path(data_dir)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

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

    df_train, df_val = sample_train_val(df, train_size, val_size, random_state)
    logger.info(
        "Tuning on train=%d  val=%d  (train phishing=%d, legitimate=%d)",
        len(df_train), len(df_val),
        int(df_train["label"].sum()), int((df_train["label"] == 0).sum()),
    )

    logger.info("Starting Optuna study: %d TPE trials, %d epochs each...", n_trials, epochs)
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=random_state),
        study_name="distilbert_phishing_tuning",
    )
    objective = make_objective(df_train, df_val, epochs)
    overall_start = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info(
        "Study finished in %.1f min.", (time.perf_counter() - overall_start) / 60
    )

    # final comparison: default vs tuned
    logger.info("Evaluating default parameters for comparison...")
    default_metrics = evaluate_params(DEFAULT_BERT_PARAMS, df_train, df_val, epochs)
    logger.info("Retraining with best parameters...")
    tuned_metrics = evaluate_params(study.best_params, df_train, df_val, epochs)

    print_comparison(study, default_metrics, tuned_metrics)
    save_results(
        study, default_metrics, tuned_metrics, reports_dir / "optuna_results_bert.json"
    )
    save_plots(study, reports_dir)

    return study


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optuna TPE hyperparameter tuning for the DistilBERT classifier.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/optuna_tune_bert.py\n"
            "  python scripts/optuna_tune_bert.py --n-trials 12 --epochs 3\n"
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
        help=f"Directory for results and plots. Default: {DEFAULT_REPORTS_DIR}",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=DEFAULT_TRAIN_SIZE,
        help=f"Stratified training subsample size. Default: {DEFAULT_TRAIN_SIZE}",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=DEFAULT_VAL_SIZE,
        help=f"Stratified validation subsample size. Default: {DEFAULT_VAL_SIZE}",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=DEFAULT_N_TRIALS,
        help=f"Number of Optuna trials. Default: {DEFAULT_N_TRIALS}",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Fine-tuning epochs per trial. Default: {DEFAULT_EPOCHS}",
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
        train_size=args.train_size,
        val_size=args.val_size,
        n_trials=args.n_trials,
        epochs=args.epochs,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
