"""Optuna (TPE) hyperparameter tuning for the XGBoost phishing classifier."""

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
import optuna
import pandas as pd
import xgboost as xgb
from optuna.samplers import TPESampler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.text_pipeline.data_loader import load_all
from src.text_pipeline.feature_extractor import (
    StructuralFeatureTransformer,
    TextStatFeatureTransformer,
    URLFeatureTransformer,
)

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

# stratified subsample keeps 50 trials tractable
DEFAULT_SAMPLE_SIZE = 100_000
DEFAULT_N_TRIALS = 50
RANDOM_STATE = 42

# XGBoost defaults used as the comparison baseline
DEFAULT_XGB_PARAMS = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "min_child_weight": 1,
    "gamma": 0.0,
}


def stratified_sample(
    df: pd.DataFrame, sample_size: int, random_state: int, label_col: str = "label"
) -> pd.DataFrame:
    """Return a stratified subsample of df preserving the label ratio."""
    if len(df) <= sample_size:
        return df.reset_index(drop=True)

    df_sample, _ = train_test_split(
        df,
        train_size=sample_size,
        stratify=df[label_col],
        random_state=random_state,
    )
    return df_sample.reset_index(drop=True)


def build_scaled_features(
    df_train: pd.DataFrame, df_val: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Extract URL + text-stat + structural features and standardise them."""
    url_t = URLFeatureTransformer()
    text_t = TextStatFeatureTransformer()
    struct_t = StructuralFeatureTransformer()

    def extract(df: pd.DataFrame) -> np.ndarray:
        return np.hstack(
            [url_t.transform(df), text_t.transform(df), struct_t.transform(df)]
        )

    logger.info("Extracting features for %d train rows...", len(df_train))
    X_train_raw = extract(df_train)
    logger.info("Extracting features for %d validation rows...", len(df_val))
    X_val_raw = extract(df_val)

    scaler = StandardScaler().fit(X_train_raw)
    return scaler.transform(X_train_raw), scaler.transform(X_val_raw)


def make_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scale_pos_weight: float,
):
    """Build the Optuna objective: sample params, train XGBoost, return val F1."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        }

        model = xgb.XGBClassifier(
            **params,
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",  # fast on large data
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        return float(f1_score(y_val, y_pred, zero_division=0))

    return objective


def evaluate_params(
    params: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scale_pos_weight: float,
) -> dict:
    """Train XGBoost with *params* and return accuracy, F1 and ROC-AUC on val."""
    model = xgb.XGBClassifier(
        **params,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "f1": float(f1_score(y_val, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_val, y_prob)),
    }


def save_plots(study: optuna.Study, reports_dir: Path) -> None:
    """Render the three Optuna diagnostic plots to PNG files."""
    from optuna.visualization.matplotlib import (
        plot_optimization_history,
        plot_parallel_coordinate,
        plot_param_importances,
    )

    plots = [
        ("optuna_optimization_history.png", plot_optimization_history,
         "Optimization History"),
        ("optuna_param_importances.png", plot_param_importances,
         "Hyperparameter Importances"),
        ("optuna_parallel_coordinate.png", plot_parallel_coordinate,
         "Parallel Coordinate (F1)"),
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
        "model": "XGBoost",
        "sampler": "TPESampler",
        "objective": "validation_f1",
        "n_trials": len(study.trials),
        "best_value": study.best_value,
        "best_params": study.best_params,
        "comparison": {
            "default_params": DEFAULT_XGB_PARAMS,
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
    print("  OPTUNA HYPERPARAMETER TUNING  --  BEST RESULT")
    print("=" * 60)
    print(f"  Best validation F1: {study.best_value:.4f}")
    print("  Best parameters:")
    for k, v in study.best_params.items():
        shown = f"{v:.5f}" if isinstance(v, float) else str(v)
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
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    n_trials: int = DEFAULT_N_TRIALS,
    random_state: int = RANDOM_STATE,
) -> optuna.Study:
    """Full tuning run: load, sample, split, optimise, evaluate, and persist."""
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

    df = stratified_sample(df, sample_size, random_state)
    logger.info(
        "Tuning on %d samples  (phishing=%d, legitimate=%d)",
        len(df), int(df["label"].sum()), int((df["label"] == 0).sum()),
    )

    df_train, df_val = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=random_state
    )
    y_train = df_train["label"].to_numpy(dtype=int)
    y_val = df_val["label"].to_numpy(dtype=int)

    X_train, X_val = build_scaled_features(df_train, df_val)

    # handle class imbalance like PhishingXGBClassifier
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)

    logger.info("Starting Optuna study: %d TPE trials...", n_trials)
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=random_state),
        study_name="xgb_phishing_tuning",
    )
    objective = make_objective(X_train, y_train, X_val, y_val, scale_pos_weight)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # final comparison: default vs tuned
    logger.info("Evaluating default parameters for comparison...")
    default_metrics = evaluate_params(
        DEFAULT_XGB_PARAMS, X_train, y_train, X_val, y_val, scale_pos_weight
    )
    logger.info("Retraining with best parameters...")
    tuned_metrics = evaluate_params(
        study.best_params, X_train, y_train, X_val, y_val, scale_pos_weight
    )

    print_comparison(study, default_metrics, tuned_metrics)
    save_results(study, default_metrics, tuned_metrics, reports_dir / "optuna_results.json")
    save_plots(study, reports_dir)

    return study


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optuna TPE hyperparameter tuning for the XGBoost classifier.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/optuna_tune.py\n"
            "  python scripts/optuna_tune.py --n-trials 50 --sample-size 100000\n"
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
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Stratified sample size for tuning. Default: {DEFAULT_SAMPLE_SIZE}",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=DEFAULT_N_TRIALS,
        help=f"Number of Optuna trials. Default: {DEFAULT_N_TRIALS}",
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
        n_trials=args.n_trials,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
