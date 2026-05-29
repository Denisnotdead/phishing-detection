"""Training and OCR-BERT inference entry points for the image phishing pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.image_pipeline.image_classifier import PhishingImageClassifier
from src.image_pipeline.ocr_extractor import OCRExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


def load_image_dataset(
    image_dir: str | Path,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[list, list, list, list, list, list]:
    """Scan image_dir for phishing/ and legitimate/ sub-folders and build stratified splits.

    Returns six lists: train_paths, train_labels, val_paths, val_labels, test_paths, test_labels.
    """
    image_dir = Path(image_dir)
    phishing_dir   = image_dir / "phishing"
    legitimate_dir = image_dir / "legitimate"

    all_paths, all_labels = [], []

    for directory, label in [(phishing_dir, 1), (legitimate_dir, 0)]:
        if not directory.exists():
            logger.warning("Directory not found, skipping: %s", directory)
            continue
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                all_paths.append(path)
                all_labels.append(label)

    if not all_paths:
        raise FileNotFoundError(
            f"No images found under {image_dir}. "
            "Make sure phishing/ and legitimate/ sub-folders exist with image files."
        )

    logger.info(
        "Found %d images  (phishing=%d, legitimate=%d)",
        len(all_paths),
        sum(all_labels),
        len(all_labels) - sum(all_labels),
    )

    # Split off test set first
    paths_trainval, paths_test, labels_trainval, labels_test = train_test_split(
        all_paths, all_labels,
        test_size=test_size,
        stratify=all_labels,
        random_state=random_state,
    )

    # Split train/val from remainder
    val_fraction = val_size / (1.0 - test_size)
    paths_train, paths_val, labels_train, labels_val = train_test_split(
        paths_trainval, labels_trainval,
        test_size=val_fraction,
        stratify=labels_trainval,
        random_state=random_state,
    )

    logger.info(
        "Split: train=%d  val=%d  test=%d",
        len(paths_train), len(paths_val), len(paths_test),
    )
    return paths_train, labels_train, paths_val, labels_val, paths_test, labels_test


def train_image_model(
    image_dir: str | Path = "data/raw/images",
    models_dir: str | Path = "models",
    reports_dir: str | Path = "reports",
    batch_size: int = 16,
    epochs: int = 20,
    learning_rate: float = 1e-4,
    patience: int = 5,
    random_state: int = 42,
) -> dict:
    """Full EfficientNet training run: load images, train, evaluate, save artefacts."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    paths_train, labels_train, paths_val, labels_val, paths_test, labels_test = (
        load_image_dataset(image_dir, random_state=random_state)
    )

    clf = PhishingImageClassifier(
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        patience=patience,
        models_dir=models_dir,
    )
    clf.fit(paths_train, labels_train, paths_val, labels_val)
    clf.save("efficientnet_phishing")

    logger.info("Evaluating on test split...")
    metrics = clf.evaluate(paths_test, labels_test)

    print("\nEfficientNet test-set classification report:")
    print(metrics["classification_report"])

    report_path = reports_dir / "evaluation_image.json"
    clean = {k: v for k, v in metrics.items() if k != "classification_report"}
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2)
    logger.info("Evaluation report saved to %s", report_path)

    _save_confusion_matrix(
        metrics["confusion_matrix"],
        title="EfficientNet Phishing Screenshot Classifier",
        save_path=reports_dir / "confusion_matrix_image.png",
    )

    return metrics


def classify_with_ocr_bert(
    image_dir: str | Path,
    bert_model_dir: str | Path = "models/bert_phishing",
    reports_dir: str | Path = "reports",
    confidence_threshold: float = 0.3,
    output_csv: bool = True,
) -> pd.DataFrame:
    """OCR-to-BERT inference for classifying screenshots without a trained image model.

    Runs EasyOCR on each image, then classifies the extracted text with DistilBERT.
    Returns a DataFrame with path, ocr_text, phishing_prob, and predicted_label columns.
    """
    # Deferred import to avoid hard dependency when only training path is used
    from src.text_pipeline.bert_classifier import PhishingBERTClassifier

    bert_model_dir = Path(bert_model_dir)
    if not bert_model_dir.exists():
        raise FileNotFoundError(
            f"DistilBERT checkpoint not found at {bert_model_dir}. "
            "Train the text model first with src.text_pipeline.trainer.train()."
        )

    logger.info("Running OCR on images in %s...", image_dir)
    extractor = OCRExtractor(confidence_threshold=confidence_threshold)
    ocr_results = extractor.extract_directory(image_dir)

    if not ocr_results:
        raise RuntimeError(f"No images were found in {image_dir}.")

    logger.info("Extracted text from %d images. Loading DistilBERT...", len(ocr_results))
    bert_clf = PhishingBERTClassifier(models_dir=str(bert_model_dir.parent))
    bert_clf.load(bert_model_dir.name)

    texts = [r["full_text"] for r in ocr_results]

    # Images with no OCR text get blank input; BERT will predict near the prior
    df_input = pd.DataFrame({"text": texts})
    probas = bert_clf.predict_proba(df_input)[:, 1]
    preds  = (probas >= 0.5).astype(int)

    results_df = pd.DataFrame({
        "path":            [r["path"] for r in ocr_results],
        "ocr_text":        texts,
        "n_ocr_words":     [r["n_detections"] for r in ocr_results],
        "mean_confidence": [r["mean_confidence"] for r in ocr_results],
        "phishing_prob":   probas.tolist(),
        "predicted_label": preds.tolist(),
        "ocr_error":       [r.get("error") or "" for r in ocr_results],
    })

    n_phishing = int(results_df["predicted_label"].sum())
    n_legit    = len(results_df) - n_phishing
    logger.info(
        "OCR-BERT results: %d images classified  (phishing=%d, legitimate=%d)",
        len(results_df), n_phishing, n_legit,
    )

    if output_csv:
        reports_dir = Path(reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)
        csv_path = reports_dir / "ocr_bert_predictions.csv"
        results_df.to_csv(csv_path, index=False)
        logger.info("Predictions saved to %s", csv_path)

    return results_df


def _save_confusion_matrix(cm: list[list[int]], title: str, save_path: Path) -> None:
    """Render and save a confusion matrix as a PNG."""
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=np.array(cm),
        display_labels=["Legitimate", "Phishing"],
    )
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Image-based phishing detection trainer / classifier."
    )
    parser.add_argument(
        "--mode",
        choices=["train", "ocr_bert"],
        required=True,
        help=(
            "train: fine-tune EfficientNet on labelled screenshots.  "
            "ocr_bert: extract text via OCR and classify with DistilBERT."
        ),
    )
    parser.add_argument(
        "--images",
        default="data/raw/images",
        help="Image directory.  For train mode, must contain phishing/ and legitimate/ sub-dirs.",
    )
    parser.add_argument(
        "--bert_model",
        default="models/bert_phishing",
        help="Path to saved DistilBERT checkpoint (ocr_bert mode only).",
    )
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--reports_dir", default="reports")
    args = parser.parse_args()

    if args.mode == "train":
        train_image_model(
            image_dir=args.images,
            models_dir=args.models_dir,
            reports_dir=args.reports_dir,
        )
    elif args.mode == "ocr_bert":
        classify_with_ocr_bert(
            image_dir=args.images,
            bert_model_dir=args.bert_model,
            reports_dir=args.reports_dir,
        )
