"""
scan_image.py
Runs phishing detection on one image or a directory of images.

Three modes via --mode:
    fusion (default) — XGBoost + DistilBERT + LightGBM fusion
    bert             — OCR + DistilBERT only
    xgb              — OCR + XGBoost only

Usage:
    python scripts/scan_image.py                              # fusion, all images
    python scripts/scan_image.py screenshot.png               # fusion, one image
    python scripts/scan_image.py --mode bert                  # BERT only, all images
    python scripts/scan_image.py --mode xgb screenshot.png    # XGBoost only, one image
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fusion.fusion_classifier import confidence_tier
from src.fusion.pipeline import PhishingDetectionPipeline, remove_ui_noise
from src.image_pipeline.ocr_extractor import OCRExtractor
from src.text_pipeline.data_loader import extract_urls

DEFAULT_MODELS_DIR = Path("D:/phishing-detection/models")
DEFAULT_SCAN_DIR   = PROJECT_ROOT / "data" / "test_images"
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


def _ocr_and_clean(image_path: Path, ocr: OCRExtractor) -> tuple[str, str, int, float]:
    """Run OCR on one image and return (extracted_text, cleaned_text, n_words, ocr_confidence)."""
    result         = ocr.extract(image_path)
    extracted_text = result["full_text"]
    cleaned_text   = remove_ui_noise(extracted_text)
    return extracted_text, cleaned_text, result["n_detections"], result["mean_confidence"]


def _text_to_df(text: str) -> pd.DataFrame:
    """Wrap a plain text string in a DataFrame suitable for both classifiers."""
    return pd.DataFrame({
        "text":    [text],
        "urls":    [extract_urls(text)],
        "type":    ["email"],
        "sender":  [""],
        "subject": [""],
    })


def _run_bert_on_image(
    image_path: Path,
    ocr: OCRExtractor,
    bert,
) -> dict:
    """OCR → clean → DistilBERT for one image."""
    extracted, cleaned, n_words, ocr_conf = _ocr_and_clean(image_path, ocr)
    bert_prob = float(bert.predict_proba(_text_to_df(cleaned))[0, 1])
    label     = "PHISHING" if bert_prob >= 0.5 else "LEGITIMATE"

    return {
        "mode":           "bert",
        "image_path":     str(image_path),
        "extracted_text": extracted,
        "cleaned_text":   cleaned,
        "n_ocr_words":    n_words,
        "score":          bert_prob,
        "label":          label,
        "confidence":     confidence_tier(bert_prob),
        "signals": {
            "xgb_prob":       None,
            "bert_prob":      bert_prob,
            "ocr_confidence": ocr_conf,
            "is_image_input": 1,
        },
    }


def _run_xgb_on_image(
    image_path: Path,
    ocr: OCRExtractor,
    xgb_clf,
) -> dict:
    """OCR → clean → XGBoost for one image."""
    extracted, cleaned, n_words, ocr_conf = _ocr_and_clean(image_path, ocr)
    xgb_prob = float(xgb_clf.predict_proba(_text_to_df(cleaned))[0, 1])
    label    = "PHISHING" if xgb_prob >= 0.5 else "LEGITIMATE"

    return {
        "mode":           "xgb",
        "image_path":     str(image_path),
        "extracted_text": extracted,
        "cleaned_text":   cleaned,
        "n_ocr_words":    n_words,
        "score":          xgb_prob,
        "label":          label,
        "confidence":     confidence_tier(xgb_prob),
        "signals": {
            "xgb_prob":       xgb_prob,
            "bert_prob":      None,
            "ocr_confidence": ocr_conf,
            "is_image_input": 1,
        },
    }


def _make_runner(mode: str, models_dir: Path):
    """Load models for the given mode and return a callable that processes one image path.

    Models are loaded once so scan_directory() can reuse them across all images.
    """
    if mode == "fusion":
        pipeline = PhishingDetectionPipeline(models_dir=models_dir)
        def run_fusion(image_path: Path) -> dict:
            result = pipeline.analyze(image_path=image_path)
            result["mode"] = "fusion"
            return result
        return run_fusion

    if mode == "bert":
        from src.text_pipeline.bert_classifier import PhishingBERTClassifier
        ocr  = OCRExtractor()
        bert = PhishingBERTClassifier(models_dir=str(models_dir)).load("bert_phishing")
        def run_bert(image_path: Path) -> dict:
            return _run_bert_on_image(image_path, ocr, bert)
        return run_bert

    if mode == "xgb":
        from src.text_pipeline.text_classifier import PhishingXGBClassifier
        ocr = OCRExtractor()
        xgb = PhishingXGBClassifier(models_dir=models_dir).load("xgb_phishing")
        def run_xgb(image_path: Path) -> dict:
            return _run_xgb_on_image(image_path, ocr, xgb)
        return run_xgb

    raise ValueError(f"Unknown mode '{mode}'. Choose from: fusion, bert, xgb.")


def print_result(result: dict) -> None:
    """Print a single scan result, adapting the scores section to the active mode."""
    width = 60
    mode  = result.get("mode", "fusion")

    print()
    print("=" * width)
    print("  PHISHING SCAN RESULT")
    if result.get("image_path"):
        print(f"  {result['image_path']}")
    print("=" * width)

    if result.get("extracted_text") is not None:
        ocr_conf = result["signals"]["ocr_confidence"]
        print(
            f"\n  Extracted text ({result['n_ocr_words']} words, "
            f"OCR confidence: {ocr_conf:.0%})"
        )
        print("  " + "-" * (width - 2))
        raw = result["extracted_text"].strip()
        if not raw:
            print("  [no text detected]")
        else:
            for ln in _wrap(raw, width=56):
                print(f"  {ln}")

    print()
    print("  " + "-" * (width - 2))

    signals = result.get("signals", {})

    if mode == "bert":
        print(f"  DistilBERT score     : {signals['bert_prob']:.1%}")

    elif mode == "xgb":
        print(f"  XGBoost score        : {signals['xgb_prob']:.1%}")

    else:
        # fusion mode — show all three scores
        if signals.get("xgb_prob") is not None:
            print(f"  XGBoost probability  : {signals['xgb_prob']:.1%}")
        if signals.get("bert_prob") is not None:
            print(f"  DistilBERT prob      : {signals['bert_prob']:.1%}")
        print(f"  Fusion score         : {result['score']:.1%}")
        print()
        if result.get("explanation"):
            print(f"  {result['explanation']}")

    print(f"  Confidence           : {result['confidence']}")
    print()

    if result["label"] == "PHISHING":
        print("  VERDICT  >>>  PHISHING  <<<")
    else:
        print("  VERDICT  >>>  LEGITIMATE  <<<")

    print("=" * width)
    print()


def print_summary(results: list[dict]) -> None:
    """Print a batch summary table after scanning a directory."""
    width      = 60
    phishing   = [r for r in results if r["label"] == "PHISHING"]
    legitimate = [r for r in results if r["label"] == "LEGITIMATE"]
    errors     = [r for r in results if r.get("error")]

    print()
    print("=" * width)
    print("  BATCH SCAN SUMMARY")
    print("=" * width)
    print(f"  Total scanned  : {len(results)}")
    print(f"  Phishing       : {len(phishing)}")
    print(f"  Legitimate     : {len(legitimate)}")
    if errors:
        print(f"  Errors         : {len(errors)}")
    print()

    if phishing:
        print("  Flagged as phishing:")
        for r in phishing:
            name  = Path(r["image_path"]).name if r.get("image_path") else "unknown"
            score = r["score"]
            conf  = r["confidence"]
            print(f"    {name:<35}  {score:.1%}  [{conf}]")

    print("=" * width)
    print()


def _wrap(text: str, width: int = 56) -> list[str]:
    """Word-wrap text to fit within a given character width."""
    words, line, lines = text.split(), [], []
    for word in words:
        if sum(len(w) + 1 for w in line) + len(word) > width:
            lines.append(" ".join(line))
            line = [word]
        else:
            line.append(word)
    if line:
        lines.append(" ".join(line))
    return lines


def scan(
    image_path: str | Path,
    mode: str = "fusion",
    models_dir: str | Path = DEFAULT_MODELS_DIR,
) -> dict:
    """Run the chosen model on a single image and return the result dict."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    run = _make_runner(mode, Path(models_dir))
    return run(image_path)


def scan_directory(
    directory: str | Path = DEFAULT_SCAN_DIR,
    mode: str = "fusion",
    models_dir: str | Path = DEFAULT_MODELS_DIR,
) -> list[dict]:
    """Scan every image in a directory, loading models only once."""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(
            f"Scan directory not found: {directory}\n"
            "Create it and add images, or pass an explicit image path."
        )

    image_paths = sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not image_paths:
        raise FileNotFoundError(
            f"No images found in {directory}\n"
            f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    print(f"\nFound {len(image_paths)} image(s) in {directory}  (mode: {mode})")

    run = _make_runner(mode, Path(models_dir))

    results = []
    for i, path in enumerate(image_paths, start=1):
        print(f"\nProcessing {i}/{len(image_paths)}: {path.name}")
        try:
            result = run(path)
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            result = {
                "mode":           mode,
                "image_path":     str(path),
                "extracted_text": None,
                "cleaned_text":   None,
                "n_ocr_words":    0,
                "score":          0.0,
                "label":          "ERROR",
                "confidence":     "N/A",
                "signals": {
                    "xgb_prob": None,
                    "bert_prob": None,
                    "ocr_confidence": 0.0,
                    "is_image_input": 1,
                },
                "error": str(exc),
            }
        print_result(result)
        results.append(result)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan one image or a directory for phishing content.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/scan_image.py\n"
            "  python scripts/scan_image.py screenshot.png\n"
            "  python scripts/scan_image.py --mode bert\n"
            "  python scripts/scan_image.py --mode xgb screenshot.png\n"
            "  python scripts/scan_image.py --mode fusion --models-dir D:/phishing-detection/models\n"
        ),
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=None,
        help=(
            "Path to a single image file. "
            f"If omitted, all images in {DEFAULT_SCAN_DIR} are scanned."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["fusion", "bert", "xgb"],
        default="fusion",
        help=(
            "fusion: XGBoost + DistilBERT + LightGBM meta-learner (default).  "
            "bert: DistilBERT only.  "
            "xgb: XGBoost only."
        ),
    )
    parser.add_argument(
        "--models-dir",
        default=str(DEFAULT_MODELS_DIR),
        help=f"Root directory containing all model checkpoints. Default: {DEFAULT_MODELS_DIR}",
    )
    args = parser.parse_args()

    try:
        if args.image is not None:
            result = scan(args.image, mode=args.mode, models_dir=args.models_dir)
            print_result(result)
            if result["label"] == "PHISHING":
                sys.exit(1)
        else:
            results = scan_directory(mode=args.mode, models_dir=args.models_dir)
            print_summary(results)
            if any(r["label"] == "PHISHING" for r in results):
                sys.exit(1)

    except FileNotFoundError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(2)
    except Exception as exc:
        print(f"\nUnexpected error: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
