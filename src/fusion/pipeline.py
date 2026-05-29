"""
pipeline.py
Unified phishing detection pipeline that routes text or image input through all sub-models.

All models are loaded lazily on the first analyze() call; missing checkpoints are skipped
so the pipeline degrades gracefully before all models are trained.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from src.fusion.fusion_classifier import FusionClassifier, confidence_tier
from src.text_pipeline.data_loader import extract_urls

logger = logging.getLogger(__name__)

# Email client UI chrome that OCR often picks up from screenshots.
# These phrases carry no phishing signal and are stripped before classification.
UI_NOISE_WORDS = [
    "Compose", "Inbox", "Starred", "Snoozed", "Sent", "Drafts", "More",
    "Labels", "Reply", "Forward", "Search mail", "Personal", "Reports",
    "Project Alpha", "Tof 25", "t0 me", "0 6",
]


def remove_ui_noise(text: str) -> str:
    """Strip email client UI chrome from OCR-extracted text.

    Matching is case-sensitive and whole-phrase so legitimate words sharing a
    substring with a noise phrase are left intact (e.g. "Sentence" is not clipped by "Sent").
    """
    for phrase in UI_NOISE_WORDS:
        pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
        text = re.sub(pattern, " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _build_input_dataframe(text: str) -> pd.DataFrame:
    """Wrap a plain text string in a one-row DataFrame expected by both classifiers."""
    return pd.DataFrame({
        "text":    [text],
        "urls":    [extract_urls(text)],
        "type":    ["email"],
        "sender":  [""],
        "subject": [""],
    })


def _build_explanation(signals: dict, score: float, label: str) -> str:
    """Generate a short human-readable explanation of the final verdict."""
    xgb  = signals["xgb_prob"]
    bert = signals["bert_prob"]
    diff = abs(xgb - bert)

    input_type = "image" if signals["is_image_input"] else "text"

    if diff > 0.30:
        agreement = (
            f"Contradictory signals: XGBoost says {xgb:.0%} phishing "
            f"while DistilBERT says {bert:.0%}."
        )
    elif xgb > 0.5 and bert > 0.5:
        agreement = (
            f"Both models agree: XGBoost {xgb:.0%}, DistilBERT {bert:.0%}."
        )
    elif xgb < 0.5 and bert < 0.5:
        agreement = (
            f"Both models agree this is legitimate: XGBoost {xgb:.0%}, "
            f"DistilBERT {bert:.0%}."
        )
    else:
        agreement = (
            f"Models are near the boundary: XGBoost {xgb:.0%}, "
            f"DistilBERT {bert:.0%}."
        )

    if signals["is_image_input"]:
        ocr_note = f" OCR confidence was {signals['ocr_confidence']:.0%}."
    else:
        ocr_note = ""

    verdict_note = f" Final verdict ({input_type} input): {label} at {score:.0%}."

    return agreement + ocr_note + verdict_note


class PhishingDetectionPipeline:
    """End-to-end phishing detection pipeline accepting text, image, or both.

    All models are loaded lazily; missing checkpoints are skipped with a warning.
    """

    def __init__(self, models_dir: str | Path = "models"):
        self.models_dir = Path(models_dir)

        # All sub-models start as None and are loaded on first use
        self._xgb:    Optional[object] = None
        self._bert:   Optional[object] = None
        self._fusion: Optional[FusionClassifier] = None
        self._ocr:    Optional[object] = None

        self._xgb_loaded    = False
        self._bert_loaded   = False
        self._fusion_loaded = False
        self._ocr_loaded    = False

    def _load_xgb(self) -> None:
        """Load PhishingXGBClassifier if the checkpoint exists."""
        if self._xgb_loaded:
            return
        self._xgb_loaded = True

        model_file = self.models_dir / "xgb_phishing.json"
        if not model_file.exists():
            logger.warning("XGBoost checkpoint not found at %s, skipping.", model_file)
            return

        from src.text_pipeline.text_classifier import PhishingXGBClassifier
        try:
            self._xgb = PhishingXGBClassifier(models_dir=self.models_dir).load("xgb_phishing")
            logger.info("XGBoost model loaded.")
        except Exception as exc:
            logger.warning("Failed to load XGBoost model: %s", exc)

    def _load_bert(self) -> None:
        """Load PhishingBERTClassifier if the checkpoint exists."""
        if self._bert_loaded:
            return
        self._bert_loaded = True

        bert_dir = self.models_dir / "bert_phishing"
        if not bert_dir.exists():
            logger.warning("DistilBERT checkpoint not found at %s, skipping.", bert_dir)
            return

        from src.text_pipeline.bert_classifier import PhishingBERTClassifier
        try:
            self._bert = PhishingBERTClassifier(models_dir=self.models_dir).load("bert_phishing")
            logger.info("DistilBERT model loaded.")
        except Exception as exc:
            logger.warning("Failed to load DistilBERT model: %s", exc)

    def _load_fusion(self) -> None:
        """Load FusionClassifier; always instantiates so the fallback is available."""
        if self._fusion_loaded:
            return
        self._fusion_loaded = True

        self._fusion = FusionClassifier(models_dir=self.models_dir)

        fusion_file = self.models_dir / "fusion_classifier.pkl"
        if not fusion_file.exists():
            logger.warning(
                "Fusion classifier not found at %s. "
                "Using weighted-average fallback until trained.",
                fusion_file,
            )
            return

        try:
            self._fusion.load("fusion_classifier")
            logger.info("FusionClassifier loaded.")
        except Exception as exc:
            logger.warning("Failed to load FusionClassifier: %s", exc)

    def _load_ocr(self) -> None:
        """Initialise the OCR extractor (lazy, slow on first call)."""
        if self._ocr_loaded:
            return
        self._ocr_loaded = True

        from src.image_pipeline.ocr_extractor import OCRExtractor
        self._ocr = OCRExtractor()

    def _get_xgb_prob(self, df: pd.DataFrame) -> float:
        """Return XGBoost phishing probability, or 0.5 (neutral) if unavailable."""
        if self._xgb is None:
            return 0.5
        try:
            return float(self._xgb.predict_proba(df)[0, 1])
        except Exception as exc:
            logger.warning("XGBoost inference failed: %s", exc)
            return 0.5

    def _get_bert_prob(self, df: pd.DataFrame) -> float:
        """Return DistilBERT phishing probability, or 0.5 (neutral) if unavailable."""
        if self._bert is None:
            return 0.5
        try:
            return float(self._bert.predict_proba(df)[0, 1])
        except Exception as exc:
            logger.warning("DistilBERT inference failed: %s", exc)
            return 0.5

    def analyze(
        self,
        text: Optional[str] = None,
        image_path: Optional[str | Path] = None,
    ) -> dict:
        """Run the full phishing detection pipeline on text, an image, or both.

        At least one of text or image_path must be provided. For image-only input,
        text is extracted by OCR and cleaned with remove_ui_noise() before classification.
        """
        if text is None and image_path is None:
            raise ValueError("At least one of text or image_path must be provided.")

        self._load_xgb()
        self._load_bert()
        self._load_fusion()

        ocr_confidence = 0.5  # neutral default for text-only input
        is_image_input = 0
        extracted_text = None
        cleaned_text   = None
        n_ocr_words    = 0

        if image_path is not None:
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            self._load_ocr()
            ocr_result     = self._ocr.extract(image_path)
            extracted_text = ocr_result["full_text"]
            ocr_confidence = ocr_result["mean_confidence"]
            n_ocr_words    = ocr_result["n_detections"]
            is_image_input = 1

            if text is None:
                cleaned_text = remove_ui_noise(extracted_text)
                text = cleaned_text
            else:
                cleaned_text = remove_ui_noise(extracted_text)

        df = _build_input_dataframe(text)

        xgb_prob  = self._get_xgb_prob(df)
        bert_prob = self._get_bert_prob(df)

        signals = {
            "xgb_prob":       xgb_prob,
            "bert_prob":      bert_prob,
            "ocr_confidence": ocr_confidence,
            "is_image_input": is_image_input,
        }

        score = self._fusion.predict_proba(signals)
        label = "PHISHING" if score >= 0.5 else "LEGITIMATE"

        return {
            "score":          score,
            "label":          label,
            "confidence":     confidence_tier(score),
            "explanation":    _build_explanation(signals, score, label),
            "signals":        signals,
            "image_path":     str(image_path) if image_path is not None else None,
            "extracted_text": extracted_text,
            "cleaned_text":   cleaned_text,
            "n_ocr_words":    n_ocr_words,
        }
