"""Extracts text from images using EasyOCR with image preprocessing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageEnhance

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Prepare a raw screenshot for OCR (resize, contrast, sharpen, denoise)."""

    def __init__(
        self,
        target_width: int = 1280,
        denoise: bool = True,
        contrast_factor: float = 1.4,
        sharpness_factor: float = 1.3,
    ):
        self.target_width = target_width
        self.denoise = denoise
        self.contrast_factor = contrast_factor
        self.sharpness_factor = sharpness_factor

    def process(self, image_input: str | Path | np.ndarray | Image.Image) -> np.ndarray:
        """Load (or accept) an image, apply preprocessing, and return a uint8 RGB numpy array."""
        pil_img = self._to_pil(image_input)
        pil_img = self._resize(pil_img)

        if self.contrast_factor != 1.0:
            pil_img = ImageEnhance.Contrast(pil_img).enhance(self.contrast_factor)

        if self.sharpness_factor != 1.0:
            pil_img = ImageEnhance.Sharpness(pil_img).enhance(self.sharpness_factor)

        img_array = np.array(pil_img.convert("RGB"), dtype=np.uint8)

        if self.denoise:
            img_array = self._denoise(img_array)

        return img_array

    def _to_pil(self, image_input) -> Image.Image:
        """Convert any supported input type to a PIL Image."""
        if isinstance(image_input, Image.Image):
            return image_input
        if isinstance(image_input, np.ndarray):
            # OpenCV images are BGR; convert to RGB
            if image_input.ndim == 3 and image_input.shape[2] == 3:
                image_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            return Image.fromarray(image_input)
        return Image.open(Path(image_input))

    def _resize(self, img: Image.Image) -> Image.Image:
        """Upscale narrow images to target_width, preserving aspect ratio."""
        w, h = img.size
        if w >= self.target_width:
            return img
        scale = self.target_width / w
        new_h = int(h * scale)
        return img.resize((self.target_width, new_h), Image.LANCZOS)

    def _denoise(self, img: np.ndarray) -> np.ndarray:
        """Apply OpenCV fast non-local means denoising."""
        try:
            return cv2.fastNlMeansDenoisingColored(
                img,
                None,
                h=10,
                hColor=10,
                templateWindowSize=7,
                searchWindowSize=21,
            )
        except Exception as exc:
            logger.debug("Denoising failed, returning original: %s", exc)
            return img


class OCRExtractor:
    """Extract text from images using EasyOCR with optional preprocessing."""

    def __init__(
        self,
        languages: list[str] | None = None,
        gpu: bool = True,
        preprocessor: Optional[ImagePreprocessor] = None,
        confidence_threshold: float = 0.3,
    ):
        self.languages = languages or ["en"]
        self.gpu = gpu
        self.confidence_threshold = confidence_threshold
        self.preprocessor = preprocessor if preprocessor is not None else ImagePreprocessor()

        # lazy: avoid slow import / model download at startup
        self._reader = None

    def _get_reader(self):
        """Initialise the EasyOCR Reader on first use."""
        if self._reader is None:
            try:
                import easyocr
            except ImportError as exc:
                raise ImportError(
                    "easyocr is required for OCRExtractor. "
                    "Install it with: pip install easyocr"
                ) from exc

            logger.info(
                "Initialising EasyOCR (languages=%s, gpu=%s). "
                "This downloads models on the first run...",
                self.languages, self.gpu,
            )
            self._reader = easyocr.Reader(self.languages, gpu=self.gpu)
            logger.info("EasyOCR ready.")
        return self._reader

    def extract(self, image_input: str | Path | np.ndarray | Image.Image) -> dict:
        """Run preprocessing and OCR on a single image."""
        preprocessed = self.preprocessor.process(image_input)
        reader = self._get_reader()

        raw_results = reader.readtext(preprocessed, detail=1, paragraph=False)

        kept = []
        for bbox, text, confidence in raw_results:
            if confidence < self.confidence_threshold:
                continue
            kept.append({
                "text": text,
                "bbox": [list(map(float, point)) for point in bbox],
                "confidence": float(confidence),
            })

        full_text = " ".join(d["text"] for d in kept)
        mean_conf = float(np.mean([d["confidence"] for d in kept])) if kept else 0.0

        return {
            "full_text": full_text,
            "detections": kept,
            "n_detections": len(kept),
            "mean_confidence": mean_conf,
        }

    def extract_batch(
        self,
        image_paths: list[str | Path],
        *,
        fail_silent: bool = True,
    ) -> list[dict]:
        """Run OCR on a list of image paths; failed images produce empty result dicts when fail_silent=True."""
        results = []
        for i, path in enumerate(image_paths):
            logger.info("OCR %d/%d: %s", i + 1, len(image_paths), path)
            try:
                result = self.extract(path)
                result["path"] = str(path)
                result["error"] = None
            except Exception as exc:
                logger.warning("OCR failed for %s: %s", path, exc)
                if not fail_silent:
                    raise
                result = {
                    "path": str(path),
                    "full_text": "",
                    "detections": [],
                    "n_detections": 0,
                    "mean_confidence": 0.0,
                    "error": str(exc),
                }
            results.append(result)
        return results

    def extract_directory(
        self,
        directory: str | Path,
        extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tiff"),
        **kwargs,
    ) -> list[dict]:
        """Extract text from all images in a directory (recursive)."""
        directory = Path(directory)
        image_paths = sorted(
            p for p in directory.rglob("*")
            if p.suffix.lower() in extensions
        )
        logger.info("Found %d images in %s", len(image_paths), directory)
        return self.extract_batch(image_paths, **kwargs)
