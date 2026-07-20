"""Streamlit demo for the multi-modal phishing detection pipeline."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fusion.pipeline import PhishingDetectionPipeline

# Trained models live in the main repository (see scripts/scan_image.py).
MODELS_DIR = Path("D:/phishing-detection/models")

# muted palette for the demo
COLOR_PHISHING = "#c0392b"   # muted red
COLOR_LEGIT = "#1e8449"      # muted green
TIER_COLORS = {
    "LOW": "#7f8c8d",        # grey — weak signal
    "MEDIUM": "#b7950b",     # amber
    "HIGH": "#2471a3",       # blue
}

# sample used by the "Try example" button
SAMPLE_PHISHING_EMAIL = (
    "From: PayPal Security <service@paypal-secure-alerts.com>\n"
    "Subject: Unusual activity detected - action required\n\n"
    "Dear Customer,\n\n"
    "We have detected unusual activity on your PayPal account. Your account "
    "access has been temporarily limited for your protection.\n\n"
    "You must verify your identity within 24 hours or your account will be "
    "permanently suspended. Please confirm your details by clicking the secure "
    "link below:\n\n"
    "http://paypal-secure-alerts.com/verify-account?id=8837201\n\n"
    "Failure to verify will result in immediate account closure.\n\n"
    "Sincerely,\n"
    "The PayPal Security Team"
)

SUPPORTED_IMAGE_TYPES = ["png", "jpg", "jpeg"]


@st.cache_resource(show_spinner="Loading detection models...")
def load_pipeline() -> PhishingDetectionPipeline:
    """Instantiate the pipeline once and reuse it across reruns."""
    return PhishingDetectionPipeline(models_dir=MODELS_DIR)


def render_sidebar() -> None:
    """Render the project description, performance summary, and about section."""
    st.sidebar.title("Phishing Detection Pipeline")
    st.sidebar.write(
        "A multi-modal phishing detector that fuses classical and deep-learning "
        "signals across email, SMS, and screenshot inputs."
    )

    st.sidebar.subheader("Model performance")
    st.sidebar.table(
        {
            "Model": ["XGBoost", "DistilBERT", "Fusion"],
            "Accuracy": ["86.19%", "98.74%", "98.62%"],
        }
    )

    with st.sidebar.expander("About"):
        st.markdown(
            "**Pipeline stages**\n"
            "- **XGBoost** on hand-crafted URL, text, and structural features "
            "(with SHAP explainability)\n"
            "- **DistilBERT** fine-tuned for semantic phishing detection\n"
            "- **LightGBM** meta-classifier fusing both signals\n"
            "- **EasyOCR** for extracting text from screenshots\n\n"
            "**Tech stack**: Python, scikit-learn, XGBoost, LightGBM, "
            "PyTorch, Hugging Face Transformers, EasyOCR, Streamlit.\n\n"
            "_MSc dissertation project._"
        )


def render_verdict(result: dict) -> None:
    """Show the verdict as coloured text plus a confidence progress bar."""
    is_phishing = result["label"] == "PHISHING"
    color = COLOR_PHISHING if is_phishing else COLOR_LEGIT
    headline = "PHISHING DETECTED" if is_phishing else "LEGITIMATE"

    st.markdown(
        f"<h2 style='color:{color}; margin-bottom:0.2rem;'>{headline}</h2>",
        unsafe_allow_html=True,
    )

    # confidence tier as small coloured text
    tier = result.get("confidence", "LOW")
    tier_color = TIER_COLORS.get(tier, "#7f8c8d")
    st.markdown(
        f"<span style='color:{tier_color}; font-size:0.9rem; font-weight:600;'>"
        f"Confidence tier: {tier}</span>",
        unsafe_allow_html=True,
    )

    score = float(result["score"])
    st.progress(score, text=f"Phishing probability: {score:.1%}")


def render_metrics(result: dict) -> None:
    """Show XGBoost, DistilBERT, and Fusion scores as three neutral metric cards."""
    signals = result.get("signals", {})
    xgb = signals.get("xgb_prob")
    bert = signals.get("bert_prob")
    fusion = float(result["score"])

    col_xgb, col_bert, col_fusion = st.columns(3)
    col_xgb.metric("XGBoost score", "N/A" if xgb is None else f"{xgb:.1%}")
    col_bert.metric("DistilBERT score", "N/A" if bert is None else f"{bert:.1%}")
    col_fusion.metric("Fusion score", f"{fusion:.1%}")


def render_result(result: dict, *, is_image: bool) -> None:
    """Render the full result block: verdict, metrics, OCR text, and explanation."""
    st.divider()
    render_verdict(result)
    st.write("")
    render_metrics(result)

    if is_image:
        extracted = result.get("extracted_text") or ""
        n_words = result.get("n_ocr_words", 0)
        with st.expander(f"Extracted OCR text ({n_words} words)"):
            st.text(extracted.strip() if extracted.strip() else "[no text detected]")

    explanation = result.get("explanation")
    if explanation:
        st.caption(explanation)


def analyze_text(pipeline: PhishingDetectionPipeline, text: str) -> None:
    """Validate the text input, run analysis, and render the result."""
    if not text or not text.strip():
        st.error("Please paste some email or SMS content before analysing.")
        return
    try:
        with st.spinner("Analysing text..."):
            result = pipeline.analyze(text=text)
    except Exception as exc:  # noqa: BLE001 - surface any failure to the user
        st.error(f"Analysis failed: {exc}")
        return
    render_result(result, is_image=False)


def analyze_image(pipeline: PhishingDetectionPipeline, uploaded_file) -> None:
    """Persist the upload to a temp file, run OCR + analysis, and render the result."""
    if uploaded_file is None:
        st.error("Please upload an image before analysing.")
        return

    suffix = Path(uploaded_file.name).suffix or ".png"
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = Path(tmp.name)

        with st.spinner("Running OCR and analysing image..."):
            result = pipeline.analyze(image_path=tmp_path)
    except Exception as exc:  # noqa: BLE001 - surface any failure to the user
        st.error(f"Analysis failed: {exc}")
        return
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    render_result(result, is_image=True)


def main() -> None:
    st.set_page_config(page_title="Phishing Detection Pipeline", layout="wide")

    render_sidebar()

    try:
        pipeline = load_pipeline()
    except Exception as exc:  # noqa: BLE001 - fatal, cannot continue without models
        st.error(f"Failed to load the detection pipeline: {exc}")
        st.stop()

    st.title("Phishing Detection Pipeline")
    st.write(
        "Analyse email/SMS text or an email screenshot and get a fused phishing "
        "verdict from three complementary models."
    )

    tab_text, tab_image = st.tabs(["Text", "Image"])

    with tab_text:
        # pre-fill the text area via session state before the widget is built
        if st.button("Try example", key="try_example"):
            st.session_state["text_input"] = SAMPLE_PHISHING_EMAIL

        text = st.text_area(
            "Email or SMS content",
            key="text_input",
            height=260,
            placeholder="Paste the suspicious email or SMS text here...",
        )
        if st.button("Analyze", key="analyze_text", type="primary"):
            analyze_text(pipeline, text)

    with tab_image:
        uploaded = st.file_uploader(
            "Upload an email or message screenshot",
            type=SUPPORTED_IMAGE_TYPES,
            key="image_upload",
        )
        if uploaded is not None:
            st.image(uploaded, caption="Preview", width=420)

        if st.button("Analyze", key="analyze_image", type="primary"):
            analyze_image(pipeline, uploaded)


if __name__ == "__main__":
    main()
