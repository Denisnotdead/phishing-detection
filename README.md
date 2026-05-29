# Phishing Detection Pipeline

A multi-modal phishing detection system that classifies emails, SMS messages, URLs, and screenshots using a combination of hand-crafted features, fine-tuned language models, and computer vision. An XGBoost classifier handles structured feature signals, DistilBERT captures semantic patterns in raw text, EfficientNet-B0 processes screenshot inputs via OCR, and a LightGBM meta-classifier fuses all signals into a single phishing probability.

## Tech Stack

| Component | Library |
|---|---|
| Language | Python 3.10 |
| Deep learning | PyTorch, Hugging Face Transformers |
| Text model | DistilBERT (`distilbert-base-uncased`) |
| Feature model | XGBoost, scikit-learn |
| Fusion model | LightGBM |
| Image / OCR | EfficientNet-B0 (torchvision), EasyOCR |
| Explainability | SHAP |
| Data | pandas, NumPy |

## Project Structure

```
phishing-detection/
├── data/
│   ├── raw/                  # original dataset files
│   └── processed/            # combined_dataset.csv after preprocessing
├── models/                   # saved model checkpoints
│   ├── xgb_phishing.json
│   ├── bert_phishing/
│   └── fusion_classifier.pkl
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_evaluation.ipynb
├── reports/                  # evaluation JSON and all plot outputs
├── scripts/
│   └── scan_image.py         # main inference entry point
├── src/
│   ├── text_pipeline/        # data loading, feature extraction, XGBoost, DistilBERT, trainer
│   ├── image_pipeline/       # OCR extractor, EfficientNet classifier, image trainer
│   ├── fusion/               # LightGBM meta-classifier and unified pipeline
│   └── evaluation/           # evaluator and SHAP / attention explainability
└── tests/
    └── test_data_loader.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

**Scan a single image (fusion mode):**
```bash
python scripts/scan_image.py screenshot.png
```

**Scan a directory:**
```bash
python scripts/scan_image.py
```

**Choose a model:**
```bash
python scripts/scan_image.py --mode bert screenshot.png
python scripts/scan_image.py --mode xgb  screenshot.png
```

**Train the text pipeline:**
```bash
python -m src.text_pipeline.trainer
```

**Run evaluation and generate all plots:**
```bash
python -m src.evaluation.evaluator
```

## Results

Evaluated on a held-out 15% test split (stratified, `random_state=42`).

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| XGBoost | 86.19% | — | — |
| DistilBERT | 98.74% | — | — |
| Ensemble (LightGBM fusion) | 98.62% | — | — |

## Datasets

| Dataset | Type | Source |
|---|---|---|
| CEAS 2008 | Email | CEAS spam challenge |
| Enron | Email | Enron corpus |
| Ling | Email | Ling spam corpus |
| Nazario | Email | Phishing corpus (Nazario) |
| Nigerian Fraud | Email | 419 fraud emails |
| Phishing Email | Email | Kaggle (`text_combined` / `label`) |
| SpamAssassin | Email | Apache SpamAssassin public corpus |
| SMS Spam Collection | SMS | UCI ML repository |
| Malicious URLs | URL | `malicious_phish.csv` (Kaggle) |
