"""
tests/test_data_loader.py
Smoke-test each dataset loader: load the file, print shape, class balance, and first two rows.
Skips any file that does not exist so the script runs before all data is downloaded.

Usage: python tests/test_data_loader.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.text_pipeline.data_loader import (
    load_ceas08,
    load_enron,
    load_ling,
    load_nazario_csv,
    load_nigerian_fraud,
    load_phishing_email,
    load_spamassassin_csv,
    load_sms_spam_collection,
    load_malicious_phish,
)

EMAIL_DIR = PROJECT_ROOT / "data" / "raw" / "emails"
SMS_DIR   = PROJECT_ROOT / "data" / "raw" / "sms_spam"
URL_DIR   = PROJECT_ROOT / "data" / "raw" / "urls"

DATASETS = [
    ("CEAS_08",           EMAIL_DIR / "CEAS_08.csv",                    load_ceas08),
    ("Enron",             EMAIL_DIR / "Enron.csv",                      load_enron),
    ("Ling",              EMAIL_DIR / "Ling.csv",                       load_ling),
    ("Nazario",           EMAIL_DIR / "Nazario.csv",                    load_nazario_csv),
    ("Nigerian_Fraud",    EMAIL_DIR / "Nigerian_Fraud.csv",             load_nigerian_fraud),
    ("phishing_email",    EMAIL_DIR / "phishing_email.csv",             load_phishing_email),
    ("SpamAssassin",      EMAIL_DIR / "SpamAssasin.csv",                load_spamassassin_csv),
    ("SMS Spam",          SMS_DIR   / "SMSSpamCollection",              load_sms_spam_collection),
    ("malicious_phish",   URL_DIR   / "malicious_phish.csv",            load_malicious_phish),
]

_SEP = "-" * 60


def _class_balance(df: pd.DataFrame) -> str:
    counts = df["label"].value_counts().sort_index()
    parts = []
    for lbl, cnt in counts.items():
        name = "phishing/spam" if lbl == 1 else "legitimate/ham"
        pct  = cnt / len(df) * 100
        parts.append(f"  {lbl} ({name}): {cnt:,}  ({pct:.1f}%)")
    return "\n".join(parts)


def _preview(df: pd.DataFrame) -> str:
    preview_cols = [c for c in ["label", "source", "type", "text"] if c in df.columns]
    rows = []
    for _, row in df[preview_cols].head(2).iterrows():
        text_snippet = str(row.get("text", ""))[:120].replace("\n", " ")
        rows.append(
            f"  label={row['label']}  source={row.get('source','')}  "
            f"type={row.get('type','')}\n  text: {text_snippet!r}"
        )
    return "\n".join(rows)


def run_all() -> None:
    passed = 0
    skipped = 0
    failed = 0

    for name, path, loader_fn in DATASETS:
        print(_SEP)
        print(f"Dataset : {name}")
        print(f"File    : {path}")

        if not path.exists():
            print("Status  : SKIPPED (file not found)")
            skipped += 1
            continue

        try:
            df = loader_fn(path)
        except Exception as exc:
            print(f"Status  : FAILED\nError   : {exc}")
            failed += 1
            continue

        print(f"Status  : OK")
        print(f"Shape   : {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"Columns : {df.columns.tolist()}")
        print("Balance :")
        print(_class_balance(df))
        print("First 2 rows:")
        print(_preview(df))
        passed += 1

    print(_SEP)
    print(f"\nSummary: {passed} passed, {skipped} skipped, {failed} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    run_all()
