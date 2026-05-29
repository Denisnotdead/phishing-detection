"""
data_loader.py
Loads and preprocesses phishing/legitimate email, SMS, and URL datasets.
Returns tidy DataFrames with columns: text, label (int), source, type.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Candidate column names in priority order for each semantic role
_LABEL_CANDIDATES = [
    "label", "Label", "LABEL",
    "class", "Class",
    "type", "Type",
    "category", "Category",
    "spam", "Spam",
    "target", "Target",
]

_TEXT_CANDIDATES = [
    "text", "Text", "TEXT",
    "body", "Body", "BODY",
    "message", "Message", "MESSAGE",
    "content", "Content",
    "mail", "Mail",
    "subject_body",
    "email", "Email",
]

_SUBJECT_CANDIDATES = ["subject", "Subject", "SUBJECT"]
_SENDER_CANDIDATES  = ["sender", "Sender", "from", "From", "FROM"]
_DATE_CANDIDATES    = ["date", "Date", "DATE"]

# Canonical label mappings (lower-cased string → int)
_LABEL_MAP: dict[str, int] = {
    "spam": 1, "phishing": 1, "phishing email": 1,
    "malicious": 1, "fraud": 1, "scam": 1,
    "1": 1, "yes": 1,
    "ham": 0, "legitimate": 0, "safe email": 0,
    "benign": 0, "normal": 0, "good": 0,
    "0": 0, "no": 0,
}


def _pick_col(columns: list[str], candidates: list[str]) -> Optional[str]:
    """Return the first candidate that appears in *columns*, else None."""
    col_set = set(columns)
    for c in candidates:
        if c in col_set:
            return c
    return None


def _normalise_label(series: pd.Series) -> pd.Series:
    """Coerce a label column of mixed type to integer 0/1."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)

    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int)

    mapped = series.astype(str).str.strip().str.lower().map(_LABEL_MAP)
    n_unknown = mapped.isna().sum()
    if n_unknown > 0:
        logger.warning(
            "%d rows have unrecognised label values and will be dropped.", n_unknown
        )
    return mapped


def _load_email_csv(
    filepath: Path,
    source: str,
    *,
    text_col: Optional[str] = None,
    label_col: Optional[str] = None,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """Load a CSV email dataset with automatic column detection."""
    try:
        raw = pd.read_csv(filepath, encoding=encoding, low_memory=False)
    except UnicodeDecodeError:
        logger.debug("UTF-8 failed for %s, retrying with latin-1", filepath.name)
        raw = pd.read_csv(filepath, encoding="latin-1", low_memory=False)

    cols = raw.columns.tolist()

    t_col = text_col or _pick_col(cols, _TEXT_CANDIDATES)
    if t_col is None:
        raise ValueError(
            f"{filepath.name}: cannot detect a text column. "
            f"Available columns: {cols}"
        )

    l_col = label_col or _pick_col(cols, _LABEL_CANDIDATES)
    if l_col is None:
        raise ValueError(
            f"{filepath.name}: cannot detect a label column. "
            f"Available columns: {cols}"
        )

    df = pd.DataFrame()
    df["text"]   = raw[t_col].astype(str).str.strip()
    df["label"]  = _normalise_label(raw[l_col])
    df["source"] = source
    df["type"]   = "email"

    # Carry optional metadata columns when present
    for role, candidates in [
        ("subject", _SUBJECT_CANDIDATES),
        ("sender",  _SENDER_CANDIDATES),
        ("date",    _DATE_CANDIDATES),
    ]:
        found = _pick_col(cols, candidates)
        if found:
            df[role] = raw[found].astype(str).str.strip()

    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    logger.info("%s: loaded %d rows from %s", source, len(df), filepath.name)
    return df


def load_ceas08(filepath: str | Path) -> pd.DataFrame:
    """Load the CEAS 2008 spam-filter challenge dataset."""
    return _load_email_csv(Path(filepath), source="CEAS_08")


def load_enron(filepath: str | Path) -> pd.DataFrame:
    """Load the Enron email corpus dataset (CSV form)."""
    return _load_email_csv(Path(filepath), source="Enron")


def load_ling(filepath: str | Path) -> pd.DataFrame:
    """Load the Ling-spam corpus dataset (CSV form)."""
    return _load_email_csv(Path(filepath), source="Ling")


def load_nazario_csv(filepath: str | Path) -> pd.DataFrame:
    """Load the Nazario phishing corpus in CSV form."""
    return _load_email_csv(Path(filepath), source="Nazario")


def load_nigerian_fraud(filepath: str | Path) -> pd.DataFrame:
    """Load the Nigerian Fraud (419) email dataset."""
    return _load_email_csv(Path(filepath), source="Nigerian_Fraud")


def load_phishing_email(filepath: str | Path) -> pd.DataFrame:
    """Load the Kaggle phishing email dataset (expects text_combined and label columns)."""
    filepath = Path(filepath)
    try:
        raw = pd.read_csv(filepath, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        raw = pd.read_csv(filepath, encoding="latin-1", low_memory=False)

    required = {"text_combined", "label"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(
            f"phishing_email.csv is missing expected columns: {missing}. "
            f"Found: {raw.columns.tolist()}"
        )

    df = pd.DataFrame({
        "text":   raw["text_combined"].astype(str).str.strip(),
        "label":  raw["label"].astype(int),
        "source": "phishing_email",
        "type":   "email",
    })

    df = df.reset_index(drop=True)
    logger.info("phishing_email: loaded %d rows", len(df))
    return df


def load_spamassassin_csv(filepath: str | Path) -> pd.DataFrame:
    """Load the SpamAssassin corpus in CSV form."""
    return _load_email_csv(Path(filepath), source="SpamAssassin")


def load_sms_spam_collection(filepath: str | Path) -> pd.DataFrame:
    """Load the UCI SMS Spam Collection (tab-separated, no header)."""
    filepath = Path(filepath)
    raw = pd.read_csv(
        filepath,
        sep="\t",
        header=None,
        names=["label_str", "text"],
        encoding="utf-8",
        on_bad_lines="skip",
    )
    raw["label"]  = _normalise_label(raw["label_str"])
    raw["source"] = "SMS Spam Collection"
    raw["type"]   = "sms"
    df = raw[["text", "label", "source", "type"]].dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    logger.info("SMS Spam Collection: loaded %d messages", len(df))
    return df


def load_malicious_phish(filepath: str | Path) -> pd.DataFrame:
    """Load the Kaggle Malicious URLs dataset; maps any non-benign type to label=1."""
    filepath = Path(filepath)
    raw = pd.read_csv(filepath, encoding="utf-8", low_memory=False)

    url_col  = _pick_col(raw.columns.tolist(), ["url", "URL", "Url"])
    type_col = _pick_col(raw.columns.tolist(), ["type", "Type", "label", "Label", "category"])

    if url_col is None or type_col is None:
        raise ValueError(
            f"malicious_phish.csv: expected 'url' and 'type' columns. "
            f"Found: {raw.columns.tolist()}"
        )

    # Malicious = any non-benign URL
    raw["label"] = (
        raw[type_col].astype(str).str.strip().str.lower() != "benign"
    ).astype(int)

    df = pd.DataFrame({
        "text":     raw[url_col].astype(str).str.strip(),
        "label":    raw["label"],
        "source":   "malicious_phish",
        "type":     "url",
        "url_type": raw[type_col].astype(str).str.strip().str.lower(),
    })

    logger.info(
        "malicious_phish: loaded %d URLs (%d malicious, %d benign)",
        len(df),
        df["label"].sum(),
        (df["label"] == 0).sum(),
    )
    return df


_URL_PATTERN = re.compile(
    r"https?://[^\s<>\"']+|www\.[^\s<>\"']+",
    flags=re.IGNORECASE,
)


def extract_urls(text: str) -> list[str]:
    """Return all URLs found in *text*; returns empty list for non-string input."""
    if not isinstance(text, str):
        return []
    return _URL_PATTERN.findall(text)


def clean_text(
    text: str,
    *,
    lowercase: bool = True,
    remove_urls: bool = False,
    remove_punctuation: bool = False,
) -> str:
    """Basic text normalisation: lowercase, optional URL removal, optional punctuation strip."""
    if lowercase:
        text = text.lower()
    if remove_urls:
        text = _URL_PATTERN.sub(" URL ", text)
    if remove_punctuation:
        text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_dataframe(
    df: pd.DataFrame,
    *,
    lowercase: bool = True,
    remove_urls: bool = False,
    remove_punctuation: bool = False,
    drop_duplicates: bool = True,
    drop_empty: bool = True,
) -> pd.DataFrame:
    """Apply cleaning steps to the text column; adds a urls column with extracted URLs."""
    df = df.copy()

    # Fill NaN before text processing so nothing downstream receives a float
    df["text"] = df["text"].fillna("")

    # Extract URLs before cleaning modifies the text
    df["urls"] = df["text"].apply(lambda x: extract_urls(x) if isinstance(x, str) else [])

    df["text"] = df["text"].apply(
        lambda t: clean_text(
            t,
            lowercase=lowercase,
            remove_urls=remove_urls,
            remove_punctuation=remove_punctuation,
        )
    )

    if drop_empty:
        df = df[df["text"].str.len() > 0]

    if drop_duplicates:
        df = df.drop_duplicates(subset=["text"])

    df = df.reset_index(drop=True)
    logger.info("After preprocessing: %d rows remain", len(df))
    return df


_DEFAULT_EMAIL_DIR = Path("data/raw/emails")
_DEFAULT_SMS_FILE  = Path("data/raw/sms_spam/SMSSpamCollection")
_DEFAULT_URL_FILE  = Path("data/raw/urls/malicious_phish.csv")

# Maps filename stem → loader function for every email CSV
_EMAIL_LOADERS: dict[str, callable] = {
    "CEAS_08":        load_ceas08,
    "Enron":          load_enron,
    "Ling":           load_ling,
    "Nazario":        load_nazario_csv,
    "Nigerian_Fraud": load_nigerian_fraud,
    "phishing_email": load_phishing_email,
    "SpamAssasin":    load_spamassassin_csv,   # note: single 's' as shipped
}


def load_all(
    *,
    email_dir: Optional[str | Path] = None,
    sms_path: Optional[str | Path] = None,
    url_path: Optional[str | Path] = None,
    preprocess: bool = True,
    **preprocess_kwargs,
) -> pd.DataFrame:
    """Load all datasets and return a single concatenated DataFrame.

    Each path defaults to the canonical project location; pass None to skip a source.
    """
    frames: list[pd.DataFrame] = []

    # Load email CSVs
    e_dir = Path(email_dir) if email_dir is not None else _DEFAULT_EMAIL_DIR
    if e_dir is not None and e_dir.is_dir():
        for stem, loader_fn in _EMAIL_LOADERS.items():
            csv_path = e_dir / f"{stem}.csv"
            if csv_path.exists():
                try:
                    frames.append(loader_fn(csv_path))
                except Exception as exc:
                    logger.warning("Skipping %s: %s", csv_path.name, exc)
            else:
                logger.debug("Not found, skipping: %s", csv_path)
    elif email_dir is not None:
        logger.warning("email_dir does not exist: %s", e_dir)

    # Load SMS dataset
    s_path = Path(sms_path) if sms_path is not None else _DEFAULT_SMS_FILE
    if s_path is not None and s_path.exists():
        try:
            frames.append(load_sms_spam_collection(s_path))
        except Exception as exc:
            logger.warning("Skipping SMS dataset: %s", exc)
    elif sms_path is not None:
        logger.warning("SMS file not found: %s", s_path)

    # Load URL dataset
    u_path = Path(url_path) if url_path is not None else _DEFAULT_URL_FILE
    if u_path is not None and u_path.exists():
        try:
            frames.append(load_malicious_phish(u_path))
        except Exception as exc:
            logger.warning("Skipping URL dataset: %s", exc)
    elif url_path is not None:
        logger.warning("URL file not found: %s", u_path)

    if not frames:
        logger.warning("No datasets loaded; returning empty DataFrame.")
        return pd.DataFrame(columns=["text", "label", "source", "type"])

    combined = pd.concat(frames, ignore_index=True)

    if preprocess:
        combined = preprocess_dataframe(combined, **preprocess_kwargs)

    return combined
