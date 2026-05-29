"""
feature_extractor.py
Extracts URL, statistical text, and structural/header features for phishing detection.
Exposes sklearn-compatible transformers and a convenience build_feature_matrix() function.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Optional
from urllib.parse import urlparse

import nltk
import numpy as np
import pandas as pd
import tldextract
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

# Ensure required NLTK data is available
for _resource in ("punkt", "stopwords", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{_resource}")
    except LookupError:
        nltk.download(_resource, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

_STOP_WORDS = set(stopwords.words("english"))

# Free/open registrars frequently abused for phishing
_SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "win", "bid",
    "click", "download", "zip", "review", "country", "kim",
    "science", "work", "party", "link",
}

# High-value brand keywords often spoofed in phishing domains
_BRAND_KEYWORDS = {
    "paypal", "apple", "google", "amazon", "microsoft", "facebook",
    "netflix", "ebay", "instagram", "twitter", "bank", "secure",
    "account", "login", "verify", "update", "confirm",
}

# Social-engineering phrases common in phishing
_URGENCY_PHRASES = [
    "urgent", "immediately", "account suspended", "verify your",
    "confirm your", "click here", "limited time", "act now",
    "your account", "unusual activity", "security alert",
    "prize", "winner", "congratulations", "free",
]


def _shannon_entropy(s: str) -> float:
    """Compute the Shannon entropy of a string (bits per character)."""
    if not s:
        return 0.0
    counts = Counter(s)
    length = len(s)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def _count_subdomains(extracted) -> int:
    """Return the number of subdomain labels (dot-separated components)."""
    if not extracted.subdomain:
        return 0
    return len(extracted.subdomain.split("."))


def _zero_url_features() -> dict:
    """Return a zero-valued URL feature dict used as a safe fallback for unparseable URLs."""
    return {
        "url_length": 0,
        "domain_length": 0,
        "path_length": 0,
        "query_length": 0,
        "num_dots": 0,
        "num_hyphens": 0,
        "num_underscores": 0,
        "num_slashes": 0,
        "num_at_signs": 0,
        "num_question_marks": 0,
        "num_equals": 0,
        "num_ampersands": 0,
        "num_percent": 0,
        "num_digits_in_domain": 0,
        "digit_ratio_url": 0.0,
        "digit_ratio_domain": 0.0,
        "entropy_url": 0.0,
        "entropy_domain": 0.0,
        "entropy_path": 0.0,
        "has_ip_address": 0,
        "has_https": 0,
        "has_port": 0,
        "num_subdomains": 0,
        "is_suspicious_tld": 0,
        "has_brand_keyword": 0,
        "has_double_slash_redirect": 0,
        "has_hex_encoding": 0,
        "has_url_shortener": 0,
    }


def extract_url_features(url: str) -> dict:
    """Extract a fixed-size feature dictionary from a single URL string.

    All operations are individually guarded; returns zeros on any failure so
    the pipeline never crashes on malformed input.
    """
    try:
        if not isinstance(url, str):
            url = ""

        full_url = url

        try:
            parsed = urlparse(full_url if "://" in full_url else "http://" + full_url)
        except Exception:
            parsed = urlparse("")

        try:
            ext = tldextract.extract(full_url)
            domain    = ext.domain   or ""
            subdomain = ext.subdomain or ""
            suffix    = ext.suffix   or ""
        except Exception:
            domain = subdomain = suffix = ""

        path  = parsed.path  or ""
        query = parsed.query or ""

        # parsed.port raises ValueError for out-of-range ports that phishing URLs sometimes use
        try:
            has_port = int(bool(parsed.port))
        except ValueError:
            has_port = 0

        try:
            has_ip = int(bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain)))
        except Exception:
            has_ip = 0

        try:
            has_brand = int(any(b in full_url.lower() for b in _BRAND_KEYWORDS))
        except Exception:
            has_brand = 0

        try:
            has_shortener = int(bool(re.search(
                r"bit\.ly|tinyurl|goo\.gl|t\.co|ow\.ly|is\.gd",
                full_url,
                re.IGNORECASE,
            )))
        except Exception:
            has_shortener = 0

        try:
            has_hex = int("%2" in query.upper())
        except Exception:
            has_hex = 0

        return {
            "url_length":             len(full_url),
            "domain_length":          len(domain),
            "path_length":            len(path),
            "query_length":           len(query),
            "num_dots":               full_url.count("."),
            "num_hyphens":            full_url.count("-"),
            "num_underscores":        full_url.count("_"),
            "num_slashes":            full_url.count("/"),
            "num_at_signs":           full_url.count("@"),
            "num_question_marks":     full_url.count("?"),
            "num_equals":             full_url.count("="),
            "num_ampersands":         full_url.count("&"),
            "num_percent":            full_url.count("%"),
            "num_digits_in_domain":   sum(c.isdigit() for c in domain),
            "digit_ratio_url":        sum(c.isdigit() for c in full_url) / max(len(full_url), 1),
            "digit_ratio_domain":     sum(c.isdigit() for c in domain)   / max(len(domain), 1),
            "entropy_url":            _shannon_entropy(full_url),
            "entropy_domain":         _shannon_entropy(domain),
            "entropy_path":           _shannon_entropy(path),
            "has_ip_address":         has_ip,
            "has_https":              int(parsed.scheme == "https"),
            "has_port":               has_port,
            "num_subdomains":         _count_subdomains(ext) if domain else 0,
            "is_suspicious_tld":      int(suffix.lower() in _SUSPICIOUS_TLDS),
            "has_brand_keyword":      has_brand,
            "has_double_slash_redirect": int("//" in path),
            "has_hex_encoding":       has_hex,
            "has_url_shortener":      has_shortener,
        }

    except Exception as exc:
        logger.debug("extract_url_features failed for %r: %s", url, exc)
        return _zero_url_features()


def aggregate_url_features(urls: list[str]) -> dict:
    """Aggregate per-URL features for a message with multiple URLs (mean, max, count)."""
    if not urls:
        dummy = extract_url_features("")
        return {"url_count": 0, **{f"url_agg_mean_{k}": 0.0 for k in dummy},
                **{f"url_agg_max_{k}": 0.0 for k in dummy}}

    feature_rows = [extract_url_features(u) for u in urls]
    keys = list(feature_rows[0].keys())
    values = {k: [r[k] for r in feature_rows] for k in keys}

    agg: dict = {"url_count": len(urls)}
    for k, vals in values.items():
        agg[f"url_agg_mean_{k}"] = float(np.mean(vals))
        agg[f"url_agg_max_{k}"] = float(np.max(vals))

    return agg


def extract_text_features(text: str) -> dict:
    """Extract statistical/lexical text features from a single message."""
    tokens = word_tokenize(text.lower())
    alpha_tokens = [t for t in tokens if t.isalpha()]
    content_tokens = [t for t in alpha_tokens if t not in _STOP_WORDS]

    num_chars = len(text)
    num_words = len(tokens)
    num_sentences = max(text.count(".") + text.count("!") + text.count("?"), 1)

    urgency_count = sum(
        1 for phrase in _URGENCY_PHRASES if phrase in text.lower()
    )

    return {
        "char_count": num_chars,
        "word_count": num_words,
        "unique_word_count": len(set(alpha_tokens)),
        "avg_word_length": (
            sum(len(t) for t in alpha_tokens) / max(len(alpha_tokens), 1)
        ),
        "sentence_count": num_sentences,
        "avg_sentence_length": num_words / num_sentences,
        "stopword_ratio": (len(alpha_tokens) - len(content_tokens)) / max(len(alpha_tokens), 1),
        "uppercase_ratio": sum(c.isupper() for c in text) / max(num_chars, 1),
        "digit_ratio": sum(c.isdigit() for c in text) / max(num_chars, 1),
        "special_char_ratio": sum(c in string.punctuation for c in text) / max(num_chars, 1),
        "exclamation_count": text.count("!"),
        "question_mark_count": text.count("?"),
        "html_tag_count": len(re.findall(r"<[^>]+>", text)),
        "urgency_word_count": urgency_count,
        "lexical_diversity": len(set(alpha_tokens)) / max(len(alpha_tokens), 1),
    }


def extract_structural_features(row: pd.Series) -> dict:
    """Extract email header and structural features from a DataFrame row.

    SMS rows receive zero/NaN for email-only features.
    """
    is_email = str(row.get("type", "sms")) == "email"
    sender = str(row.get("sender", "")) if is_email else ""
    subject = str(row.get("subject", "")) if is_email else ""
    text = str(row.get("text", ""))

    # Sender domain vs Reply-To mismatch is a strong phishing signal
    sender_domain = ""
    if "@" in sender:
        sender_domain = sender.split("@")[-1].strip(">").lower()

    reply_to_match = re.search(r"reply-to:\s*\S+@(\S+)", text, re.IGNORECASE)
    reply_to_domain = reply_to_match.group(1).lower() if reply_to_match else ""

    domain_mismatch = int(
        bool(sender_domain and reply_to_domain and sender_domain != reply_to_domain)
    )

    html_ratio = 0.0
    if text:
        tag_chars = sum(len(m.group()) for m in re.finditer(r"<[^>]+>", text))
        html_ratio = tag_chars / len(text)

    return {
        "is_email": int(is_email),
        "has_sender": int(bool(sender)),
        "sender_domain_length": len(sender_domain),
        "sender_is_numeric_heavy": int(
            bool(re.search(r"\d{3,}", sender_domain))
        ),
        "subject_length": len(subject),
        "subject_has_urgency": int(
            any(p in subject.lower() for p in _URGENCY_PHRASES)
        ),
        "subject_has_re_fwd": int(
            bool(re.match(r"^(re:|fwd?:)", subject.strip(), re.IGNORECASE))
        ),
        "subject_all_caps": int(subject.isupper() and len(subject) > 3),
        "reply_to_domain_mismatch": domain_mismatch,
        "html_content_ratio": html_ratio,
        "has_attachment_hint": int(
            bool(re.search(r"content-disposition:\s*attachment", text, re.IGNORECASE))
        ),
    }


class URLFeatureTransformer(BaseEstimator, TransformerMixin):
    """sklearn transformer: converts a DataFrame urls column into a dense feature matrix."""

    def __init__(self, url_col: str = "urls"):
        self.url_col = url_col

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        rows = [
            aggregate_url_features(urls if isinstance(urls, list) else [])
            for urls in X[self.url_col]
        ]
        return pd.DataFrame(rows).values.astype(float)

    def get_feature_names_out(self, input_features=None):
        dummy = aggregate_url_features([])
        return np.array(list(dummy.keys()))


class TextStatFeatureTransformer(BaseEstimator, TransformerMixin):
    """sklearn transformer: converts a DataFrame text column into statistical features."""

    def __init__(self, text_col: str = "text"):
        self.text_col = text_col

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        rows = [extract_text_features(t) for t in X[self.text_col]]
        return pd.DataFrame(rows).values.astype(float)

    def get_feature_names_out(self, input_features=None):
        dummy = extract_text_features("")
        return np.array(list(dummy.keys()))


class StructuralFeatureTransformer(BaseEstimator, TransformerMixin):
    """sklearn transformer: extracts email header/structural features from a DataFrame."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        rows = [extract_structural_features(row) for _, row in X.iterrows()]
        return pd.DataFrame(rows).values.astype(float)

    def get_feature_names_out(self, input_features=None):
        dummy = extract_structural_features(pd.Series({"type": "email"}))
        return np.array(list(dummy.keys()))


class TfidfTextTransformer(BaseEstimator, TransformerMixin):
    """TfidfVectorizer wrapper that reads from a named DataFrame column."""

    def __init__(
        self,
        text_col: str = "text",
        max_features: int = 10_000,
        ngram_range: tuple = (1, 2),
        analyzer: str = "word",
    ):
        self.text_col = text_col
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.analyzer = analyzer
        self._vectorizer: Optional[TfidfVectorizer] = None

    def fit(self, X: pd.DataFrame, y=None):
        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            analyzer=self.analyzer,
            sublinear_tf=True,
            strip_accents="unicode",
            stop_words="english" if self.analyzer == "word" else None,
        )
        self._vectorizer.fit(X[self.text_col].fillna(""))
        return self

    def transform(self, X: pd.DataFrame):
        if self._vectorizer is None:
            raise RuntimeError("Transformer has not been fitted yet.")
        return self._vectorizer.transform(X[self.text_col].fillna(""))

    def get_feature_names_out(self, input_features=None):
        if self._vectorizer is None:
            return np.array([])
        return self._vectorizer.get_feature_names_out()


def build_feature_matrix(
    df: pd.DataFrame,
    *,
    include_url: bool = True,
    include_text_stats: bool = True,
    include_structural: bool = True,
    include_tfidf: bool = False,
    scale: bool = True,
) -> np.ndarray:
    """Extract and optionally scale all hand-crafted feature groups from a preprocessed DataFrame.

    TF-IDF is excluded by default (sparse matrix); use TfidfTextTransformer separately.
    """
    if include_tfidf:
        raise NotImplementedError(
            "TF-IDF features return a sparse matrix. "
            "Use TfidfTextTransformer and combine with scipy.sparse.hstack."
        )

    parts: list[np.ndarray] = []

    if include_url:
        parts.append(URLFeatureTransformer().fit_transform(df))

    if include_text_stats:
        parts.append(TextStatFeatureTransformer().fit_transform(df))

    if include_structural:
        parts.append(StructuralFeatureTransformer().fit_transform(df))

    if not parts:
        raise ValueError("At least one feature group must be enabled.")

    X = np.hstack(parts)

    if scale:
        X = StandardScaler().fit_transform(X)

    return X
