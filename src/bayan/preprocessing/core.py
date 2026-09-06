"""Lab 1 starter: versioned bilingual preprocessing for Bayan."""

import re
import unicodedata

PREPROC_VERSION = "1.2.0"

_TATWEEL = "\u0640"
_WHITESPACE_RE = re.compile(r"\s+")
_REPEAT_RE = re.compile(r"(.)\1{2,}")

_PHONE_RE = re.compile(r"(?<!\d)(?:\+?966|0)5\d{8}(?!\d)")
_NATIONAL_ID_RE = re.compile(r"(?<!\d)[12]\d{9}(?!\d)")

def normalize(text: str) -> str:
    """Return deterministic Bayan normalisation while preserving task signal."""
    # TODO(Lab 1): implement the course normalisation contract.
    
    text = unicodedata.normalize("NFKC", text)
    text = text.replace(_TATWEEL, "")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    text = _REPEAT_RE.sub(r"\1\1", text)
    return text
    # raise NotImplementedError("Implement normalize() in Lab 1")


def mask_pii(text: str) -> str:
    """Mask supported phone numbers and Saudi national-ID-shaped values."""
    # TODO(Lab 1): replace supported PII with <PHONE> / <NATIONAL_ID>.
    text = _PHONE_RE.sub("<PHONE>", text)
    text = _NATIONAL_ID_RE.sub("<NATIONAL_ID>", text)
    return text
    # raise NotImplementedError("Implement mask_pii() in Lab 1")


def preprocess(text: str) -> str:
    """Apply the shared train/eval/serve preprocessing contract."""
    # TODO(Lab 1): compose masking and normalisation in the intended order.
    return normalize(mask_pii(text))
    # raise NotImplementedError("Implement preprocess() in Lab 1")
