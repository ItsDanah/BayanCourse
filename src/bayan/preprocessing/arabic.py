"""Lab 4 starter: per-model Arabic normalisation profiles."""
from dataclasses import dataclass
import re

@dataclass(frozen=True)
class ArabicProfile:
    name: str
    dediacritize: bool = False


def normalize_arabic(text: str, profile: ArabicProfile) -> str:
    normalized = text

    # Remove tatweel / kashida
    normalized = normalized.replace("ـ", "")

    # Remove Arabic diacritics when requested
    if profile.dediacritize:
        normalized = re.sub(r"[\u064B-\u065F\u0670]", "", normalized)

    if profile.name == "bayan_ar_v1":
        # Normalize Alef forms
        normalized = re.sub(r"[إأآ]", "ا", normalized)

        # Normalize hamza-on-waw
        normalized = normalized.replace("ؤ", "و")

        # Normalize hamza-on-ya
        normalized = normalized.replace("ئ", "ي")

        # Alef maqsura -> ya
        normalized = normalized.replace("ى", "ي")

        # Ta marbuta -> ha
        normalized = normalized.replace("ة", "ه")

    return normalized


def segment(text: str) -> list[str]:
    # TODO(Lab 4): wire the chosen CAMeL Tools clitic segmentation scheme.
    raise NotImplementedError
