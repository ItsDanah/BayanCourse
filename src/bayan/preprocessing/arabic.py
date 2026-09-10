"""Lab 4 starter: per-model Arabic normalisation profiles."""

from dataclasses import dataclass
import re

from camel_tools.disambig.mle import MLEDisambiguator
from camel_tools.tokenizers.morphological import MorphologicalTokenizer
from camel_tools.tokenizers.word import simple_word_tokenize


@dataclass(frozen=True)
class ArabicProfile:
    name: str
    dediacritize: bool = False


def normalize_arabic(text: str, profile: ArabicProfile) -> str:
    normalized = text

    normalized = normalized.replace("ـ", "")

    if profile.dediacritize:
        normalized = re.sub(r"[\u064B-\u065F\u0670]", "", normalized)

    if profile.name == "bayan_ar_v1":
        normalized = re.sub(r"[إأآ]", "ا", normalized)
        normalized = normalized.replace("ؤ", "و")
        normalized = normalized.replace("ئ", "ي")
        normalized = normalized.replace("ى", "ي")
        normalized = normalized.replace("ة", "ه")

    return normalized


_mle = None
_tokenizer = None


def segment(text: str) -> list[str]:
    global _mle, _tokenizer

    if _mle is None:
        _mle = MLEDisambiguator.pretrained("calima-msa-r13")

    if _tokenizer is None:
        _tokenizer = MorphologicalTokenizer(
            disambiguator=_mle,
            scheme="d3tok",
            split=True,
            diac=False,
        )

    words = simple_word_tokenize(text)

    segmented_words = _tokenizer.tokenize(words)

    result = []

    for word in segmented_words:
        if isinstance(word, list):
            result.extend(word)
        else:
            result.extend(word.split("_"))

    return result