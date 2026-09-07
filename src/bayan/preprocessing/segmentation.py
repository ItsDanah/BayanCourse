"""Lab 1 starter: sentence segmentation."""

import spacy
from spacy.symbols import ORTH

from .core import preprocess

def build_pipeline():
    # TODO(Lab 1): build the spaCy segmentation pipeline.
    nlp = spacy.blank("xx")

    abbreviations = (
        "Dr.",
        "Mr.",
        "Mrs.",
        "Ms.",
        "Prof.",
        "e.g.",
        "i.e.",
        "د.",
    )

    for abbreviation in abbreviations:
        nlp.tokenizer.add_special_case(
            abbreviation,
            [{ORTH: abbreviation}],
        )

    nlp.add_pipe(
        "sentencizer",
        config={"punct_chars": [".", "!", "?", "؟", "…"]},
    )

    return nlp
    # raise NotImplementedError("Implement build_pipeline() in Lab 1")


def split_sentences(raw: str, nlp) -> list[str]:
    # TODO(Lab 1): preprocess then return non-empty sentence strings.
    cleaned_text = preprocess(raw)
    document = nlp(cleaned_text)

    return [
        sentence.text.strip()
        for sentence in document.sents
        if sentence.text.strip()
    ]
    # raise NotImplementedError("Implement split_sentences() in Lab 1")
