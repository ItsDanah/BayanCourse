"""Lab 1 starter: audit four tokenizer candidates on Bayan AR/EN text."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import AutoTokenizer


CANDIDATES = {
    "bert-base-multilingual-cased": "mBERT",
    "xlm-roberta-base": "XLM-R",
    "CAMeL-Lab/bert-base-arabic-camelbert-mix": "CAMeLBERT",
    "distilbert-base-uncased": "DistilBERT",
}

DATA = Path("data/raw/bayan_feedback.csv")


def fertility(tokenizer, texts) -> float:
    """Return the total subword pieces per whitespace word."""
    total_pieces = 0
    total_words = 0

    for text in texts:
        text = str(text)
        total_pieces += len(
            tokenizer.encode(text, add_special_tokens=False)
        )
        total_words += len(text.split())

    if total_words == 0:
        return 0.0

    return total_pieces / total_words


def main():
    """Audit fertility and sequence lengths for AR and EN text."""
    data = pd.read_csv(DATA)

    arabic_texts = (
        data.loc[data["lang"].str.lower() == "ar", "text"]
        .dropna()
        .astype(str)
        .tolist()
    )

    english_texts = (
        data.loc[data["lang"].str.lower() == "en", "text"]
        .dropna()
        .astype(str)
        .tolist()
    )

    texts_by_language = {
        "AR": arabic_texts,
        "EN": english_texts,
    }

    results = []
    figure, axes = plt.subplots(4, 2, figsize=(12, 14))

    for row, (model_id, model_name) in enumerate(CANDIDATES.items()):
        print(f"Loading {model_name}...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        for column, (language, texts) in enumerate(
            texts_by_language.items()
        ):
            lengths = [
                len(
                    tokenizer.encode(
                        text,
                        add_special_tokens=True,
                        truncation=False,
                    )
                )
                for text in texts
            ]

            score = fertility(tokenizer, texts)
            p95 = float(np.percentile(lengths, 95))

            results.append(
                {
                    "tokenizer": model_name,
                    "language": language,
                    "fertility": score,
                    "p95": p95,
                }
            )

            axes[row, column].hist(
                lengths,
                bins=15,
                edgecolor="black",
            )
            axes[row, column].set_title(
                f"{model_name} - {language}"
            )
            axes[row, column].set_xlabel("Sequence length")
            axes[row, column].set_ylabel("Number of texts")

    print("\n| Tokenizer | Language | Fertility | P95 length |")
    print("|---|---|---:|---:|")

    for result in results:
        print(
            f"| {result['tokenizer']} "
            f"| {result['language']} "
            f"| {result['fertility']:.3f} "
            f"| {result['p95']:.1f} |"
        )

    output = Path("artifacts/tokenizer_sequence_lengths.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)

    print(f"\nHistogram saved to: {output}")


if __name__ == "__main__":
    main()