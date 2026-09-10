"""Lab 6: sliced evaluation report."""

import pandas as pd


def sliced_report(
    data,
    *,
    target_col="correct",
    language_col="lang",
    dialect_col="dialect_region",
    class_col="topic",
    text_col="text",
    min_slice_size=30,
):
    """
    Compute accuracy-style metrics across useful evaluation slices.

    Returns a dictionary containing language, dialect, class, and length slices.
    Small slices are flagged when their sample size is below min_slice_size.
    """

    df = pd.DataFrame(data).copy()

    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    def summarize(group):
        n = len(group)

        return {
            "n": int(n),
            "score": float(group[target_col].mean()),
            "small_slice": n < min_slice_size,
        }

    report = {}

    # Language slices
    if language_col in df.columns:
        report["language"] = {
            str(name): summarize(group)
            for name, group in df.groupby(language_col)
        }

    # Dialect slices
    if dialect_col in df.columns:
        report["dialect"] = {
            str(name): summarize(group)
            for name, group in df.groupby(dialect_col)
        }

    # Class/topic slices
    if class_col in df.columns:
        report["class"] = {
            str(name): summarize(group)
            for name, group in df.groupby(class_col)
        }

    # Length slices
    if text_col in df.columns:
        df["_length"] = df[text_col].fillna("").astype(str).str.split().str.len()

        df["_length_bucket"] = pd.cut(
            df["_length"],
            bins=[-1, 10, 25, float("inf")],
            labels=["short", "medium", "long"],
        )

        report["length"] = {
            str(name): summarize(group)
            for name, group in df.groupby(
                "_length_bucket",
                observed=True,
            )
        }

    return report