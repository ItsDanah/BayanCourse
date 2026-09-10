"""Lab 6: generate EVALUATION_REPORT.md + model-card evidence."""

from pathlib import Path

import pandas as pd
from jinja2 import Template

from bayan.evaluation.bootstrap import bootstrap_ci
from bayan.evaluation.slices import sliced_report


PREDICTIONS_PATH = "data/eval/validation_predictions.csv"
REPORT_PATH = "EVALUATION_REPORT.md"
TEMPLATE_PATH = "templates/model_card.md.j2"
MODEL_CARD_DIR = Path("docs/model_cards")


def markdown_table(rows, headers):
    lines = []

    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    for row in rows:
        lines.append(
            "| " + " | ".join(str(value) for value in row) + " |"
        )

    return "\n".join(lines)


def main():
    df = pd.read_csv(PREDICTIONS_PATH)

    # Correct / incorrect indicator
    df["correct"] = (df["y_true"] == df["y_pred"]).astype(int)

    # -------------------------------------------------
    # Overall accuracy + bootstrap confidence interval
    # -------------------------------------------------

    accuracy, lo, hi = bootstrap_ci(
        df["correct"].tolist(),
        n_boot=2000,
        seed=42,
    )

    # -------------------------------------------------
    # Slice report
    # -------------------------------------------------

    slice_input = df.rename(
        columns={
            "y_true": "topic",
        }
    )

    slices = sliced_report(
        slice_input,
        target_col="correct",
        language_col="lang",
        dialect_col="dialect_region",
        class_col="topic",
        min_slice_size=30,
    )

    slice_rows = []

    for slice_type, groups in slices.items():
        for name, result in groups.items():
            slice_rows.append(
                [
                    slice_type,
                    name,
                    result["n"],
                    f"{result['score']:.4f}",
                    "Yes" if result["small_slice"] else "No",
                ]
            )

    slices_table = markdown_table(
        slice_rows,
        [
            "Slice",
            "Group",
            "N",
            "Score",
            "Small slice",
        ],
    )

    # -------------------------------------------------
    # Class errors
    # -------------------------------------------------

    errors = df[df["correct"] == 0]

    error_pairs = (
        errors.groupby(["y_true", "y_pred"])
        .size()
        .sort_values(ascending=False)
    )

    error_rows = []

    for (true_label, pred_label), count in error_pairs.head(10).items():
        error_rows.append(
            [
                true_label,
                pred_label,
                int(count),
            ]
        )

    if error_rows:
        error_table = markdown_table(
            error_rows,
            [
                "True class",
                "Predicted class",
                "Count",
            ],
        )
    else:
        error_table = "No validation errors found."

    # -------------------------------------------------
    # Manager headline
    # -------------------------------------------------

    headline = (
        f"Overall validation accuracy is {accuracy:.1%}, "
        f"with a 95% bootstrap confidence interval from "
        f"{lo:.1%} to {hi:.1%}. "
        "Slice-level results should be interpreted carefully, "
        "especially where sample sizes are small."
    )

    # -------------------------------------------------
    # Main evaluation report
    # -------------------------------------------------

    report = f"""# Bayan Evaluation Report

## Executive Summary

{headline}

## Overall Performance

- Validation accuracy: {accuracy:.4f}
- 95% bootstrap CI: [{lo:.4f}, {hi:.4f}]
- Validation examples: {len(df)}
- Validation errors: {len(errors)}

## Sliced Evaluation

{slices_table}

## Most Common Prediction Errors

{error_table}

## Behavioural Tests

Behavioural test results should be added after running the invariance,
directional, and minimum-functionality suites.

Targets:

- Invariance: >= 95%
- MFT: >= 90%

## Manual Error Taxonomy

120 validation errors must be manually reviewed and tagged using
`docs/ERROR_TAXONOMY.md`.

Record the final category histogram and the top three prioritised fixes here
after completing the manual review.

## Top 3 Prioritised Fixes

TODO — complete after manual error review.

## Model Cards

Three model cards are generated in `docs/model_cards/`.
Known limitations must be completed manually.
"""

    Path(REPORT_PATH).write_text(
        report,
        encoding="utf-8",
    )

    # -------------------------------------------------
    # Model cards
    # -------------------------------------------------

    MODEL_CARD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    template_text = Path(TEMPLATE_PATH).read_text(
        encoding="utf-8"
    )

    template = Template(template_text)

    metrics_table = markdown_table(
        [
            [
                "Validation accuracy",
                f"{accuracy:.4f}",
            ],
            [
                "95% CI lower",
                f"{lo:.4f}",
            ],
            [
                "95% CI upper",
                f"{hi:.4f}",
            ],
        ],
        [
            "Metric",
            "Value",
        ],
    )

    behavioural_table = (
        "| Test | Result |\n"
        "|---|---|\n"
        "| Invariance | TODO |\n"
        "| Directional | TODO |\n"
        "| MFT | TODO |"
    )

    models = [
        {
            "model_name": "Topic Classifier",
            "checkpoint": "trained Bayan topic classifier",
            "intended_use": (
                "Classify bilingual municipal feedback into service topics."
            ),
        },
        {
            "model_name": "NER Model",
            "checkpoint": "xlm-roberta-base",
            "intended_use": (
                "Extract entities such as locations, dates, references, "
                "and services from Bayan text."
            ),
        },
        {
            "model_name": "Semantic Search",
            "checkpoint": (
                "sentence-transformers/"
                "paraphrase-multilingual-MiniLM-L12-v2"
            ),
            "intended_use": (
                "Retrieve semantically similar historical Bayan cases."
            ),
        },
    ]

    for model in models:
        content = template.render(
            model_name=model["model_name"],
            intended_use=model["intended_use"],
            checkpoint=model["checkpoint"],
            preproc_version="v1",
            data_version="Bayan course dataset",
            metrics_table=metrics_table,
            slices_table=slices_table,
            behavioural_table=behavioural_table,
        )

        filename = (
            model["model_name"]
            .lower()
            .replace(" ", "_")
            + ".md"
        )

        (MODEL_CARD_DIR / filename).write_text(
            content,
            encoding="utf-8",
        )

    print(f"Generated {REPORT_PATH}")
    print(f"Generated 3 model cards in {MODEL_CARD_DIR}")


if __name__ == "__main__":
    main()