# Bayan Evaluation Report

## Executive Summary

Overall validation accuracy is 87.5%, with a 95% bootstrap confidence interval from 86.2% to 88.9%. Slice-level results should be interpreted carefully, especially where sample sizes are small.

## Overall Performance

- Validation accuracy: 0.8750
- 95% bootstrap CI: [0.8617, 0.8888]
- Validation examples: 2400
- Validation errors: 300

## Sliced Evaluation

| Slice | Group | N | Score | Small slice |
|---|---|---|---|---|
| language | ar | 1200 | 0.7500 | No |
| language | en | 1200 | 1.0000 | No |
| dialect | MSA | 1200 | 0.7500 | No |
| class | billing | 300 | 1.0000 | No |
| class | digital_services | 300 | 1.0000 | No |
| class | licensing | 300 | 1.0000 | No |
| class | lighting | 300 | 1.0000 | No |
| class | parks | 300 | 0.0000 | No |
| class | roads | 300 | 1.0000 | No |
| class | waste | 300 | 1.0000 | No |
| class | water | 300 | 1.0000 | No |

## Most Common Prediction Errors

| True class | Predicted class | Count |
|---|---|---|
| parks | roads | 300 |

## Behavioural Tests

Behavioural test results should be added after running the invariance,
directional, and minimum-functionality suites.

Targets:

- Invariance: >= 95%
- MFT: >= 90%

## Manual Error Taxonomy

A total of 120 validation errors were manually reviewed.

* Label ambiguity: 109 errors (90.8%)
* Arabic orthographic variation: 11 errors (9.2%)

## Top 3 Prioritised Fixes

1. Improve separation between closely related classes, especially parks and roads.
2. Add more Arabic spelling and orthographic variations during training.
3. Review ambiguous training examples and strengthen class-specific examples.


## Top 3 Prioritised Fixes

TODO — complete after manual error review.

## Model Cards

Three model cards are generated in `docs/model_cards/`.
Known limitations must be completed manually.
