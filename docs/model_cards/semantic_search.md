# Model Card — Semantic Search

## Intended use
Retrieve semantically similar historical Bayan cases.

## Artefact / data versions
- Model/checkpoint: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- Preprocessing version: v1
- Data version/snapshot: Bayan course dataset

## Metrics
| Metric | Value |
|---|---|
| Validation accuracy | 0.8750 |
| 95% CI lower | 0.8617 |
| 95% CI upper | 0.8888 |

## Slice metrics
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

## Behavioural tests
| Test | Result |
|---|---|
| Invariance | TODO |
| Directional | TODO |
| MFT | TODO |

## Known limitations
<!-- Lab 6: write this section by hand. Do not auto-generate it. -->
TODO

## Contact / owner
TODO