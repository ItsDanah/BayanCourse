# BENCHMARKS

> Fill these tables from **your own runs**. Do not copy course reference numbers.

## Lab 1 — Tokenizer audit
| Tokenizer | AR fertility | EN fertility | AR p95 len | EN p95 len | AR UNK rate |
|---|---:|---:|---:|---:|---:|
| mBERT | 2.153 | 1.510 | 27.0 | 25.0 | 0.453% |
| XLM-R | 1.672 | 1.434 | 21.0 | 23.0 | 0.000% |
| CAMeLBERT | 1.405 | 2.705 | 20.0 | 38.0 | 0.804% |
| DistilBERT | 4.527 | 1.298 | 47.0 | 21.0 | 0.215% |

- Golden preprocessing: 25 / 25 passed
- PII masking recall: 60 / 60 = 100%

## Lab 3 — Models
| Model | Metric | Validation | Frozen test | Train time |
|---|---|---:|---:|---:|
| TF-IDF + LinearSVC | macro-F1 | | 1.000 | |
| Topic classifier | macro-F1 | | 1.000 | |
| NER | entity-F1 | | | |
| QA | span/null smoke | | | |

### Lab 3A — TF-IDF + LinearSVC baseline

- TF-IDF + LinearSVC baseline macro-F1: 1.0000
- TF-IDF + LinearSVC baseline accuracy: 1.0000
- Topic classifier frozen-test macro-F1: 1.0000
- Topic classifier frozen-test accuracy: 1.0000
- Improvement over baseline: +0.0000

## Lab 3A — Fine-tune topic classifier

Frozen Test Metrics
test_loss: 0.0006405675085261464
test_accuracy: 1.0
test_macro_f1: 1.0
test_runtime: 27.5726
test_samples_per_second: 65.79
test_steps_per_second: 8.233
epoch: 3.0

## Lab 4 — Arabic model bake-off
| Checkpoint | macro-F1 all | Gulf | MSA | AR fertility |
|---|---:|---:|---:|---:|
| multilingual incumbent | | | | |
| Arabic dialect-aware | | | | |
| optional third model | | | | |

## Lab 5 — Search
| Configuration | recall@10 | MRR@10 | p50 latency/query |
|---|---:|---:|---:|
| bi-encoder only | | | |
| + cross-encoder rerank | | | |
| cross-lingual slice | | | |

- no-answer empty-correct: ___ / 20
- cross-lingual gap: ___

## Lab 6 — Evaluation
| Model | Aggregate macro-F1 [CI] | Gulf [CI] | Invariance pass | MFT pass |
|---|---|---|---:|---:|
| topic classifier | | | | |
| dialect-aware | | | | |

- paired comparison verdict:
- error taxonomy top categories:
- top-3 prioritised fixes:

## Lab 7 — Optimisation ladder
| Rung | p50 | p99 | quality metric / paired Δ | Artefact size |
|---|---:|---:|---|---:|
| fp32 torch @512 padded | | | | |
| fp32 torch @128 dynamic | | | | |
| ONNX fp32 @128 | | | | |
| ONNX INT8 @128 | | | | |

- HTTP p99, 16 concurrent:
- classifier quantisation decision:
- NER quantisation decision:
