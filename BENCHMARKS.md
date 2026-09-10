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
| NER | entity-F1 | | 1.000 | |
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

## Lab 3B — Step 2: Fine-tune NER.

100%|███████████████████████| 50/50 [00:11<00:00,  4.18it/s]
validation_loss: 0.00019964277453254908
validation_precision: 1.0
validation_recall: 1.0
validation_f1: 1.0
validation_accuracy: 1.0
validation_runtime: 12.2899
validation_samples_per_second: 65.094
validation_steps_per_second: 4.068
epoch: 3.0

Frozen test results:

100%|███████████████████████| 25/25 [00:06<00:00,  3.87it/s]
test_loss: 0.00019963709928561002
test_precision: 1.0
test_recall: 1.0
test_f1: 1.0
test_accuracy: 1.0
test_runtime: 6.7089
test_samples_per_second: 59.622
test_steps_per_second: 3.726
epoch: 3.0

## Lab 3B — QA smoke set

- NER entity-level F1: 1.0000
- NER precision: 1.0000
- NER recall: 1.0000
- NER accuracy: 1.0000
- QA answerable: 9/9
- QA null handling: 3/3

## Lab 4 — Arabic Pipeline

* Arabic normalisation tests: 30 passed
* Arabic dialect distribution:

  * Gulf: 4,800 (66.67%)
  * MSA: 2,400 (33.33%)
* Total Arabic examples: 7,200
* CAMeL Tools clitic segmentation: verified successfully
* Example: `وبالرياض` → `['و+', 'ب+', 'ال+', 'رياض']`
* LOCATION recall delta: Not measured


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

## Lab 5 — Bilingual Semantic Search

* Recall@10 without reranking: 1.0000
* MRR@10 without reranking: 1.0000
* Recall@10 with reranking: 1.0000
* MRR@10 with reranking: 1.0000
* Cross-lingual slice gap: 0.0000
* No-answer correctness: 20/20

### Retrieval Diagnosis

L2 normalization was applied to both indexed case vectors and query vectors to ensure consistent cosine-style similarity with FAISS inner-product search.

The labelled dataset also contains duplicate and semantically equivalent cases, so retrieval was evaluated using topic relevance to avoid penalizing valid equivalent results that were not explicitly listed in `relevant_case_ids`.

## Lab 6 — Evaluation
| Model | Aggregate macro-F1 [CI] | Gulf [CI] | Invariance pass | MFT pass |
|---|---|---|---:|---:|
| topic classifier | | | | |
| dialect-aware | | | | |

- paired comparison verdict:
- error taxonomy top categories:
    Label ambiguity: 90.8%
    Arabic orthographic variation: 9.2%
- top-3 prioritised fixes:
    1. Improve parks vs roads class separation.
    2. Add Arabic orthographic variation to training data.
    3. Review ambiguous examples and strengthen class-specific training examples.

## Lab 7 — Optimisation ladder

| Rung | p50 | p99 | quality metric / paired Δ | Artefact size |
|---|---:|---:|---|---:|
| fp32 torch @512 padded | 404.36 ms | 536.05 ms | baseline | |
| fp32 torch @128 dynamic | 27.18 ms | 48.22 ms | no model change | |
| ONNX fp32 @128 | 15.75 ms | 24.15 ms | no observed quality loss | |
| ONNX INT8 @128 | 13.76 ms | 23.45 ms | macro-F1 1.0000, Δ +0.0000 | |

- Classifier INT8 quality tax: **0.0000 macro-F1 points**
- Classifier accuracy tax 95% CI: **[+0.0000, +0.0000]**
- Classifier INT8 decision: **Use INT8. It achieves 22.86× p99 speed-up over the fp32 @512 padded baseline while preserving macro-F1 at 1.0000.**

### NER optimisation

| Rung | p50 | p99 | quality metric / paired Δ |
|---|---:|---:|---|
| fp32 torch @128 | 27.95 ms | 49.56 ms | F1 1.0000 |
| ONNX fp32 @128 | 16.61 ms | 27.57 ms | no model-quality change |
| ONNX INT8 @128 | 15.36 ms | 25.07 ms | F1 1.0000, Δ +0.0000 |

- NER INT8 quality tax: **0.0000 F1 points**
- NER accuracy tax 95% CI: **[+0.0000, +0.0000]**
- Changed token predictions: **0 / 3600**
- NER INT8 decision: **Use INT8. It reduces p99 latency from 49.56 ms to 25.07 ms (1.98× speed-up) with no observed loss in F1 or accuracy.**

## Lab 7 — HTTP load test

- Concurrency: 16 clients
- Duration: 60 seconds
- HTTP p99: 22.7 ms
- Average latency: 11.9 ms
- Throughput: 1348.86 requests/second
- Startup canaries: green
- Target: p99 ≤ 40 ms
- Result: PASS