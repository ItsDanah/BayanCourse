# Bayan | بيان

## Natural Language Processing with Transformers

**Bayan (بيان)** is a bilingual Natural Language Processing project designed to process and analyze Arabic and English citizen feedback. The project was developed as part of the **SDA-AIE-211 — Natural Language Processing with Transformers** course.

The project explores the complete NLP workflow, from text preprocessing and tokenization to Transformer models, classification, Named Entity Recognition (NER), semantic search, evaluation, optimization, and deployment.

## Project Overview

Bayan processes bilingual Arabic and English text through a structured NLP pipeline. The project focuses on building reliable preprocessing methods, understanding Transformer architecture, training task-specific models, and evaluating and optimizing their performance.

### Main Components

* **Bilingual Text Preprocessing** — Normalizes Arabic and English text, handles whitespace and repeated characters, and masks personally identifiable information (PII).
* **Tokenizer Analysis** — Compares multilingual and Arabic-focused tokenizers using measurements such as token fertility and sequence length to support model selection.
* **Transformer Attention** — Implements scaled dot-product attention and Multi-Head Attention while exploring masking, attention weights, and Transformer parameter distribution.
* **Topic Classification** — Builds a TF-IDF + LinearSVC baseline and fine-tunes a Transformer-based classifier for categorizing citizen feedback into different service topics.
* **Named Entity Recognition (NER)** — Fine-tunes a token-classification model using BIO label alignment to identify entities such as locations, dates, services, and references.
* **Question Answering** — Supports extractive question answering over provided textual information.
* **Semantic Search** — Retrieves relevant historical cases using vector representations, FAISS similarity search, and reranking.
* **Model Evaluation** — Evaluates NLP models using performance metrics, behavioural tests, data slices, confidence intervals, and error analysis.
* **Model Optimization & Serving** — Optimizes trained models using ONNX and INT8 quantization, benchmarks inference performance, and serves the optimized classifier through a FastAPI API with startup canaries and health checks.

## Results

* Topic classification achieved **1.000 macro-F1** on the frozen test set.
* NER achieved **1.000 entity-level F1** on the frozen test set.
* Semantic search achieved **1.000 Recall@10** and **1.000 MRR@10** with and without reranking.
* INT8 classifier inference achieved **23.45 ms p99 latency**, compared with **536.05 ms** for the original FP32 padded baseline.
* INT8 quantization preserved classifier macro-F1 at **1.0000** with **0.0000 measured quality loss**.
* The final automated test suite completed with **78 passing tests**.

## Project Structure

```
├── data/
│   ├── eval/                 # Evaluation datasets and behavioural test data
│   ├── models/               # NER and QA model datasets
│   ├── raw/                  # Raw citizen-feedback datasets
│   ├── search/               # Semantic search datasets and results
│   ├── serving/              # Inference benchmark and load-test configuration
│   └── DATA_DICTIONARY.md
│
├── docs/
│   ├── model_cards/          # Model cards for trained NLP components
│   ├── CAPSTONE_CHECKLIST.md
│   ├── ERROR_TAXONOMY.md
│   └── LABS.md
│
├── notebooks/
│   ├── 00_colab_setup.ipynb
│   ├── 01_tokenizer_audit.py
│   ├── 02_transformer_anatomy.py
│   └── 05_retrieval_eval.py
│
├── scripts/                  # Training, evaluation, optimization, and utility scripts
│
├── src/
│   └── bayan/
│       ├── evaluation/       # Behavioural tests, bootstrap CIs, and evaluation slices
│       ├── models/           # Topic classification, NER, and QA components
│       ├── preprocessing/    # Text normalization, PII masking, and segmentation
│       ├── search/           # Semantic search and retrieval components
│       ├── serving/          # FastAPI serving and startup canaries
│       └── attention.py      # Attention and Multi-Head Attention implementation
│
├── templates/
│   └── model_card.md.j2      # Model card template
│
├── tests/                    # Automated test suite
│
├── BENCHMARKS.md             # Experimental results and performance metrics
├── DECISIONS.md              # Model and tokenizer decisions
├── EVALUATION_REPORT.md      # Evaluation findings and analysis
├── NOTES.md                  # Technical observations and experiment notes
├── Makefile                  # Project commands
├── pyproject.toml            # Python project configuration
├── requirements.txt          # Python dependencies
└── README.md
```


## Technologies

Python, PyTorch, Hugging Face Transformers, ONNX Runtime, FastAPI, FAISS, spaCy, CAMeL Tools, scikit-learn, Datasets, and seqeval.

## Purpose

The purpose of Bayan is to apply the concepts covered in the NLP with Transformers course through practical implementation. Each component contributes to a complete bilingual NLP system while demonstrating preprocessing, Transformer architecture, fine-tuning, semantic search, evaluation, optimization, and deployment techniques.
