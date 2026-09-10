"""Lab 7: ONNX export, INT8 quantisation, and quality checks."""

from pathlib import Path
import shutil

import numpy as np
import torch

from optimum.onnxruntime import (
    ORTModelForSequenceClassification,
    ORTModelForTokenClassification,
    ORTQuantizer,
)
from optimum.onnxruntime.configuration import AutoQuantizationConfig

from sklearn.metrics import (
    accuracy_score as classifier_accuracy,
    f1_score as classifier_f1,
)

from seqeval.metrics import (
    accuracy_score as ner_accuracy,
    f1_score as ner_f1,
)

from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from bayan.evaluation.bootstrap import paired_bootstrap_diff
from bayan.models.data import build_topic_dataset
from bayan.models.ner import align_labels


# --------------------------------------------------
# Paths
# --------------------------------------------------

CLASSIFIER_PATH = "artifacts/topic_classifier"
NER_PATH = "artifacts/ner"

NER_DATA_PATH = "data/models/bayan_ner.conll"

CLASSIFIER_FP32_DIR = Path(
    "artifacts/onnx_classifier_fp32"
)

CLASSIFIER_INT8_DIR = Path(
    "artifacts/onnx_classifier_int8"
)

NER_FP32_DIR = Path(
    "artifacts/onnx_ner_fp32"
)

NER_INT8_DIR = Path(
    "artifacts/onnx_ner_int8"
)


# --------------------------------------------------
# Classifier prediction helpers
# --------------------------------------------------

def predict_torch(model, tokenizer, texts):
    predictions = []

    for text in texts:
        inputs = tokenizer(
            str(text),
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )

        with torch.no_grad():
            logits = model(**inputs).logits

        predictions.append(
            int(torch.argmax(logits, dim=-1).item())
        )

    return np.array(predictions)


def predict_onnx(model, tokenizer, texts):
    predictions = []

    for text in texts:
        inputs = tokenizer(
            str(text),
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )

        logits = model(**inputs).logits

        predictions.append(
            int(np.argmax(logits, axis=-1)[0])
        )

    return np.array(predictions)


# --------------------------------------------------
# Classifier export
# --------------------------------------------------

def export_classifier():
    print("Exporting classifier to ONNX fp32...")

    CLASSIFIER_FP32_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CLASSIFIER_INT8_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        CLASSIFIER_PATH
    )

    ort_model = (
        ORTModelForSequenceClassification
        .from_pretrained(
            CLASSIFIER_PATH,
            export=True,
        )
    )

    ort_model.save_pretrained(
        CLASSIFIER_FP32_DIR
    )

    tokenizer.save_pretrained(
        CLASSIFIER_FP32_DIR
    )

    fp32_model = (
        CLASSIFIER_FP32_DIR / "model.onnx"
    )

    rollback_model = (
        CLASSIFIER_FP32_DIR
        / "model_fp32_rollback.onnx"
    )

    if fp32_model.exists():
        shutil.copy2(
            fp32_model,
            rollback_model,
        )

    print(
        "Saved classifier fp32 rollback artefact to: "
        f"{CLASSIFIER_FP32_DIR}"
    )

    print("\nQuantising classifier to INT8...")

    quantizer = ORTQuantizer.from_pretrained(
        ort_model
    )

    qconfig = AutoQuantizationConfig.avx2(
        is_static=False,
        per_channel=False,
    )

    quantizer.quantize(
        save_dir=CLASSIFIER_INT8_DIR,
        quantization_config=qconfig,
    )

    tokenizer.save_pretrained(
        CLASSIFIER_INT8_DIR
    )

    print(
        "Saved classifier INT8 artefact to: "
        f"{CLASSIFIER_INT8_DIR}"
    )

    return tokenizer


# --------------------------------------------------
# Classifier quality
# --------------------------------------------------

def check_classifier_quality(tokenizer):
    print(
        "\nLoading frozen classifier test split..."
    )

    dataset = build_topic_dataset()

    topics = sorted(
        set(dataset["train"]["topic"])
    )

    label2id = {
        label: index
        for index, label in enumerate(topics)
    }

    test = dataset["test"]

    texts = list(test["text"])

    labels = np.array([
        label2id[topic]
        for topic in test["topic"]
    ])

    print(f"Test examples: {len(texts)}")

    print("\nEvaluating FP32 classifier...")

    fp32_model = (
        AutoModelForSequenceClassification
        .from_pretrained(CLASSIFIER_PATH)
    )

    fp32_model.eval()

    fp32_predictions = predict_torch(
        fp32_model,
        tokenizer,
        texts,
    )

    print("Evaluating INT8 classifier...")

    int8_model = (
        ORTModelForSequenceClassification
        .from_pretrained(
            CLASSIFIER_INT8_DIR,
            file_name="model_quantized.onnx",
        )
    )

    int8_predictions = predict_onnx(
        int8_model,
        tokenizer,
        texts,
    )

    fp32_accuracy = classifier_accuracy(
        labels,
        fp32_predictions,
    )

    int8_accuracy = classifier_accuracy(
        labels,
        int8_predictions,
    )

    fp32_f1 = classifier_f1(
        labels,
        fp32_predictions,
        average="macro",
    )

    int8_f1 = classifier_f1(
        labels,
        int8_predictions,
        average="macro",
    )

    f1_delta = int8_f1 - fp32_f1
    accuracy_tax = fp32_accuracy - int8_accuracy

    fp32_correct = (
        fp32_predictions == labels
    ).astype(float)

    int8_correct = (
        int8_predictions == labels
    ).astype(float)

    _, ci_lo, ci_hi = paired_bootstrap_diff(
        fp32_correct,
        int8_correct,
    )

    changed = int(
        np.sum(
            fp32_predictions
            != int8_predictions
        )
    )

    print(
        "\n--- Classifier quality comparison ---"
    )

    print(
        f"FP32 accuracy: {fp32_accuracy:.4f}"
    )

    print(
        f"INT8 accuracy: {int8_accuracy:.4f}"
    )

    print(
        f"FP32 macro-F1: {fp32_f1:.4f}"
    )

    print(
        f"INT8 macro-F1: {int8_f1:.4f}"
    )

    print(
        f"Macro-F1 delta: {f1_delta:+.4f}"
    )

    print(
        f"Accuracy tax: {accuracy_tax:+.4f}"
    )

    print(
        "Accuracy tax 95% CI: "
        f"[{ci_lo:+.4f}, {ci_hi:+.4f}]"
    )

    print(
        f"Changed predictions: "
        f"{changed}/{len(labels)}"
    )

    if fp32_f1 - int8_f1 <= 0.01:
        print(
            "PASS: classifier quality tax "
            "<= 1 macro-F1 point."
        )
    else:
        print(
            "FAIL: classifier quality tax "
            "> 1 macro-F1 point."
        )


# --------------------------------------------------
# NER data
# --------------------------------------------------

def read_conll(path):
    sentences = []
    labels = []

    current_tokens = []
    current_labels = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:
            line = line.strip()

            if not line:
                if current_tokens:
                    sentences.append(
                        current_tokens
                    )

                    labels.append(
                        current_labels
                    )

                    current_tokens = []
                    current_labels = []

                continue

            token, label = line.rsplit(
                maxsplit=1
            )

            current_tokens.append(token)
            current_labels.append(label)

    if current_tokens:
        sentences.append(
            current_tokens
        )

        labels.append(
            current_labels
        )

    return sentences, labels


# --------------------------------------------------
# NER export
# --------------------------------------------------

def export_ner():
    print("\nExporting NER to ONNX fp32...")

    NER_FP32_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    NER_INT8_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        NER_PATH
    )

    ort_model = (
        ORTModelForTokenClassification
        .from_pretrained(
            NER_PATH,
            export=True,
        )
    )

    ort_model.save_pretrained(
        NER_FP32_DIR
    )

    tokenizer.save_pretrained(
        NER_FP32_DIR
    )

    print(
        f"Saved NER fp32 artefact to: "
        f"{NER_FP32_DIR}"
    )

    fp32_model = (
        NER_FP32_DIR / "model.onnx"
    )

    rollback_model = (
        NER_FP32_DIR
        / "model_fp32_rollback.onnx"
    )

    if fp32_model.exists():
        shutil.copy2(
            fp32_model,
            rollback_model,
        )

    print("\nQuantising NER to INT8...")

    quantizer = ORTQuantizer.from_pretrained(
        ort_model
    )

    qconfig = AutoQuantizationConfig.avx2(
        is_static=False,
        per_channel=False,
    )

    quantizer.quantize(
        save_dir=NER_INT8_DIR,
        quantization_config=qconfig,
    )

    tokenizer.save_pretrained(
        NER_INT8_DIR
    )

    print(
        f"Saved NER INT8 artefact to: "
        f"{NER_INT8_DIR}"
    )

    return tokenizer


# --------------------------------------------------
# NER quality
# --------------------------------------------------

def check_ner_quality(tokenizer):
    print("\nLoading frozen NER test split...")

    sentences, bio_labels = read_conll(
        NER_DATA_PATH
    )

    unique_labels = sorted({
        label
        for sentence in bio_labels
        for label in sentence
    })

    if "O" in unique_labels:
        unique_labels.remove("O")

        unique_labels = [
            "O"
        ] + unique_labels

    label2id = {
        label: i
        for i, label in enumerate(unique_labels)
    }

    id2label = {
        i: label
        for label, i in label2id.items()
    }

    numeric_labels = [
        [
            label2id[label]
            for label in sentence_labels
        ]
        for sentence_labels in bio_labels
    ]

    total = len(sentences)

    validation_end = int(
        total * 0.90
    )

    test_sentences = sentences[
        validation_end:
    ]

    test_labels = numeric_labels[
        validation_end:
    ]

    print(
        f"NER test examples: "
        f"{len(test_sentences)}"
    )

    print("\nEvaluating FP32 NER...")

    fp32_model = (
        AutoModelForTokenClassification
        .from_pretrained(NER_PATH)
    )

    fp32_model.eval()

    print("Evaluating INT8 NER...")

    int8_model = (
        ORTModelForTokenClassification
        .from_pretrained(
            NER_INT8_DIR,
            file_name="model_quantized.onnx",
        )
    )

    true_sequences = []
    fp32_sequences = []
    int8_sequences = []

    fp32_token_correct = []
    int8_token_correct = []

    for tokens, word_labels in zip(
        test_sentences,
        test_labels,
    ):
        inputs = tokenizer(
            tokens,
            return_tensors="pt",
            truncation=True,
            is_split_into_words=True,
            max_length=128,
        )

        word_ids = inputs.word_ids(
            batch_index=0
        )

        aligned_labels = align_labels(
            word_ids,
            word_labels,
        )

        with torch.no_grad():
            fp32_logits = fp32_model(
                **inputs
            ).logits

        fp32_predictions = (
            torch.argmax(
                fp32_logits,
                dim=-1,
            )
            .squeeze(0)
            .cpu()
            .numpy()
        )

        int8_logits = int8_model(
            **inputs
        ).logits

        int8_predictions = np.argmax(
            int8_logits,
            axis=-1,
        )[0]

        true_sentence = []
        fp32_sentence = []
        int8_sentence = []

        for true_id, fp32_id, int8_id in zip(
            aligned_labels,
            fp32_predictions,
            int8_predictions,
        ):
            if true_id == -100:
                continue

            true_label = id2label[
                int(true_id)
            ]

            fp32_label = id2label[
                int(fp32_id)
            ]

            int8_label = id2label[
                int(int8_id)
            ]

            true_sentence.append(
                true_label
            )

            fp32_sentence.append(
                fp32_label
            )

            int8_sentence.append(
                int8_label
            )

            fp32_token_correct.append(
                float(
                    fp32_label == true_label
                )
            )

            int8_token_correct.append(
                float(
                    int8_label == true_label
                )
            )

        true_sequences.append(
            true_sentence
        )

        fp32_sequences.append(
            fp32_sentence
        )

        int8_sequences.append(
            int8_sentence
        )

    fp32_f1 = ner_f1(
        true_sequences,
        fp32_sequences,
    )

    int8_f1 = ner_f1(
        true_sequences,
        int8_sequences,
    )

    fp32_accuracy = ner_accuracy(
        true_sequences,
        fp32_sequences,
    )

    int8_accuracy = ner_accuracy(
        true_sequences,
        int8_sequences,
    )

    f1_delta = int8_f1 - fp32_f1

    accuracy_tax = (
        fp32_accuracy - int8_accuracy
    )

    _, ci_lo, ci_hi = paired_bootstrap_diff(
        fp32_token_correct,
        int8_token_correct,
    )

    changed = sum(
        fp32 != int8
        for fp32, int8 in zip(
            [
                label
                for sentence in fp32_sequences
                for label in sentence
            ],
            [
                label
                for sentence in int8_sequences
                for label in sentence
            ],
        )
    )

    total_tokens = sum(
        len(sentence)
        for sentence in true_sequences
    )

    print(
        "\n--- NER quality comparison ---"
    )

    print(
        f"FP32 accuracy: {fp32_accuracy:.4f}"
    )

    print(
        f"INT8 accuracy: {int8_accuracy:.4f}"
    )

    print(
        f"FP32 F1: {fp32_f1:.4f}"
    )

    print(
        f"INT8 F1: {int8_f1:.4f}"
    )

    print(
        f"F1 delta: {f1_delta:+.4f}"
    )

    print(
        f"Accuracy tax: {accuracy_tax:+.4f}"
    )

    print(
        "Accuracy tax 95% CI: "
        f"[{ci_lo:+.4f}, {ci_hi:+.4f}]"
    )

    print(
        f"Changed token predictions: "
        f"{changed}/{total_tokens}"
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    classifier_tokenizer = export_classifier()

    check_classifier_quality(
        classifier_tokenizer
    )

    ner_tokenizer = export_ner()

    check_ner_quality(
        ner_tokenizer
    )

    print(
        "\nClassifier and NER export "
        "and quality checks complete."
    )


if __name__ == "__main__":
    main()