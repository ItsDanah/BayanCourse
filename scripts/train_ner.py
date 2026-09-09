"""Lab 3B: fine-tune token classification with correct BIO alignment."""

import argparse
from pathlib import Path

import numpy as np
from datasets import Dataset
from seqeval.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from bayan.models.ner import align_labels


CHECKPOINT = "xlm-roberta-base"
DATA_PATH = "data/models/bayan_ner.conll"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        default="artifacts/ner",
        help="Where to save the trained NER artefact.",
    )

    return parser.parse_args()


def read_conll(path):
    sentences = []
    labels = []

    current_tokens = []
    current_labels = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                if current_tokens:
                    sentences.append(current_tokens)
                    labels.append(current_labels)

                    current_tokens = []
                    current_labels = []

                continue

            token, label = line.rsplit(maxsplit=1)

            current_tokens.append(token)
            current_labels.append(label)

    if current_tokens:
        sentences.append(current_tokens)
        labels.append(current_labels)

    return sentences, labels


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 1. Read CoNLL data
    # --------------------------------------------------

    sentences, bio_labels = read_conll(DATA_PATH)

    print("Number of sentences:")
    print(len(sentences))

    # --------------------------------------------------
    # 2. Create label mappings
    # --------------------------------------------------

    unique_labels = sorted(
        {
            label
            for sentence_labels in bio_labels
            for label in sentence_labels
        }
    )

    # Keep O as label 0
    if "O" in unique_labels:
        unique_labels.remove("O")
        unique_labels = ["O"] + unique_labels

    label2id = {
        label: i
        for i, label in enumerate(unique_labels)
    }

    id2label = {
        i: label
        for label, i in label2id.items()
    }

    print("\nLabels:")
    print(unique_labels)

    # --------------------------------------------------
    # 3. Convert BIO labels to numeric IDs
    # --------------------------------------------------

    numeric_labels = [
        [label2id[label] for label in sentence_labels]
        for sentence_labels in bio_labels
    ]

    # --------------------------------------------------
    # 4. Deterministic train/validation/test split
    # --------------------------------------------------

    total = len(sentences)

    train_end = int(total * 0.70)
    validation_end = int(total * 0.90)

    train_data = Dataset.from_dict(
        {
            "tokens": sentences[:train_end],
            "ner_tags": numeric_labels[:train_end],
        }
    )

    validation_data = Dataset.from_dict(
        {
            "tokens": sentences[train_end:validation_end],
            "ner_tags": numeric_labels[train_end:validation_end],
        }
    )

    test_data = Dataset.from_dict(
        {
            "tokens": sentences[validation_end:],
            "ner_tags": numeric_labels[validation_end:],
        }
    )

    print("\nDataset sizes:")
    print("Train:", len(train_data))
    print("Validation:", len(validation_data))
    print("Test:", len(test_data))

    # --------------------------------------------------
    # 5. Load tokenizer
    # --------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT
    )

    # --------------------------------------------------
    # 6. Tokenize and align BIO labels
    # --------------------------------------------------

    def tokenize_and_align_labels(examples):
        tokenized = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=128,
        )

        aligned_batch = []

        for batch_index, labels in enumerate(
            examples["ner_tags"]
        ):
            word_ids = tokenized.word_ids(
                batch_index=batch_index
            )

            aligned = align_labels(
                word_ids,
                labels,
            )

            aligned_batch.append(aligned)

        tokenized["labels"] = aligned_batch

        return tokenized

    train_data = train_data.map(
        tokenize_and_align_labels,
        batched=True,
    )

    validation_data = validation_data.map(
        tokenize_and_align_labels,
        batched=True,
    )

    test_data = test_data.map(
        tokenize_and_align_labels,
        batched=True,
    )

    # --------------------------------------------------
    # 7. Load token-classification model
    # --------------------------------------------------

    model = AutoModelForTokenClassification.from_pretrained(
        CHECKPOINT,
        num_labels=len(unique_labels),
        id2label=id2label,
        label2id=label2id,
    )

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer
    )

    # --------------------------------------------------
    # 8. Entity-level seqeval metrics
    # --------------------------------------------------

    def compute_metrics(eval_pred):
        logits, true_labels = eval_pred

        predictions = np.argmax(
            logits,
            axis=2,
        )

        true_predictions = []
        true_label_names = []

        for prediction, labels in zip(
            predictions,
            true_labels,
        ):
            sentence_predictions = []
            sentence_labels = []

            for predicted_id, label_id in zip(
                prediction,
                labels,
            ):
                if label_id == -100:
                    continue

                sentence_predictions.append(
                    id2label[int(predicted_id)]
                )

                sentence_labels.append(
                    id2label[int(label_id)]
                )

            true_predictions.append(
                sentence_predictions
            )

            true_label_names.append(
                sentence_labels
            )

        return {
            "precision": precision_score(
                true_label_names,
                true_predictions,
            ),
            "recall": recall_score(
                true_label_names,
                true_predictions,
            ),
            "f1": f1_score(
                true_label_names,
                true_predictions,
            ),
            "accuracy": accuracy_score(
                true_label_names,
                true_predictions,
            ),
        }

    # --------------------------------------------------
    # 9. Training configuration
    # --------------------------------------------------

    training_args = TrainingArguments(
        output_dir=str(
            output_dir / "checkpoints"
        ),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        report_to="none",
        seed=42,
    )

    # --------------------------------------------------
    # 10. Trainer
    # --------------------------------------------------

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=validation_data,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # --------------------------------------------------
    # 11. Train
    # --------------------------------------------------

    print("\nStarting NER training...\n")

    trainer.train()

    # --------------------------------------------------
    # 12. Validation evaluation
    # --------------------------------------------------

    print("\nValidation results:\n")

    validation_results = trainer.evaluate(
        validation_data,
        metric_key_prefix="validation",
    )

    for name, value in validation_results.items():
        print(f"{name}: {value}")

    # --------------------------------------------------
    # 13. Frozen test evaluation
    # --------------------------------------------------

    print("\nFrozen test results:\n")

    test_results = trainer.evaluate(
        test_data,
        metric_key_prefix="test",
    )

    for name, value in test_results.items():
        print(f"{name}: {value}")

    # --------------------------------------------------
    # 14. Save NER artefact
    # --------------------------------------------------

    trainer.save_model(
        str(output_dir)
    )

    tokenizer.save_pretrained(
        str(output_dir)
    )

    print("\nSaved NER artefact to:")
    print(output_dir)


if __name__ == "__main__":
    main()