"""Lab 3A starter: fine-tune the Bayan topic classifier."""

import argparse
from pathlib import Path

import numpy as np

from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from bayan.models.data import build_topic_dataset


CHECKPOINT = "xlm-roberta-base"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        default="artifacts/topic_classifier",
        help="Where to save the trained classifier artefact.",
    )

    return parser.parse_args()


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(
            labels,
            predictions,
            average="macro",
        ),
    }


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 1. Load leakage-safe grouped dataset
    # --------------------------------------------------
    dataset = build_topic_dataset()

    # --------------------------------------------------
    # 2. Create label mappings
    # --------------------------------------------------
    topics = sorted(set(dataset["train"]["topic"]))

    label2id = {
        label: index
        for index, label in enumerate(topics)
    }

    id2label = {
        index: label
        for label, index in label2id.items()
    }

    print("Topic labels:")
    print(label2id)

    # Convert topic strings to integer labels
    def add_labels(example):
        example["labels"] = label2id[example["topic"]]
        return example

    dataset = dataset.map(add_labels)

    # --------------------------------------------------
    # 3. Load Lab 1 tokenizer
    # --------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT
    )

    # --------------------------------------------------
    # 4. Tokenize text
    # --------------------------------------------------
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=256,
        )

    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
    )

    # Dynamic padding
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    # --------------------------------------------------
    # 5. Load classification model
    # --------------------------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT,
        num_labels=len(topics),
        label2id=label2id,
        id2label=id2label,
    )

    # --------------------------------------------------
    # 6. Training configuration
    # --------------------------------------------------
    training_args = TrainingArguments(
        output_dir=str(output_dir),

        learning_rate=2e-5,

        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,

        num_train_epochs=3,

        weight_decay=0.01,

        eval_strategy="epoch",
        save_strategy="epoch",

        load_best_model_at_end=True,

        metric_for_best_model="macro_f1",
        greater_is_better=True,

        logging_steps=10,

        seed=42,
    )

    # --------------------------------------------------
    # 7. Create Trainer
    # --------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,

        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],

        processing_class=tokenizer,
        data_collator=data_collator,

        compute_metrics=compute_metrics,
    )

    # --------------------------------------------------
    # 8. Fine-tune
    # --------------------------------------------------
    print("\nStarting training...\n")

    trainer.train()

    # --------------------------------------------------
    # 9. Evaluate on frozen test split
    # --------------------------------------------------
    print("\nEvaluating on frozen test split...\n")

    test_metrics = trainer.evaluate(
        tokenized_dataset["test"],
        metric_key_prefix="test",
    )

    print("\nFrozen Test Metrics")

    for name, value in test_metrics.items():
        print(f"{name}: {value}")

    # --------------------------------------------------
    # 10. Save re-runnable artefact
    # --------------------------------------------------
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(
        f"\nSaved trained classifier to: {output_dir}"
    )


if __name__ == "__main__":
    main()