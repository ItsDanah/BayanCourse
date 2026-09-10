"""Lab 3B starter: run the 12-question QA smoke set."""

import json

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from bayan.models.qa import best_span


DATA_PATH = "data/models/bayan_qa.json"
CHECKPOINT = "deepset/roberta-base-squad2"
NULL_THRESHOLD = 0.0


def load_smoke_set(path):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    answerable = []
    null_examples = []

    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]

            for qa in paragraph["qas"]:
                example = {
                    "question": qa["question"],
                    "context": context,
                    "answers": qa["answers"],
                    "is_impossible": qa["is_impossible"],
                }

                if qa["is_impossible"]:
                    null_examples.append(example)
                else:
                    answerable.append(example)

    return answerable[:9] + null_examples[:3]


def main():
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    model = AutoModelForQuestionAnswering.from_pretrained(CHECKPOINT)

    model.eval()

    examples = load_smoke_set(DATA_PATH)

    correct_answerable = 0
    correct_null = 0

    total_answerable = 0
    total_null = 0

    for number, example in enumerate(examples, start=1):
        question = example["question"]
        context = example["context"]

        encoded = tokenizer(
            question,
            context,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
        )

        offsets = encoded.pop("offset_mapping")[0].tolist()
        sequence_ids = encoded.sequence_ids(0)

        context_offsets = [
            tuple(offset) if sequence_id == 1 else None
            for offset, sequence_id in zip(offsets, sequence_ids)
        ]

        with torch.no_grad():
            outputs = model(**encoded)

        start_logits = outputs.start_logits[0].cpu().numpy()
        end_logits = outputs.end_logits[0].cpu().numpy()

        null_score = float(
            start_logits[0] + end_logits[0]
        )

        result = best_span(
            start_logits,
            end_logits,
            context_offsets,
            null_score=null_score,
            null_threshold=NULL_THRESHOLD,
        )

        if result["answer"] is None:
            predicted_answer = None
        else:
            start_char, end_char = result["answer"]
            predicted_answer = context[start_char:end_char]

        print(f"\nQuestion {number}: {question}")
        print("Prediction:", predicted_answer)

        if example["is_impossible"]:
            total_null += 1

            if predicted_answer is None:
                correct_null += 1
                print("PASS")
            else:
                print("FAIL")

        else:
            total_answerable += 1

            expected_answers = [
                answer["text"]
                for answer in example["answers"]
            ]

            print("Expected:", expected_answers)

            if predicted_answer in expected_answers:
                correct_answerable += 1
                print("PASS")
            else:
                print("FAIL")

    print("\nQA smoke results:")
    print(f"Answerable: {correct_answerable}/{total_answerable}")
    print(f"Null: {correct_null}/{total_null}")


if __name__ == "__main__":
    main()