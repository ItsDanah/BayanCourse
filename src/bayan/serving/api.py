"""Lab 7/capstone: Bayan FastAPI serving application."""

import numpy as np

from fastapi import FastAPI, HTTPException
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

from bayan.models.data import build_topic_dataset
from bayan.preprocessing.core import preprocess
from bayan.serving.canaries import run_startup_canaries


CLASSIFIER_PATH = "artifacts/onnx_classifier_int8"
MODEL_FILE = "model_quantized.onnx"
MAX_LENGTH = 128


# --------------------------------------------------
# Load classifier artefact
# --------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    CLASSIFIER_PATH
)

classifier = ORTModelForSequenceClassification.from_pretrained(
    CLASSIFIER_PATH,
    file_name=MODEL_FILE,
)


# Keep the same label ordering used during training.
_dataset = build_topic_dataset()

TOPICS = sorted(
    set(_dataset["train"]["topic"])
)

ID2LABEL = {
    index: label
    for index, label in enumerate(TOPICS)
}


# --------------------------------------------------
# Startup canaries
# --------------------------------------------------

run_startup_canaries()


# --------------------------------------------------
# API
# --------------------------------------------------

app = FastAPI(
    title="Bayan — Bilingual Citizen-Feedback Intelligence Service"
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "classifier": "onnx-int8",
        "canaries": "green",
    }


@app.post("/v1/classify")
def classify(payload: dict):
    text = payload.get("text")

    if not isinstance(text, str) or not text.strip():
        raise HTTPException(
            status_code=400,
            detail="A non-empty 'text' field is required.",
        )

    # Shared preprocessing contract
    processed_text = preprocess(text)

    inputs = tokenizer(
        processed_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    outputs = classifier(**inputs)

    logits = outputs.logits

    if hasattr(logits, "detach"):
        logits = logits.detach().cpu().numpy()
    else:
        logits = np.asarray(logits)

    prediction_id = int(
        np.argmax(logits, axis=-1)[0]
    )

    prediction = ID2LABEL.get(
        prediction_id,
        str(prediction_id),
    )

    return {
        "topic": prediction,
    }


@app.post("/v1/entities")
def entities(payload: dict):
    # TODO(Capstone): shared preprocessing/Arabic segmentation -> NER -> case fields.
    raise NotImplementedError("Wire the NER artefact")


@app.post("/v1/search")
def search(payload: dict):
    # TODO(Capstone): Lab 5 two-stage bilingual search.
    raise NotImplementedError("Wire the semantic-search component")


@app.post("/v1/analyse")
def analyse(payload: dict):
    # TODO(Capstone): one bilingual request -> classification + entities + similar cases.
    raise NotImplementedError("Assemble the Bayan capstone service")