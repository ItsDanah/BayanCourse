"""Lab 7/capstone: startup skew and behaviour canaries."""

from pathlib import Path

from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

from bayan.preprocessing.core import preprocess


CLASSIFIER_PATH = Path("artifacts/onnx_classifier_int8")
MODEL_FILE = "model_quantized.onnx"


def run_startup_canaries() -> None:
    """Verify preprocessing and the selected classifier artefact."""

    model_path = CLASSIFIER_PATH / MODEL_FILE

    if not CLASSIFIER_PATH.exists():
        raise RuntimeError(
            f"Classifier artefact directory not found: {CLASSIFIER_PATH}"
        )

    if not model_path.exists():
        raise RuntimeError(
            f"Classifier model not found: {model_path}"
        )

    # Shared preprocessing contract
    raw_text = "  الخدمة   ممتازة  "
    processed_text = preprocess(raw_text)

    if not isinstance(processed_text, str):
        raise RuntimeError(
            "Preprocessing canary failed: preprocess() must return a string."
        )

    if not processed_text.strip():
        raise RuntimeError(
            "Preprocessing canary failed: output is empty."
        )

    # Verify tokenizer and INT8 model can load together.
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            CLASSIFIER_PATH
        )

        model = ORTModelForSequenceClassification.from_pretrained(
            CLASSIFIER_PATH,
            file_name=MODEL_FILE,
        )

        inputs = tokenizer(
            processed_text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )

        outputs = model(**inputs)

        if outputs.logits.shape[-1] <= 0:
            raise RuntimeError(
                "Classifier canary returned invalid logits."
            )

    except Exception as exc:
        raise RuntimeError(
            f"Classifier startup canary failed: {exc}"
        ) from exc