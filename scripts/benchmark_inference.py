"""Lab 7: honest p50/p99 benchmark harness over production length mix."""

import os
import time

import numpy as np
import torch

from optimum.onnxruntime import (
    ORTModelForSequenceClassification,
    ORTModelForTokenClassification,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

CLASSIFIER_PATH = "artifacts/topic_classifier"
CLASSIFIER_ONNX_FP32 = "artifacts/onnx_classifier_fp32"
CLASSIFIER_ONNX_INT8 = "artifacts/onnx_classifier_int8"

NER_PATH = "artifacts/ner"
NER_ONNX_FP32 = "artifacts/onnx_ner_fp32"
NER_ONNX_INT8 = "artifacts/onnx_ner_int8"

MIX_PATH = "data/serving/bench_mix.npy"

THREADS = 4
SAMPLE_SIZE = 200


# --------------------------------------------------
# Benchmark helper
# --------------------------------------------------

def benchmark(
    model,
    tokenizer,
    texts,
    *,
    max_length,
    dynamic_padding,
    use_torch=False,
):
    texts = [str(text) for text in texts]

    # Warm-up
    for text in texts[:10]:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False if dynamic_padding else "max_length",
        )

        if use_torch:
            with torch.no_grad():
                model(**inputs)
        else:
            model(**inputs)

    latencies = []

    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False if dynamic_padding else "max_length",
        )

        start = time.perf_counter()

        if use_torch:
            with torch.no_grad():
                model(**inputs)
        else:
            model(**inputs)

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        latencies.append(elapsed_ms)

    return {
        "p50": float(
            np.percentile(latencies, 50)
        ),
        "p99": float(
            np.percentile(latencies, 99)
        ),
    }


def print_result(
    name,
    result,
    baseline_p99=None,
):
    print(f"\n--- {name} ---")

    print(
        f"p50: {result['p50']:.2f} ms"
    )

    print(
        f"p99: {result['p99']:.2f} ms"
    )

    if baseline_p99 is not None:
        speedup = (
            baseline_p99 / result["p99"]
        )

        print(
            f"speed-up vs baseline: "
            f"{speedup:.2f}x"
        )


# --------------------------------------------------
# Classifier benchmark
# --------------------------------------------------

def benchmark_classifier(texts):
    print("\n==============================")
    print("CLASSIFIER BENCHMARK")
    print("==============================")

    tokenizer = AutoTokenizer.from_pretrained(
        CLASSIFIER_PATH
    )

    # Torch FP32 @512
    torch_model = (
        AutoModelForSequenceClassification
        .from_pretrained(CLASSIFIER_PATH)
    )

    torch_model.eval()

    baseline = benchmark(
        torch_model,
        tokenizer,
        texts,
        max_length=512,
        dynamic_padding=False,
        use_torch=True,
    )

    print_result(
        "Classifier FP32 torch @512 padded",
        baseline,
    )

    # Torch FP32 @128
    dynamic = benchmark(
        torch_model,
        tokenizer,
        texts,
        max_length=128,
        dynamic_padding=True,
        use_torch=True,
    )

    print_result(
        "Classifier FP32 torch @128 dynamic",
        dynamic,
        baseline["p99"],
    )

    # ONNX FP32
    onnx_fp32_model = (
        ORTModelForSequenceClassification
        .from_pretrained(
            CLASSIFIER_ONNX_FP32,
            file_name="model.onnx",
        )
    )

    onnx_fp32 = benchmark(
        onnx_fp32_model,
        tokenizer,
        texts,
        max_length=128,
        dynamic_padding=True,
    )

    print_result(
        "Classifier ONNX FP32 @128",
        onnx_fp32,
        baseline["p99"],
    )

    # ONNX INT8
    onnx_int8_model = (
        ORTModelForSequenceClassification
        .from_pretrained(
            CLASSIFIER_ONNX_INT8,
            file_name="model_quantized.onnx",
        )
    )

    onnx_int8 = benchmark(
        onnx_int8_model,
        tokenizer,
        texts,
        max_length=128,
        dynamic_padding=True,
    )

    print_result(
        "Classifier ONNX INT8 @128",
        onnx_int8,
        baseline["p99"],
    )


# --------------------------------------------------
# NER benchmark
# --------------------------------------------------

def benchmark_ner(texts):
    print("\n==============================")
    print("NER BENCHMARK")
    print("==============================")

    tokenizer = AutoTokenizer.from_pretrained(
        NER_PATH
    )

    # Torch FP32
    torch_model = (
        AutoModelForTokenClassification
        .from_pretrained(NER_PATH)
    )

    torch_model.eval()

    fp32 = benchmark(
        torch_model,
        tokenizer,
        texts,
        max_length=128,
        dynamic_padding=True,
        use_torch=True,
    )

    print_result(
        "NER FP32 torch @128",
        fp32,
    )

    # ONNX FP32
    onnx_fp32_model = (
        ORTModelForTokenClassification
        .from_pretrained(
            NER_ONNX_FP32,
            file_name="model.onnx",
        )
    )

    onnx_fp32 = benchmark(
        onnx_fp32_model,
        tokenizer,
        texts,
        max_length=128,
        dynamic_padding=True,
    )

    print_result(
        "NER ONNX FP32 @128",
        onnx_fp32,
        fp32["p99"],
    )

    # ONNX INT8
    onnx_int8_model = (
        ORTModelForTokenClassification
        .from_pretrained(
            NER_ONNX_INT8,
            file_name="model_quantized.onnx",
        )
    )

    onnx_int8 = benchmark(
        onnx_int8_model,
        tokenizer,
        texts,
        max_length=128,
        dynamic_padding=True,
    )

    print_result(
        "NER ONNX INT8 @128",
        onnx_int8,
        fp32["p99"],
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    os.environ["OMP_NUM_THREADS"] = str(
        THREADS
    )

    torch.set_num_threads(
        THREADS
    )

    print(
        f"CPU threads pinned to: {THREADS}"
    )

    texts = np.load(
        MIX_PATH,
        allow_pickle=True,
    )[:SAMPLE_SIZE]

    print(
        f"Loaded {len(texts)} "
        "production examples."
    )

    benchmark_classifier(texts)

    benchmark_ner(texts)


if __name__ == "__main__":
    main()