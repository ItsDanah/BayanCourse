"""Lab 3 starter: extractive QA post-processing."""

import numpy as np


def best_span(
    start_logits,
    end_logits,
    offsets,
    *,
    null_score,
    null_threshold,
    max_answer_len=30,
    top_k=20,
):
    start_logits = np.asarray(start_logits)
    end_logits = np.asarray(end_logits)

    start_indices = np.argsort(start_logits)[-top_k:][::-1]
    end_indices = np.argsort(end_logits)[-top_k:][::-1]

    best_score = float("-inf")
    best_answer = None

    for start_idx in start_indices:
        for end_idx in end_indices:

            # Skip special/question tokens.
            if offsets[start_idx] is None or offsets[end_idx] is None:
                continue

            # Reject inverted spans.
            if end_idx < start_idx:
                continue

            # Reject answers that are too long.
            if end_idx - start_idx + 1 > max_answer_len:
                continue

            score = float(
                start_logits[start_idx] + end_logits[end_idx]
            )

            if score > best_score:
                best_score = score
                best_answer = (
                    offsets[start_idx][0],
                    offsets[end_idx][1],
                )

    # No valid span found.
    if best_answer is None:
        return {
            "answer": None,
            "score": float(null_score),
        }

    # Model prefers no-answer.
    if null_score - best_score > null_threshold:
        return {
            "answer": None,
            "score": float(null_score),
        }

    return {
        "answer": best_answer,
        "score": best_score,
    }