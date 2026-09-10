"""Lab 5: labelled-query retrieval evaluation."""

import json

import faiss
from bayan.search.service import CaseSearch


QUERIES_PATH = "data/search/bayan_queries.jsonl"
INDEX_PREFIX = "artifacts/case_index_v1"


def load_queries():
    queries = []

    with open(QUERIES_PATH, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                queries.append(json.loads(line))

    return queries


def relevant_by_topic(results, topic):
    return [
        result
        for result in results
        if result.get("topic") == topic
    ]


def recall_at_10(results, topic):
    top_10 = results[:10]
    return 1.0 if relevant_by_topic(top_10, topic) else 0.0


def reciprocal_rank(results, topic):
    for rank, result in enumerate(results[:10], start=1):
        if result.get("topic") == topic:
            return 1.0 / rank

    return 0.0


def bi_encoder_search(searcher, query, k=10):
    vector = searcher.encoder.encode(
        [query],
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")

    faiss.normalize_L2(vector)

    _, indices = searcher.index.search(vector, k)

    return [
        searcher.metadata[int(i)]
        for i in indices[0]
        if i >= 0
    ]


def evaluate(searcher, queries, rerank=False):
    recalls = []
    mrrs = []
    language_recalls = {}

    for item in queries:
        if item["no_answer"]:
            continue

        if rerank:
            results = searcher.search(
                item["query"],
                k=10,
                candidates=50,
                min_score=float("-inf"),
            )
        else:
            results = bi_encoder_search(
                searcher,
                item["query"],
                k=10,
            )

        recall = recall_at_10(
            results,
            item["topic"],
        )

        rr = reciprocal_rank(
            results,
            item["topic"],
        )

        recalls.append(recall)
        mrrs.append(rr)

        lang = item["lang"]

        language_recalls.setdefault(
            lang,
            [],
        ).append(recall)

    mean_recall = sum(recalls) / len(recalls)
    mean_mrr = sum(mrrs) / len(mrrs)

    language_scores = {
        lang: sum(values) / len(values)
        for lang, values in language_recalls.items()
    }

    return mean_recall, mean_mrr, language_scores


def evaluate_no_answer(searcher, queries):
    no_answer_queries = [
        item
        for item in queries
        if item["no_answer"]
    ]

    correct = 0

    for item in no_answer_queries:
        results = searcher.search(
            item["query"],
            k=10,
            candidates=50,
            min_score=0.25,
        )

        if not results:
            correct += 1

    return correct, len(no_answer_queries)


def main():
    print("Loading search models and index...")

    searcher = CaseSearch(INDEX_PREFIX)

    queries = load_queries()

    print(
        f"Loaded {len(queries)} labelled queries."
    )

    print("\n--- Without reranking ---")

    recall_no, mrr_no, lang_no = evaluate(
        searcher,
        queries,
        rerank=False,
    )

    print(f"Recall@10: {recall_no:.4f}")
    print(f"MRR@10:    {mrr_no:.4f}")

    print("\n--- With reranking ---")

    recall_rerank, mrr_rerank, lang_rerank = evaluate(
        searcher,
        queries,
        rerank=True,
    )

    print(
        f"Recall@10: {recall_rerank:.4f}"
    )

    print(
        f"MRR@10:    {mrr_rerank:.4f}"
    )

    print("\n--- Language slices ---")

    for lang, score in lang_rerank.items():
        print(
            f"{lang}: Recall@10 = {score:.4f}"
        )

    if len(lang_rerank) >= 2:
        scores = list(lang_rerank.values())

        gap = max(scores) - min(scores)

        print(
            f"Cross-lingual slice gap: {gap:.4f}"
        )

    print("\n--- No-answer behaviour ---")

    correct, total = evaluate_no_answer(
        searcher,
        queries,
    )

    print(
        f"No-answer correctness: {correct}/{total}"
    )


if __name__ == "__main__":
    main()