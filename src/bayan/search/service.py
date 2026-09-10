"""Lab 5: two-stage bilingual case search."""

import json
from pathlib import Path

import faiss
from sentence_transformers import CrossEncoder, SentenceTransformer


RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class CaseSearch:
    def __init__(self, prefix: str):
        """Load and validate the persisted search index."""

        manifest_path = Path(f"{prefix}_manifest.json")
        metadata_path = Path(f"{prefix}_metadata.json")
        index_path = Path(f"{prefix}.faiss")

        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")

        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata: {metadata_path}")

        if not index_path.exists():
            raise FileNotFoundError(f"Missing FAISS index: {index_path}")

        # Load persisted files
        self.manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )

        self.metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )

        self.index = faiss.read_index(str(index_path))

        # Validate manifest
        required = {"model", "preproc_version", "n_vectors", "dim"}

        missing = required - self.manifest.keys()

        if missing:
            raise ValueError(
                f"Manifest missing required fields: {sorted(missing)}"
            )

        if self.index.ntotal != self.manifest["n_vectors"]:
            raise ValueError(
                "Manifest vector count does not match FAISS index."
            )

        if self.index.d != self.manifest["dim"]:
            raise ValueError(
                "Manifest dimension does not match FAISS index."
            )

        if len(self.metadata) != self.manifest["n_vectors"]:
            raise ValueError(
                "Metadata count does not match FAISS index."
            )

        # Load models pinned by the index manifest
        self.encoder = SentenceTransformer(
            self.manifest["model"]
        )

        self.reranker = CrossEncoder(RERANKER_MODEL)

    def search(
        self,
        query: str,
        k: int = 5,
        candidates: int = 50,
        min_score: float = 0.25,
    ):
        """Retrieve with a bi-encoder, then rerank with a cross-encoder."""

        query = " ".join(query.split())

        if not query:
            return []

        # Encode and L2-normalise query
        query_vector = self.encoder.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")

        faiss.normalize_L2(query_vector)

        # Retrieve candidates
        candidate_count = min(
            candidates,
            self.index.ntotal,
        )

        _, indices = self.index.search(
            query_vector,
            candidate_count,
        )

        results = []

        for index_id in indices[0]:
            if index_id < 0:
                continue

            item = dict(self.metadata[index_id])
            item["_index_id"] = int(index_id)
            results.append(item)

        if not results:
            return []

        # Determine case text field
        if "text" in results[0]:
            text_field = "text"
        elif "case_text" in results[0]:
            text_field = "case_text"
        else:
            raise ValueError(
                "Metadata does not contain a case text field."
            )

        # Cross-encoder reranking
        pairs = [
            [query, result[text_field]]
            for result in results
        ]

        scores = self.reranker.predict(pairs)

        for result, score in zip(results, scores):
            result["score"] = float(score)

        results.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        # Honest no-result behaviour
        results = [
            result
            for result in results
            if result["score"] >= min_score
        ]

        return results[:k]