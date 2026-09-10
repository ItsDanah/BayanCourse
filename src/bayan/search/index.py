"""Lab 5: versioned FAISS index build."""

import json
from pathlib import Path

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer


DATA_PATH = "data/search/bayan_cases.csv"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PREPROC_VERSION = "v1"


def build_index(prefix: str, limit: int | None = None):
    """Build and persist a versioned FAISS index over Bayan cases."""

    # Load case corpus
    df = pd.read_csv(DATA_PATH)

    if limit is not None:
        df = df.head(limit)

    # Find the text column used by the case corpus
    if "text" in df.columns:
        text_column = "text"
    elif "case_text" in df.columns:
        text_column = "case_text"
    else:
        raise ValueError(
            f"Could not find case text column. Available columns: {list(df.columns)}"
        )

    texts = df[text_column].fillna("").astype(str).tolist()

    if not texts:
        raise ValueError("No cases available to index.")

    # Encode corpus
    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")

    # L2-normalise so inner product behaves as cosine similarity
    faiss.normalize_L2(embeddings)

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Persist index
    index_path = f"{prefix}.faiss"
    faiss.write_index(index, index_path)

    # Persist metadata
    metadata_path = f"{prefix}_metadata.json"
    Path(metadata_path).write_text(
        df.to_json(orient="records", force_ascii=False),
        encoding="utf-8",
    )

    # Persist manifest with pinned model/preprocessing information
    manifest = {
        "model": MODEL_NAME,
        "preproc_version": PREPROC_VERSION,
        "n_vectors": int(index.ntotal),
        "dim": int(dim),
    }

    manifest_path = f"{prefix}_manifest.json"
    Path(manifest_path).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return index