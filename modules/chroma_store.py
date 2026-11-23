
from pathlib import Path
import hashlib
import json

import chromadb


# Project database location
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_PATH = PROJECT_ROOT / "database" / "chroma_db"

COLLECTION_NAME = "documents"


class ChromaStore:
    """Store and search document chunks using ChromaDB."""

    def __init__(self):
        DATABASE_PATH.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(DATABASE_PATH)
        )

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def generate_chunk_id(self, chunk):
        """Generate a stable ID for each document chunk."""

        identity = {
            "text": chunk["text"],
            "metadata": chunk["metadata"]
        }

        content = json.dumps(
            identity,
            sort_keys=True,
            ensure_ascii=False,
            default=str
        )

        return hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()

    def store_chunks(self, embedded_chunks):
        """Save document chunks, metadata and embeddings."""

        if not embedded_chunks:
            return 0

        ids = []
        documents = []
        metadatas = []
        embeddings = []

        for chunk in embedded_chunks:

            text = chunk["text"].strip()
            metadata = chunk["metadata"].copy()
            embedding = chunk["embedding"]

            if not text:
                continue

            if not embedding:
                raise ValueError("Embedding cannot be empty")

            if len(embedding) != 768:
                raise ValueError(
                    "Expected 768-dimensional embedding"
                )

            # ChromaDB metadata must use supported scalar values.
            clean_metadata = {
                key: value
                for key, value in metadata.items()
                if isinstance(value, (str, int, float, bool))
                and value is not None
            }

            ids.append(self.generate_chunk_id(chunk))
            documents.append(text)
            metadatas.append(clean_metadata)
            embeddings.append(embedding)

        if not ids:
            return 0

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )

        return len(ids)

    def search(self, query_embedding, top_k=5):
        """Find the most relevant document chunks."""

        if not query_embedding:
            raise ValueError("Query embedding cannot be empty")

        if len(query_embedding) != 768:
            raise ValueError(
                "Expected 768-dimensional query embedding"
            )

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        total_records = self.collection.count()

        if total_records == 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, total_records),
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        matches = []

        for index, text in enumerate(results["documents"][0]):
            matches.append({
                "text": text,
                "metadata": results["metadatas"][0][index],
                "distance": results["distances"][0][index]
            })

        return matches

    def count(self):
        """Return the number of stored chunks."""

        return self.collection.count()
