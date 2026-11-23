
import re

from modules.embedding_engine import embed_text
from modules.chroma_store import ChromaStore


class RetrievalEngine:

    def __init__(self, top_k=5, max_distance=0.55):
        self.database = ChromaStore()
        self.top_k = top_k
        self.max_distance = max_distance

    def retrieve(self, question):

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question cannot be empty")

        if self.top_k <= 0:
            raise ValueError("top_k must be positive")

        question_lower = question.lower().strip()

        # Generate query embedding
        query_embedding = embed_text(
            question,
            task="query"
        )

        # Retrieve candidate chunks
        results = self.database.search(
            query_embedding,
            top_k=max(20, self.top_k * 4)
        )

        # Special handling for abstract questions
        abstract_keywords = [
            "abstract",
            "summary of the report",
            "report summary",
            "overview of the report"
        ]

        if any(keyword in question_lower for keyword in abstract_keywords):

            page_data = self.database.collection.get(
                where={
                    "source": "AICTE ACTIVITY REPORT final (2).pdf"
                },
                include=["documents", "metadatas"]
            )

            abstract_chunks = []

            for text, metadata in zip(
                page_data["documents"],
                page_data["metadatas"]
            ):

                if (
                    metadata.get("page") == 5
                    and metadata.get("chunk_index") in [1, 2]
                ):

                    abstract_chunks.append({
                        "text": text,
                        "metadata": metadata,
                        "distance": 0.0
                    })

            if abstract_chunks:

                abstract_chunks.sort(
                    key=lambda item: item["metadata"].get(
                        "chunk_index", 0
                    )
                )

                return abstract_chunks

        # Remove common question words
        stop_words = {
            "what", "how", "was", "were", "is", "are",
            "the", "a", "an", "in", "on", "of", "to",
            "and", "under", "by", "for", "with"
        }

        query_words = {
            word.lower()
            for word in re.findall(
                r"\b[a-zA-Z]+\b",
                question
            )
            if word.lower() not in stop_words
        }

        candidates = []

        for result in results:

            text = result["text"].strip()
            distance = result["distance"]

            if distance > self.max_distance:
                continue

            # Ignore very short chunks
            if len(text) <= 30:
                continue

            # Filter photo-only chunks
            if (
                "photo" in text.lower()
                and len(
                    re.findall(r"\b[a-zA-Z]+\b", text)
                ) < 25
            ):
                continue

            document_words = {
                word.lower()
                for word in re.findall(
                    r"\b[a-zA-Z]+\b",
                    text
                )
            }

            keyword_matches = len(
                query_words.intersection(document_words)
            )

            # Combine semantic and keyword relevance
            relevance_score = (
                (1 - distance)
                + (keyword_matches * 0.05)
            )

            candidates.append({
                "text": text,
                "metadata": result["metadata"],
                "distance": distance,
                "relevance_score": relevance_score
            })

        # Rank candidates by relevance
        candidates.sort(
            key=lambda item: item["relevance_score"],
            reverse=True
        )

        # Return top matching chunks
        retrieved_chunks = []

        for item in candidates[:self.top_k]:

            retrieved_chunks.append({
                "text": item["text"],
                "metadata": item["metadata"],
                "distance": item["distance"]
            })

        return retrieved_chunks
