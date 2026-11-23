
import ollama


MODEL_NAME = "nomic-embed-text:latest"


def embed_text(text, task="document"):
    """
    Convert text into a numerical embedding vector.

    task:
        document = text being stored
        query = user question being searched
    """

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Text cannot be empty")

    if task == "document":
        prepared_text = "search_document: " + text

    elif task == "query":
        prepared_text = "search_query: " + text

    else:
        raise ValueError("Task must be 'document' or 'query'")

    response = ollama.embed(
        model=MODEL_NAME,
        input=prepared_text
    )

    return response.embeddings[0]


def embed_chunks(chunks, batch_size=32):
    """
    Generate embeddings for document chunks in batches.

    Preserve original text and metadata.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    embedded_chunks = []

    for start in range(0, len(chunks), batch_size):

        batch = chunks[start:start + batch_size]

        valid_chunks = [
            chunk for chunk in batch
            if chunk.get("text", "").strip()
        ]

        if not valid_chunks:
            continue

        texts = [
            "search_document: " + chunk["text"]
            for chunk in valid_chunks
        ]

        response = ollama.embed(
            model=MODEL_NAME,
            input=texts
        )

        if len(response.embeddings) != len(valid_chunks):
            raise RuntimeError(
                "Embedding count does not match chunk count"
            )

        for chunk, vector in zip(
            valid_chunks,
            response.embeddings
        ):
            embedded_chunks.append({
                "text": chunk["text"],
                "metadata": chunk["metadata"].copy(),
                "embedding": vector
            })

    return embedded_chunks
