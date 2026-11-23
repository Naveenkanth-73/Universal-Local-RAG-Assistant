
import re


def split_long_text(text, chunk_size, overlap):
    """
    Split long text into smaller chunks.
    Preserve word boundaries and maintain overlap.
    """

    words = text.split()

    if not words:
        return []

    chunks = []
    current_words = []
    current_length = 0

    for word in words:

        additional_length = len(word) + (
            1 if current_words else 0
        )

        if current_length + additional_length <= chunk_size:
            current_words.append(word)
            current_length += additional_length

        else:
            if current_words:
                chunks.append(" ".join(current_words))

            # Keep overlapping complete words.
            overlap_words = []
            overlap_length = 0

            for previous_word in reversed(current_words):
                word_length = len(previous_word) + (
                    1 if overlap_words else 0
                )

                if overlap_length + word_length > overlap:
                    break

                overlap_words.insert(0, previous_word)
                overlap_length += word_length

            current_words = overlap_words
            current_length = len(" ".join(current_words))

            # Add the next word.
            if len(word) > chunk_size:
                chunks.append(word)
                current_words = []
                current_length = 0
            else:
                if current_words:
                    current_length += 1

                current_words.append(word)
                current_length += len(word)

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def chunk_documents(documents, chunk_size=1000, overlap=150):
    """
    Convert extracted document records into smaller chunks.

    Each chunk retains its original document metadata.
    """

    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    if not isinstance(overlap, int):
        raise ValueError("overlap must be an integer")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size"
        )

    all_chunks = []

    for document in documents:

        if "text" not in document or "metadata" not in document:
            raise ValueError(
                "Each document must contain text and metadata"
            )

        text = document["text"].strip()
        metadata = document["metadata"].copy()

        if not text:
            continue

        # Split document using paragraph boundaries.
        paragraphs = re.split(r"\n\s*\n", text)

        document_chunks = []
        current_chunk = ""

        for paragraph in paragraphs:

            paragraph = paragraph.strip()

            if not paragraph:
                continue

            # Split paragraphs larger than the chunk limit.
            if len(paragraph) > chunk_size:

                if current_chunk:
                    document_chunks.append(current_chunk)
                    current_chunk = ""

                document_chunks.extend(
                    split_long_text(
                        paragraph,
                        chunk_size,
                        overlap
                    )
                )

                continue

            candidate = (
                current_chunk + "\n\n" + paragraph
                if current_chunk
                else paragraph
            )

            if len(candidate) <= chunk_size:
                current_chunk = candidate

            else:
                if current_chunk:
                    document_chunks.append(current_chunk)

                current_chunk = paragraph

        if current_chunk:
            document_chunks.append(current_chunk)

        # Attach original metadata to each chunk.
        for index, chunk_text in enumerate(document_chunks):

            chunk_metadata = metadata.copy()

            chunk_metadata["chunk_index"] = index

            chunk_metadata["chunk_size"] = len(chunk_text)

            all_chunks.append({
                "text": chunk_text,
                "metadata": chunk_metadata
            })

    return all_chunks
