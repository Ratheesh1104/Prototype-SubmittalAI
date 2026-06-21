import uuid


def chunk_text(
    text,
    chunk_size=1200,
    overlap=200,
    document_type="unknown"
):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(
            {
                "chunk_id": str(uuid.uuid4()),
                "document_type": document_type,
                "page_number": 1,
                "text": chunk
            }
        )

        start += chunk_size - overlap

    return chunks