import os

from config import *
from ingestion.docling_loader import DoclingLoader

# from indexing.page_indexer import build_page_index
from indexing.chunker import chunk_text

from embeddings.embedder import Embedder

from vectordb.chroma_manager import ChromaManager


def process_document(
    loader,
    embedder,
    chroma,
    pdf_path,
    doc_type
):

    document = loader.load_pdf(pdf_path)
    print("Pages detected:", len(document.pages))

    markdown_text = document.export_to_markdown()
    print("Markdown length:", len(markdown_text))

    chunks = chunk_text(
        markdown_text,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
        doc_type
    )

    embeddings = embedder.encode(
        [c["text"] for c in chunks]
    )

    chroma.add_chunks(
        chunks,
        embeddings
    )

    print(
        f"{doc_type}: "
        f"{len(chunks)} chunks indexed"
    )


def main():

    loader = DoclingLoader()

    embedder = Embedder(
        EMBEDDING_MODEL
    )

    chroma = ChromaManager(
        CHROMA_PATH
    )

    process_document(
        loader,
        embedder,
        chroma,
        SPEC_PDF,
        "spec"
    )

    process_document(
        loader,
        embedder,
        chroma,
        SUBMITTAL_PDF,
        "submittal"
    )

    print("Phase 1 Complete")


if __name__ == "__main__":
    main()