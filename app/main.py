from config import (
    SPEC_PDF,
    SUBMITTAL_PDF,
    CHROMA_PATH,
    EMBEDDING_MODEL,
    RetrievalConfig
)

from ingestion.docling_loader import DoclingLoader

from indexing.chunker import chunk_text

from embeddings.embedder import Embedder

from vectordb.chroma_manager import ChromaManager

from config import RetrievalConfig
from retrieval.bm25_retriever import BM25Retriever
from retrieval.vector_retriever import VectorRetriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import BGECrossEncoder
from retrieval.retrieval_engine import RetrievalEngine


def load_document(
    pdf_path: str,
    document_type: str,
):
    """
    Phase 1 Pipeline

    PDF
      ↓
    Docling
      ↓
    Markdown
      ↓
    Chunk
      ↓
    Return chunks
    """

    loader = DoclingLoader()

    # Load the PDF
    document = loader.load_pdf(pdf_path)

    # Convert to markdown
    markdown = document.export_to_markdown()

    # Chunk the markdown
    chunks = chunk_text(
        markdown,
        document_type=document_type,
    )

    return chunks

def main():

    print("=" * 80)
    print("Loading documents...")
    print("=" * 80)

    spec_chunks = load_document(
        SPEC_PDF,
        "spec"
    )

    submittal_chunks = load_document(
        SUBMITTAL_PDF,
        "submittal"
    )

    all_chunks = spec_chunks + submittal_chunks

    print(f"Total Chunks : {len(all_chunks)}")

    # -------------------------------------------------
    # Embedding
    # -------------------------------------------------

    embedder = Embedder(EMBEDDING_MODEL)

    chroma = ChromaManager(CHROMA_PATH)

    print("Creating embeddings...")

    embeddings = embedder.encode(

        [chunk["text"] for chunk in all_chunks]

    )

    chroma.add_chunks(

        chunks=all_chunks,

        embeddings=embeddings

    )

    # -------------------------------------------------
    # BM25
    # -------------------------------------------------

    bm25 = BM25Retriever()

    bm25.build_index(all_chunks)

    # -------------------------------------------------
    # Vector
    # -------------------------------------------------

    vector = VectorRetriever(

        chroma,

        embedder

    )

    # -------------------------------------------------
    # Hybrid
    # -------------------------------------------------

    config = RetrievalConfig()

    hybrid = HybridRetriever(

        bm25=bm25,

        vector=vector,

        bm25_weight=config.bm25_weight,

        vector_weight=config.vector_weight

    )

    # -------------------------------------------------
    # Reranker
    # -------------------------------------------------

    reranker = BGECrossEncoder(

        model_name=config.reranker_model

    )

    # -------------------------------------------------
    # Engine
    # -------------------------------------------------

    engine = RetrievalEngine(

        hybrid_retriever=hybrid,

        reranker=reranker,

        config=config

    )

    print("=" * 80)

    while True:

        query = input("\nEnter Query (or exit): ")

        if query.lower() == "exit":

            break

        results = engine.search(query)

        print()

        print("=" * 80)

        print("RESULTS")

        print("=" * 80)

        for idx, result in enumerate(results, start=1):

            print(f"\n[{idx}]")

            print(f"Document : {result.document_type}")

            print(f"Page     : {result.page_number}")

            print(f"Score    : {result.score:.4f}")

            print("-" * 80)

            print(result.text[:1000])

            print("-" * 80)


if __name__ == "__main__":

    main()