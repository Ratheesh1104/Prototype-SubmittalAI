"""
Vector Retriever

Responsibilities
----------------
1. Query ChromaDB
2. Return semantic search results
3. Preserve metadata
4. Return RetrievalResult objects
"""

from __future__ import annotations

from typing import List

from models.retrieval_models import SearchCandidate
from vectordb.chroma_manager import ChromaManager
from embeddings.embedder import Embedder
from retrieval.base_retriever import BaseRetriever


class VectorRetriever(BaseRetriever):

    def __init__(
        self,
        chroma: ChromaManager,
        embedder: Embedder
    ):

        self.chroma = chroma
        self.embedder = embedder

    def build_index(
        self,
        chunks: List[dict]
    ) -> None:
        """
        ChromaDB already stores embeddings.
        No additional indexing is required.
        """
        return

    def search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[SearchCandidate]:

        embedding = self.embedder.encode([query])[0]

        result = self.chroma.query(
            embedding.tolist(),
            top_k
        )

        if not result["ids"] or not result["ids"][0]:
            return []

        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        ids = result["ids"][0]
        distances = result["distances"][0]

        results = []

        for doc_id, doc, meta, dist in zip(
            ids,
            documents,
            metadatas,
            distances
        ):

            results.append(
                SearchCandidate(
                    chunk_id=doc_id,
                    document_type=meta["document_type"],
                    page_number=meta["page_number"],
                    text=doc,
                    score=1.0 - float(dist)
                )
            )

        return results