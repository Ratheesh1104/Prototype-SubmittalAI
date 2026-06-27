"""
BM25 Retriever

Responsibilities
----------------
1. Build BM25 index from Phase 1 chunks
2. Retrieve Top-K relevant chunks
3. Preserve metadata
4. Return RetrievalResult objects
"""

from __future__ import annotations

from typing import List
from rank_bm25 import BM25Okapi

from models.retrieval_models import SearchCandidate
from retrieval.base_retriever import BaseRetriever



class BM25Retriever(BaseRetriever):

    def __init__(self):

        self._bm25 = None

        self._chunks = []

        self._tokenized_docs = []

    @staticmethod
    def _tokenize(text: str) -> List[str]:

        return text.lower().split()

    def build_index(self, chunks: List[dict]) -> None:
        """
        Build BM25 index.

        Parameters
        ----------
        chunks : List[dict]

        Expected format

        {
            "chunk_id":"",
            "text":"",
            "page_number":1,
            "document_type":"spec"
        }
        """

        self._chunks = chunks

        self._tokenized_docs = [

            self._tokenize(chunk["text"])

            for chunk in chunks

        ]

        self._bm25 = BM25Okapi(self._tokenized_docs)

    def search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[SearchCandidate]:

        if self._bm25 is None:

            raise RuntimeError(
                "BM25 index has not been built."
            )

        query_tokens = self._tokenize(query)

        scores = self._bm25.get_scores(query_tokens)

        ranked = sorted(

            enumerate(scores),

            key=lambda x: x[1],

            reverse=True

        )

        results = []

        for index, score in ranked[:top_k]:

            chunk = self._chunks[index]

            results.append(

                SearchCandidate(

                    chunk_id=chunk["chunk_id"],
                    document_type=chunk["document_type"],
                    page_number=chunk["page_number"],
                    text=chunk["text"],
                    bm25_score=float(score)

                )

            )

        return results