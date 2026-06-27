"""
Hybrid Retriever

Responsibilities
----------------
1. Execute BM25 retrieval
2. Execute Vector retrieval
3. Merge candidates
4. Normalize scores
5. Produce final hybrid score
"""

from __future__ import annotations

from typing import List

from models.retrieval_models import SearchCandidate

from retrieval.merge import CandidateMerger

from retrieval.base_retriever import BaseRetriever


class HybridRetriever:

    def __init__(

        self,

        bm25: BaseRetriever,

        vector: BaseRetriever,

        bm25_weight: float,

        vector_weight: float

    ):

        self.bm25 = bm25

        self.vector = vector

        self.bm25_weight = bm25_weight

        self.vector_weight = vector_weight

    @staticmethod
    def _normalize(

        values: List[float]

    ) -> List[float]:

        if not values:

            return values

        maximum = max(values)

        minimum = min(values)

        if maximum == minimum:

            return [1.0] * len(values)

        return [

            (v - minimum) / (maximum - minimum)

            for v in values

        ]

    def search(

        self,

        query: str,

        top_k: int = 20

    ) -> List[SearchCandidate]:

        bm25_results = self.bm25.search(

            query,

            top_k

        )

        vector_results = self.vector.search(

            query,

            top_k

        )

        merged = CandidateMerger.merge(

            bm25_results,

            vector_results

        )

        bm25_scores = [

            c.bm25_score

            for c in merged

        ]

        vector_scores = [

            c.vector_score

            for c in merged

        ]

        bm25_norm = self._normalize(

            bm25_scores

        )

        vector_norm = self._normalize(

            vector_scores

        )

        for i, candidate in enumerate(merged):

            candidate.final_score = (

                self.bm25_weight * bm25_norm[i]

                +

                self.vector_weight * vector_norm[i]

            )

        merged.sort(

            key=lambda x: x.final_score,

            reverse=True

        )

        return merged