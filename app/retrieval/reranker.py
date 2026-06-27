"""
BGE Cross Encoder Reranker

Responsibilities
----------------
1. Accept search candidates
2. Cross-encode query + candidate
3. Compute rerank scores
4. Return Top-K candidates
"""

from __future__ import annotations

from typing import List

from FlagEmbedding import FlagReranker

from models.retrieval_models import SearchCandidate


class BGECrossEncoder:

    def __init__(

        self,

        model_name: str = "BAAI/bge-reranker-large",

        use_fp16: bool = False

    ):

        self.model = FlagReranker(

            model_name,

            use_fp16=use_fp16

        )

    def rerank(

        self,

        query: str,

        candidates: List[SearchCandidate],

        top_k: int = 5

    ) -> List[SearchCandidate]:

        if not candidates:

            return []

        pairs = [

            [

                query,

                c.text

            ]

            for c in candidates

        ]

        scores = self.model.compute_score(

            pairs

        )

        for candidate, score in zip(

            candidates,

            scores

        ):

            candidate.rerank_score = float(score)

        candidates.sort(

            key=lambda x: x.rerank_score,

            reverse=True

        )

        return candidates[:top_k]