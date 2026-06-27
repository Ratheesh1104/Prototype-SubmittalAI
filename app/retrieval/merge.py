from __future__ import annotations

from typing import Dict
from typing import List

from models.retrieval_models import SearchCandidate


class CandidateMerger:

    @staticmethod
    def merge(
        bm25_results: List[SearchCandidate],
        vector_results: List[SearchCandidate],
    ) -> List[SearchCandidate]:

        merged: Dict[str, SearchCandidate] = {}

        for candidate in bm25_results:

            merged[candidate.chunk_id] = candidate

        for candidate in vector_results:

            if candidate.chunk_id in merged:

                merged[candidate.chunk_id].vector_score = (
                    candidate.vector_score
                )

            else:

                merged[candidate.chunk_id] = candidate

        return list(merged.values())