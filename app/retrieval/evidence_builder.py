"""
Converts SearchCandidate into RetrievalResult.

Nothing in Phase 3 should ever see SearchCandidate.
"""

from __future__ import annotations

from typing import List

from models.retrieval_models import (
    RetrievalResult,
    SearchCandidate,
)


class EvidenceBuilder:

    def build(
        self,
        candidates: List[SearchCandidate],
    ) -> List[RetrievalResult]:

        evidence = []

        for candidate in candidates:

            score = max(

                candidate.rerank_score,

                candidate.final_score,

                candidate.vector_score,

                candidate.bm25_score,

            )

            evidence.append(

                RetrievalResult(

                    chunk_id=candidate.chunk_id,

                    document_type=candidate.document_type,

                    page_number=candidate.page_number,

                    text=candidate.text,

                    score=float(score)

                )

            )

        evidence.sort(

            key=lambda x: x.score,

            reverse=True

        )

        return evidence