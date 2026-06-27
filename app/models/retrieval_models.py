from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class SearchCandidate(BaseModel):
    """
    Internal retrieval candidate.
    """

    chunk_id: str

    document_type: str

    page_number: int

    text: str

    bm25_score: float = 0.0

    vector_score: float = 0.0

    rerank_score: float = 0.0

    final_score: float = 0.0


class RetrievalResult(BaseModel):
    """
    Final evidence returned to the LLM.
    """

    chunk_id: str

    document_type: str

    page_number: int

    text: str

    score: float