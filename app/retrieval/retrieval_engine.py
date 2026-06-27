"""
retrieval_engine.py

Main retrieval orchestrator.

Pipeline

Query
    ↓
QueryProcessor
    ↓
HybridRetriever
    ↓
CrossEncoder Reranker
    ↓
EvidenceBuilder
    ↓
Return Evidence
"""

from __future__ import annotations

from typing import List

from models.retrieval_models import RetrievalResult

from retrieval.query_processor import QueryProcessor
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import BGECrossEncoder
from retrieval.evidence_builder import EvidenceBuilder
from config import RetrievalConfig


class RetrievalEngine:

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: BGECrossEncoder,
        config: RetrievalConfig,
    ):

        self.query_processor = QueryProcessor()

        self.hybrid_retriever = hybrid_retriever

        self.reranker = reranker

        self.evidence_builder = EvidenceBuilder()

        self.config = config

    def search(
        self,
        query: str,
    ) -> List[RetrievalResult]:

        # Step 1
        processed_query = self.query_processor.process(query)

        # Step 2
        candidates = self.hybrid_retriever.search(

            processed_query,

            top_k=self.config.hybrid_top_k

        )

        # Step 3
        reranked = self.reranker.rerank(

            processed_query,

            candidates,

            top_k=self.config.rerank_top_k

        )

        # Step 4
        return self.evidence_builder.build(reranked)