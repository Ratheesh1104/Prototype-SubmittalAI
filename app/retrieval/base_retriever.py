"""
Base interface for all retrievers.

Every retriever must implement:

1. build_index()
2. search()
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod

from typing import List

from models.retrieval_models import SearchCandidate



class BaseRetriever(ABC):

    @abstractmethod
    def build_index(self, chunks: List[dict]) -> None:
        """
        Build retrieval index.
        """
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[SearchCandidate]:
        """
        Retrieve Top-K candidates.
        """
        raise NotImplementedError