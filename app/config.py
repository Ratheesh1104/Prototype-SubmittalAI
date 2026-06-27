from pathlib import Path
from dataclasses import dataclass

# ==========================================================
# Project Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

SPEC_PDF = DATA_DIR / "spec.pdf"
SUBMITTAL_PDF = DATA_DIR / "submittal.pdf"

CHROMA_PATH = BASE_DIR / "chroma_db"

# ==========================================================
# AI Models
# ==========================================================

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

RERANKER_MODEL = "BAAI/bge-reranker-large"

LLM_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"

# ==========================================================
# Chunking
# ==========================================================

CHUNK_SIZE = 1200

CHUNK_OVERLAP = 200

# ==========================================================
# Retrieval Configuration
# ==========================================================

@dataclass(frozen=True)
class RetrievalConfig:

    # Initial retrieval
    bm25_top_k: int = 20
    vector_top_k: int = 20
    hybrid_top_k: int = 20

    # Final reranked results
    rerank_top_k: int = 5

    # Hybrid search weights
    bm25_weight: float = 0.40
    vector_weight: float = 0.60

    # Cross Encoder
    reranker_model: str = RERANKER_MODEL