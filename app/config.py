from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SPEC_PDF = BASE_DIR / "data" / "spec.pdf"
SUBMITTAL_PDF = BASE_DIR / "data" / "submittal.pdf"

CHROMA_PATH = BASE_DIR / "chroma_db"

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200