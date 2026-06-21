from pydantic import BaseModel

class DocumentChunk(BaseModel):
    chunk_id : int
    document_type: str
    page_number: int
    text: str

    