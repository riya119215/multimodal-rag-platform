from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str
    faiss_indexed_chunks: int
    bm25_indexed_chunks: int
    embedding_model: str
    llm_model: str

class QueryRequest(BaseModel):
    query: str = Field(..., description="User question to answer using knowledge base")
    top_k: Optional[int] = Field(default=5, ge=1, le=20)
    retrieval_mode: Optional[str] = Field(default="hybrid", description="hybrid, dense, or sparse")
    min_score: Optional[float] = Field(default=0.10, ge=0.0, le=1.0)

class SourceChunk(BaseModel):
    chunk_id: int
    text: str
    source_file: str
    doc_type: str
    score: Optional[float] = None
    rerank_score: Optional[float] = None
    start_formatted: Optional[str] = None
    end_formatted: Optional[str] = None
    page_number: Optional[int] = None
    title: Optional[str] = None
    video_number: Optional[str] = None

class ConfidenceScore(BaseModel):
    score: float
    percentage: str
    level: str
    is_grounded: bool
    breakdown: Dict[str, Any]

class QueryResponse(BaseModel):
    query: str
    rewritten_query: str
    answer: str
    confidence: ConfidenceScore
    sources: List[Dict[str, Any]]
    code_blocks: List[str]

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"
    top_k: Optional[int] = Field(default=5, ge=1, le=20)
    retrieval_mode: Optional[str] = Field(default="hybrid")

class ChatResponse(BaseModel):
    session_id: str
    user_message: str
    rewritten_query: str
    assistant_response: str
    confidence: ConfidenceScore
    sources: List[Dict[str, Any]]
    code_blocks: List[str]

class DocumentItem(BaseModel):
    filename: str
    file_type: str
    size_bytes: int
    chunks_count: Optional[int] = 0

class DocumentListResponse(BaseModel):
    total_documents: int
    documents: List[DocumentItem]

class UploadResponse(BaseModel):
    message: str
    uploaded_files: List[str]
    indexed_chunks: int

class ReindexResponse(BaseModel):
    status: str
    total_chunks_indexed: int

class CodeExecutionRequest(BaseModel):
    code: str

class CodeExecutionResponse(BaseModel):
    success: bool
    output: str
    has_plot: bool
