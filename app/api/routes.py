import os
import shutil
from pathlib import Path
from typing import List, Dict
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks

from app.api.schemas import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    ChatRequest,
    ChatResponse,
    DocumentListResponse,
    DocumentItem,
    UploadResponse,
    ReindexResponse,
    CodeExecutionRequest,
    CodeExecutionResponse
)
from app.generation.answer_generator import RAGAnswerGenerator
from app.generation.memory import ConversationMemory
from app.ingestion.pipeline import IngestionPipeline
from app.utils.helpers import execute_matplotlib_code
from app.core.config import settings
from app.core.logging_config import logger

router = APIRouter()

# Singletons for API lifecycle
generator = RAGAnswerGenerator()
pipeline = IngestionPipeline()
session_memories: Dict[str, ConversationMemory] = {}

def get_session_memory(session_id: str) -> ConversationMemory:
    if session_id not in session_memories:
        session_memories[session_id] = ConversationMemory(max_turns=6)
    return session_memories[session_id]

@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Returns the operational status and index size of the RAG system."""
    faiss_count = len(generator.retriever.vector_store.metadata) if generator.retriever.vector_store.metadata else 0
    bm25_count = len(generator.retriever.bm25_index.metadata) if generator.retriever.bm25_index.metadata else 0
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        faiss_indexed_chunks=faiss_count,
        bm25_indexed_chunks=bm25_count,
        embedding_model=settings.EMBEDDING_MODEL,
        llm_model=settings.LLM_MODEL
    )

@router.get("/documents", response_model=DocumentListResponse, tags=["Documents"])
def list_documents():
    """Lists all raw and processed documents available in the knowledge repository."""
    items = []

    # Documents directory
    if settings.DOCUMENTS_DIR.exists():
        for f in settings.DOCUMENTS_DIR.iterdir():
            if f.is_file():
                items.append(DocumentItem(
                    filename=f.name,
                    file_type=f.suffix.lstrip(".").lower(),
                    size_bytes=f.stat().st_size
                ))

    # Processed Transcripts directory
    target_dirs = [settings.PROCESSED_DIR, settings.LEGACY_JSON_DIR]
    seen = set()
    for d in target_dirs:
        if d.exists():
            for f in d.glob("*.json"):
                if f.name in seen:
                    continue
                seen.add(f.name)
                items.append(DocumentItem(
                    filename=f.name,
                    file_type="transcript_json",
                    size_bytes=f.stat().st_size
                ))

    return DocumentListResponse(total_documents=len(items), documents=items)

@router.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_documents(files: List[UploadFile] = File(...), auto_reindex: bool = True):
    """Upload one or more documents (PDF, DOCX, TXT, MD, CSV, JSON)."""
    saved_files = []
    settings.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for file in files:
        filename = file.filename
        ext = Path(filename).suffix.lower()
        if ext not in settings.SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format '{ext}'. Allowed: {settings.SUPPORTED_EXTENSIONS}"
            )

        if ext == ".json":
            save_path = settings.PROCESSED_DIR / filename
        elif ext in [".mp3", ".wav", ".m4a"]:
            save_path = settings.AUDIO_DIR / filename
        else:
            save_path = settings.DOCUMENTS_DIR / filename

        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_files.append(filename)

    indexed_count = 0
    if auto_reindex:
        logger.info("Auto-reindexing after document upload...")
        indexed_count = pipeline.run()
        # Reload indices in running retriever
        generator.retriever.vector_store.load_index()
        generator.retriever.bm25_index.load_index()

    return UploadResponse(
        message=f"Successfully uploaded {len(saved_files)} file(s).",
        uploaded_files=saved_files,
        indexed_chunks=indexed_count
    )

@router.post("/reindex", response_model=ReindexResponse, tags=["Documents"])
def reindex_all():
    """Trigger a full rebuild of FAISS and BM25 search indices."""
    total = pipeline.run()
    generator.retriever.vector_store.load_index()
    generator.retriever.bm25_index.load_index()
    return ReindexResponse(status="success", total_chunks_indexed=total)

@router.post("/query", response_model=QueryResponse, tags=["RAG"])
def stateless_query(request: QueryRequest):
    """Stateless single-turn grounded question answering."""
    res = generator.generate_answer(
        query=request.query,
        top_k=request.top_k or settings.TOP_K,
        retrieval_mode=request.retrieval_mode or settings.RETRIEVAL_MODE,
        min_score=request.min_score or settings.MIN_SIMILARITY_SCORE,
        use_memory=False
    )
    return QueryResponse(
        query=res["query"],
        rewritten_query=res["rewritten_query"],
        answer=res["answer"],
        confidence=res["confidence"],
        sources=res["sources"],
        code_blocks=res["code_blocks"]
    )

from fastapi.responses import StreamingResponse

@router.post("/query/stream", tags=["RAG"])
def stream_query(request: QueryRequest):
    """Stream grounded answer tokens in real-time."""
    token_gen, meta = generator.generate_answer_stream(
        query=request.query,
        top_k=request.top_k or settings.TOP_K,
        retrieval_mode=request.retrieval_mode or settings.RETRIEVAL_MODE,
        min_score=request.min_score or settings.MIN_SIMILARITY_SCORE,
        use_memory=False
    )
    return StreamingResponse(token_gen, media_type="text/plain")


@router.post("/chat", response_model=ChatResponse, tags=["RAG"])
def conversational_chat(request: ChatRequest):
    """Multi-turn conversational RAG with query rewriting and memory."""
    memory = get_session_memory(request.session_id or "default")
    generator.memory = memory

    res = generator.generate_answer(
        query=request.message,
        top_k=request.top_k or settings.TOP_K,
        retrieval_mode=request.retrieval_mode or settings.RETRIEVAL_MODE,
        use_memory=True
    )

    return ChatResponse(
        session_id=request.session_id or "default",
        user_message=res["query"],
        rewritten_query=res["rewritten_query"],
        assistant_response=res["answer"],
        confidence=res["confidence"],
        sources=res["sources"],
        code_blocks=res["code_blocks"]
    )

@router.post("/clear-chat", tags=["RAG"])
def clear_chat_history(session_id: str = "default"):
    """Reset the conversational memory buffer for a session."""
    if session_id in session_memories:
        session_memories[session_id].clear()
    return {"status": "cleared", "session_id": session_id}

@router.post("/execute-code", response_model=CodeExecutionResponse, tags=["Tools"])
def execute_code(request: CodeExecutionRequest):
    """Execute Python / Matplotlib code generated by the LLM."""
    fig, stdout = execute_matplotlib_code(request.code)
    has_plot = fig is not None and len(fig.get_axes()) > 0
    return CodeExecutionResponse(
        success=True,
        output=stdout,
        has_plot=has_plot
    )
