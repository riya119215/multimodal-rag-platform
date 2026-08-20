import os
import glob
from pathlib import Path
from typing import List, Union, Optional
from app.ingestion.metadata import Document, DocumentChunk
from app.ingestion.loaders import DocumentLoaderFactory, AudioTranscriptJSONLoader
from app.ingestion.chunking import RecursiveCharacterChunker
from app.core.config import settings
from app.core.logging_config import logger

class IngestionPipeline:
    """
    Coordinates end-to-end ingestion:
    1. Discovers files across data/documents and data/processed / jsons
    2. Loads documents via specialized loaders
    3. Segments into semantically coherent chunks
    4. Builds and persists both FAISS Dense and BM25 Sparse search indices
    """
    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP
    ):
        self.chunker = RecursiveCharacterChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def load_single_file(self, file_path: Union[str, Path]) -> List[DocumentChunk]:
        """Load and chunk a single file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        loaded = DocumentLoaderFactory.load_document(path)
        
        # If loader already returned DocumentChunk (e.g. transcript json), return directly
        if loaded and isinstance(loaded[0], DocumentChunk):
            return loaded
        
        # Otherwise chunk the raw Document objects
        return self.chunker.chunk_documents(loaded)

    def collect_all_chunks(
        self,
        docs_dir: Optional[Path] = None,
        json_dir: Optional[Path] = None
    ) -> List[DocumentChunk]:
        """Collect and chunk all documents and transcript JSONs."""
        docs_path = docs_dir or settings.DOCUMENTS_DIR
        json_path = json_dir or settings.PROCESSED_DIR
        legacy_json_path = settings.LEGACY_JSON_DIR

        all_chunks: List[DocumentChunk] = []

        # 1. Process regular documents (PDF, DOCX, TXT, MD, CSV)
        if docs_path.exists():
            for ext in settings.SUPPORTED_EXTENSIONS:
                if ext in [".mp3", ".wav", ".m4a", ".json"]:
                    continue
                for file_path in docs_path.glob(f"*{ext}"):
                    try:
                        chunks = self.load_single_file(file_path)
                        all_chunks.extend(chunks)
                        logger.info(f"Loaded {len(chunks)} chunks from {file_path.name}")
                    except Exception as e:
                        logger.error(f"Error loading {file_path.name}: {e}")

        # 2. Process Transcript JSONs from data/processed
        target_json_dirs = [json_path]
        if legacy_json_path.exists() and legacy_json_path != json_path:
            target_json_dirs.append(legacy_json_path)

        seen_sources = set()
        for jdir in target_json_dirs:
            if jdir.exists():
                for json_file in jdir.glob("*.json"):
                    if json_file.name in seen_sources:
                        continue
                    try:
                        loader = AudioTranscriptJSONLoader()
                        chunks = loader.load(json_file)
                        all_chunks.extend(chunks)
                        seen_sources.add(json_file.name)
                        logger.info(f"Loaded {len(chunks)} transcript chunks from {json_file.name}")
                    except Exception as e:
                        logger.error(f"Error loading transcript {json_file.name}: {e}")

        # Re-index sequential chunk IDs
        for i, chunk in enumerate(all_chunks):
            chunk.chunk_id = i

        logger.info(f"Total unified chunks collected: {len(all_chunks)}")
        return all_chunks

    def run(self) -> int:
        """Run full ingestion and rebuild both vectorstore and BM25 index."""
        logger.info("Starting End-to-End Ingestion Pipeline...")
        from app.retrieval.vector_store import VectorStore
        from app.retrieval.bm25_store import BM25Index

        chunks = self.collect_all_chunks()
        if not chunks:
            logger.warning("No document or transcript chunks found to index.")
            return 0

        # Build Dense FAISS Index
        vs = VectorStore()
        vs.build_index(chunks)

        # Build Sparse BM25 Index
        bm25 = BM25Index()
        bm25.build_index(chunks)

        logger.info(f"Successfully indexed {len(chunks)} chunks in VectorStore & BM25!")
        return len(chunks)
