import re
from typing import List
from app.ingestion.metadata import Document, DocumentChunk
from app.core.config import settings
from app.core.logging_config import logger

class RecursiveCharacterChunker:
    """
    Intelligently chunks text by recursively splitting along paragraph, sentence,
    and word boundaries to maintain semantic cohesion.
    """
    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
        separators: List[str] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Split text using highest priority separator."""
        final_chunks = []
        separator = separators[-1]
        new_separators = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator else list(text)

        good_splits = []
        for s in splits:
            if separator and s:
                # Add separator back if not end
                piece = s if separator == "\n" or separator == "\n\n" else s + separator
            else:
                piece = s
            
            if len(piece) < self.chunk_size:
                good_splits.append(piece)
            else:
                if new_separators:
                    nested = self._split_text(piece, new_separators)
                    good_splits.extend(nested)
                else:
                    good_splits.append(piece)

        # Merge pieces up to chunk_size respecting chunk_overlap
        current_chunk = []
        current_len = 0

        for piece in good_splits:
            if current_len + len(piece) > self.chunk_size and current_chunk:
                merged = "".join(current_chunk).strip()
                if merged:
                    final_chunks.append(merged)
                
                # Keep overlap pieces
                overlap_len = 0
                overlap_chunk = []
                for p in reversed(current_chunk):
                    if overlap_len + len(p) <= self.chunk_overlap:
                        overlap_chunk.insert(0, p)
                        overlap_len += len(p)
                    else:
                        break
                current_chunk = overlap_chunk
                current_len = overlap_len

            current_chunk.append(piece)
            current_len += len(piece)

        if current_chunk:
            merged = "".join(current_chunk).strip()
            if merged:
                final_chunks.append(merged)

        return final_chunks

    def chunk_document(self, document: Document, start_chunk_id: int = 0) -> List[DocumentChunk]:
        """Convert a single Document into a list of DocumentChunks."""
        raw_text = document.content
        if not raw_text.strip():
            return []

        text_chunks = self._split_text(raw_text, self.separators)
        chunk_objects = []

        for i, chunk_text in enumerate(text_chunks):
            # Construct parent context window from adjacent chunks (Small-to-Big)
            prev_chunk = text_chunks[i - 1] if i > 0 else ""
            next_chunk = text_chunks[i + 1] if i < len(text_chunks) - 1 else ""
            parent_window = f"{prev_chunk}\n{chunk_text}\n{next_chunk}".strip()

            chunk_obj = DocumentChunk(
                chunk_id=start_chunk_id + i,
                text=chunk_text,
                parent_context=parent_window,
                source_file=document.source_file,
                doc_type=document.doc_type,
                doc_id=document.doc_id,
                page_number=document.metadata.get("page_number"),
                metadata=document.metadata
            )
            chunk_objects.append(chunk_obj)

        return chunk_objects

    def chunk_documents(self, documents: List[Document]) -> List[DocumentChunk]:
        """Chunk a collection of documents."""
        all_chunks = []
        current_id = 0
        for doc in documents:
            chunks = self.chunk_document(doc, start_chunk_id=current_id)
            all_chunks.extend(chunks)
            current_id += len(chunks)
        
        logger.info(f"Chunked {len(documents)} documents into {len(all_chunks)} chunks.")
        return all_chunks
