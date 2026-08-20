from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import hashlib
import time

@dataclass
class Document:
    """Represents an ingested raw document or transcript."""
    content: str
    source_file: str
    doc_type: str  # pdf, docx, txt, md, csv, audio_transcript
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: Optional[str] = None

    def __post_init__(self):
        if not self.doc_id:
            raw = f"{self.source_file}_{self.content[:100]}_{time.time()}"
            self.doc_id = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]

@dataclass
class DocumentChunk:
    """Represents a chunk of text with rich metadata for retrieval."""
    chunk_id: int
    text: str
    source_file: str
    doc_type: str
    doc_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Audio/Video specific attributes (if applicable)
    video_number: Optional[str] = "N/A"
    title: Optional[str] = "Untitled"
    start: Optional[float] = 0.0
    end: Optional[float] = 0.0
    start_formatted: Optional[str] = "00:00"
    end_formatted: Optional[str] = "00:00"
    
    # Document specific attributes (if applicable)
    page_number: Optional[int] = None
    section: Optional[str] = None
    
    # Small-to-Big / Hierarchical Parent Context
    parent_context: Optional[str] = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk into a dictionary representation."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "parent_context": self.parent_context or self.text,
            "source_file": self.source_file,
            "doc_type": self.doc_type,
            "doc_id": self.doc_id,
            "video_number": self.video_number,
            "title": self.title,
            "start": self.start,
            "end": self.end,
            "start_formatted": self.start_formatted,
            "end_formatted": self.end_formatted,
            "page_number": self.page_number,
            "section": self.section,
            "metadata": self.metadata
        }

