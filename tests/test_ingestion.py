import pytest
from pathlib import Path
from app.ingestion.metadata import Document, DocumentChunk
from app.ingestion.chunking import RecursiveCharacterChunker
from app.ingestion.loaders import TextLoader, DocumentLoaderFactory

def test_document_metadata_creation():
    doc = Document(
        content="Sample document content for testing.",
        source_file="test.txt",
        doc_type="text"
    )
    assert doc.doc_id is not None
    assert doc.source_file == "test.txt"
    assert doc.doc_type == "text"

def test_recursive_chunker_basic():
    chunker = RecursiveCharacterChunker(chunk_size=50, chunk_overlap=10)
    doc = Document(
        content="Artificial intelligence is transforming industries. Natural Language Processing enables semantic understanding. Retrieval Augmented Generation connects LLMs to factual data.",
        source_file="ai_overview.txt",
        doc_type="text"
    )
    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 2
    for c in chunks:
        assert isinstance(c, DocumentChunk)
        assert c.source_file == "ai_overview.txt"
        assert len(c.text) <= 80  # Reasonable boundary

def test_loader_factory_text_file(tmp_path):
    sample_file = tmp_path / "sample.md"
    sample_file.write_text("# Overview\nThis is a test markdown file.", encoding="utf-8")

    loader = DocumentLoaderFactory.get_loader(sample_file)
    docs = loader.load(sample_file)
    assert len(docs) == 1
    assert docs[0].doc_type == "markdown"
    assert "test markdown file" in docs[0].content
