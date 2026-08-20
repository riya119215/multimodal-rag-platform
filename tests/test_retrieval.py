import pytest
from pathlib import Path
from app.ingestion.metadata import DocumentChunk
from app.retrieval.vector_store import VectorStore
from app.retrieval.bm25_store import BM25Index
from app.retrieval.hybrid_search import HybridRetriever

@pytest.fixture
def sample_chunks():
    return [
        DocumentChunk(
            chunk_id=0,
            text="To plot a bar chart in Matplotlib, use the plt.bar() function passing x and height values.",
            source_file="matplotlib_bars.txt",
            doc_type="text",
            doc_id="doc_1"
        ),
        DocumentChunk(
            chunk_id=1,
            text="Subplots allow multiple figures side-by-side using plt.subplots(nrows, ncols).",
            source_file="matplotlib_subplots.txt",
            doc_type="text",
            doc_id="doc_2"
        ),
        DocumentChunk(
            chunk_id=2,
            text="Pandas dataframes provide fast tabular data analysis and data manipulation routines.",
            source_file="pandas_guide.txt",
            doc_type="text",
            doc_id="doc_3"
        )
    ]

def test_vector_store_build_and_search(tmp_path, sample_chunks):
    vs = VectorStore(vector_store_dir=tmp_path)
    vs.build_index(sample_chunks)
    assert vs.index is not None
    assert vs.index.ntotal == 3

    results = vs.search("How do I make a bar chart?", top_k=2)
    assert len(results) > 0
    assert "plt.bar" in results[0]["text"]

def test_bm25_build_and_search(tmp_path, sample_chunks):
    bm25 = BM25Index(vector_store_dir=tmp_path)
    bm25.build_index(sample_chunks)
    assert bm25.bm25 is not None

    results = bm25.search("subplots nrows ncols", top_k=2)
    assert len(results) > 0
    assert "subplots" in results[0]["text"].lower()

def test_hybrid_search_rrf(tmp_path, sample_chunks):
    vs = VectorStore(vector_store_dir=tmp_path)
    vs.build_index(sample_chunks)
    bm25 = BM25Index(vector_store_dir=tmp_path)
    bm25.build_index(sample_chunks)

    retriever = HybridRetriever(vector_store=vs, bm25_index=bm25)
    results = retriever.retrieve("plt.bar chart", top_k=2, mode="hybrid")
    assert len(results) > 0
    assert "rrf_score" in results[0]
