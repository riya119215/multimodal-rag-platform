from typing import List, Dict, Any, Optional
from app.retrieval.vector_store import VectorStore
from app.retrieval.bm25_store import BM25Index
from app.retrieval.reranker import ReRanker
from app.core.config import settings
from app.core.logging_config import logger

class HybridRetriever:
    """
    Combines FAISS Dense vector search and BM25 Sparse keyword search
    using Reciprocal Rank Fusion (RRF) and Cross-Encoder Re-ranking.
    """
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        bm25_index: Optional[BM25Index] = None,
        reranker: Optional[ReRanker] = None,
        rrf_k: int = settings.RRF_K,
        dense_weight: float = settings.DENSE_WEIGHT,
        sparse_weight: float = settings.SPARSE_WEIGHT
    ):
        self.vector_store = vector_store or VectorStore()
        self.bm25_index = bm25_index or BM25Index()
        self.reranker = reranker or ReRanker()
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    @staticmethod
    def extract_self_query_filters(query: str) -> Dict[str, Any]:
        """Extract explicit filters (e.g. video number, doc type) from user natural language query."""
        import re
        filters = {}
        # Video number pattern (e.g., 'video 2', 'video #3', 'part 1', 'audio 4')
        v_match = re.search(r"\b(?:video|part|audio|tutorial)\s*#?\s*(\d+)\b", query, re.IGNORECASE)
        if v_match:
            filters["video_number"] = v_match.group(1)

        # Document type pattern
        if re.search(r"\b(?:pdf|document)\b", query, re.IGNORECASE):
            filters["doc_type"] = "pdf"
        elif re.search(r"\b(?:audio|video|transcript|mp3)\b", query, re.IGNORECASE):
            filters["doc_type"] = "audio_transcript"
        elif re.search(r"\b(?:csv|table|data)\b", query, re.IGNORECASE):
            filters["doc_type"] = "csv"
        elif re.search(r"\b(?:word|docx)\b", query, re.IGNORECASE):
            filters["doc_type"] = "docx"

        return filters

    def retrieve(
        self,
        query: str,
        top_k: int = settings.TOP_K,
        mode: str = settings.RETRIEVAL_MODE,  # hybrid, dense, sparse
        min_score: float = settings.MIN_SIMILARITY_SCORE,
        filter_metadata: Optional[Dict[str, Any]] = None,
        auto_extract_filters: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Execute multi-stage retrieval:
        1. Auto-extract metadata filters (Self-querying) if enabled
        2. Fetch candidates from VectorStore and/or BM25 with metadata filtering
        3. Fuse ranks via Reciprocal Rank Fusion (RRF)
        4. Re-rank top candidates with Cross-Encoder
        """
        if not query.strip():
            return []

        active_filters = dict(filter_metadata) if filter_metadata else {}
        if auto_extract_filters:
            inferred = self.extract_self_query_filters(query)
            active_filters.update(inferred)

        # Candidate pool size before re-ranking
        candidate_k = max(top_k * 3, 10)

        dense_results: List[Dict[str, Any]] = []
        sparse_results: List[Dict[str, Any]] = []

        if mode in ["dense", "hybrid"]:
            dense_results = self.vector_store.search(
                query, top_k=candidate_k, min_score=min_score, filter_metadata=active_filters or None
            )

        if mode in ["sparse", "hybrid"]:
            sparse_results = self.bm25_index.search(
                query, top_k=candidate_k, filter_metadata=active_filters or None
            )

        # If strict filtering yielded 0 results, retry without filters as graceful fallback
        if active_filters and not dense_results and not sparse_results:
            logger.info("Filtered search returned 0 results. Retrying with global search fallback...")
            if mode in ["dense", "hybrid"]:
                dense_results = self.vector_store.search(query, top_k=candidate_k, min_score=min_score)
            if mode in ["sparse", "hybrid"]:
                sparse_results = self.bm25_index.search(query, top_k=candidate_k)

        # If only dense or only sparse requested
        if mode == "dense":
            fused = dense_results
        elif mode == "sparse":
            fused = sparse_results
        else:
            # Hybrid Reciprocal Rank Fusion (RRF)
            fused = self._reciprocal_rank_fusion(dense_results, sparse_results)

        if not fused:
            return []

        # Re-rank candidates
        reranked = self.reranker.rerank(query, fused, top_k=top_k)
        
        # Attach rank index
        for rank, item in enumerate(reranked, start=1):
            item["final_rank"] = rank

        return reranked

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Fuse dense and sparse retrieval ranks using weighted RRF."""
        chunk_map: Dict[str, Dict[str, Any]] = {}
        rrf_scores: Dict[str, float] = {}

        # Process Dense results
        for rank, item in enumerate(dense_results, start=1):
            key = f"{item.get('source_file')}_{item.get('chunk_id')}_{item.get('text', '')[:40]}"
            chunk_map[key] = item
            score = self.dense_weight / (self.rrf_k + rank)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + score
            chunk_map[key]["retrieval_method"] = "dense"

        # Process Sparse results
        for rank, item in enumerate(sparse_results, start=1):
            key = f"{item.get('source_file')}_{item.get('chunk_id')}_{item.get('text', '')[:40]}"
            if key not in chunk_map:
                chunk_map[key] = item
                chunk_map[key]["retrieval_method"] = "sparse"
            else:
                chunk_map[key]["retrieval_method"] = "hybrid"

            score = self.sparse_weight / (self.rrf_k + rank)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + score

        # Sort combined results by RRF score
        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)
        
        fused_list = []
        for k in sorted_keys:
            item = chunk_map[k]
            item["rrf_score"] = round(rrf_scores[k], 6)
            fused_list.append(item)

        return fused_list
