import re
from typing import List, Dict, Any, Tuple
from app.core.logging_config import logger

class GroundingChecker:
    """
    Evaluates context sufficiency, verifies answer grounding,
    and computes a composite, explainable confidence score.
    """
    def __init__(self, min_relevance_threshold: float = 0.15):
        self.min_relevance_threshold = min_relevance_threshold

    def calculate_confidence(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        answer: str = ""
    ) -> Dict[str, Any]:
        """
        Calculates an explainable confidence score based on:
        1. Dense Vector Similarity (weight: 0.40)
        2. Sparse BM25 / RRF Score (weight: 0.20)
        3. Cross-Encoder Re-ranker Score (weight: 0.25)
        4. Query-Context Lexical Overlap (weight: 0.15)
        """
        if not retrieved_chunks:
            return {
                "score": 0.0,
                "percentage": "0%",
                "level": "Low",
                "is_grounded": False,
                "breakdown": {
                    "dense_similarity": 0.0,
                    "rerank_score": 0.0,
                    "keyword_overlap": 0.0,
                    "num_sources": 0
                }
            }

        top_chunk = retrieved_chunks[0]

        # 1. Dense score (typically 0.0 - 1.0)
        dense_sim = float(top_chunk.get("dense_score", 0.5))

        # 2. Rerank score (typically 0.0 - 1.0)
        rerank_score = float(top_chunk.get("rerank_score", dense_sim))

        # 3. Keyword / lexical overlap between query and all retrieved chunks
        query_words = set(re.findall(r"\w+", query.lower()))
        context_words = set(re.findall(r"\w+", " ".join([c.get("text", "") for c in retrieved_chunks]).lower()))
        
        overlap_count = len(query_words.intersection(context_words))
        overlap_ratio = min(1.0, overlap_count / max(1, len(query_words)))

        # 4. Composite weighted score
        composite = (
            0.40 * dense_sim +
            0.25 * rerank_score +
            0.20 * min(1.0, len(retrieved_chunks) / 4.0) +
            0.15 * overlap_ratio
        )
        composite = max(0.0, min(1.0, composite))
        pct = round(composite * 100, 1)

        if composite >= 0.70:
            level = "High"
        elif composite >= 0.45:
            level = "Medium"
        else:
            level = "Low"

        is_grounded = composite >= self.min_relevance_threshold and overlap_count >= 1

        return {
            "score": round(composite, 4),
            "percentage": f"{pct}%",
            "level": level,
            "is_grounded": is_grounded,
            "breakdown": {
                "dense_similarity": round(dense_sim, 4),
                "rerank_score": round(rerank_score, 4),
                "keyword_overlap": round(overlap_ratio, 4),
                "num_sources": len(retrieved_chunks)
            }
        }

    def check_context_sufficiency(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> Tuple[bool, str]:
        """
        Check if retrieved context has enough substance to justify generating an answer.
        Returns (is_sufficient, fallback_message).
        """
        if not retrieved_chunks:
            return False, "I could not find sufficient information in the provided documents/tutorials to answer this question."

        conf = self.calculate_confidence(query, retrieved_chunks)
        if not conf["is_grounded"]:
            logger.info(f"Query '{query}' rejected by grounding check (confidence: {conf['percentage']}).")
            return False, "I could not find sufficient information in the provided documents/tutorials to answer this question."

        return True, ""
