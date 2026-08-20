from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder

from app.core.config import settings
from app.core.logging_config import logger

class ReRanker:
    """
    Two-stage re-ranker using Cross-Encoder models to re-evaluate and re-order
    the top retrieved candidate chunks for maximum precision.
    """
    def __init__(
        self,
        model_name: str = settings.RERANKER_MODEL,
        enabled: bool = settings.ENABLE_RERANKER
    ):
        self.model_name = model_name
        self.enabled = enabled
        self.model: Optional[CrossEncoder] = None
        self._load_failed = False

    def _ensure_model(self):
        if not self.enabled or self._load_failed or self.model is not None:
            return

        try:
            logger.info(f"Loading Cross-Encoder re-ranker: '{self.model_name}'...")
            self.model = CrossEncoder(self.model_name)
            logger.info("Re-ranker model loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load CrossEncoder model ({e}). Re-ranking will use graceful fallback.")
            self._load_failed = True

    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_k: int = settings.TOP_K) -> List[Dict[str, Any]]:
        """
        Re-rank candidates using cross-encoder query-document attention.
        """
        if not chunks or not query.strip():
            return []

        self._ensure_model()

        if not self.enabled or self.model is None:
            # Fallback: maintain current ordering but add default rerank_score
            for c in chunks:
                if "rerank_score" not in c:
                    c["rerank_score"] = c.get("dense_score", c.get("rrf_score", 0.5))
            return chunks[:top_k]

        try:
            pairs = [[query, c.get("text", "")] for c in chunks]
            scores = self.model.predict(pairs)

            # Sigmoid / Min-Max normalization for cross-encoder logits
            for c, raw_score in zip(chunks, scores):
                c["rerank_raw_score"] = float(raw_score)
                # Convert logit to approx 0-1 probability via sigmoid
                import math
                c["rerank_score"] = round(1.0 / (1.0 + math.exp(-float(raw_score))), 4)

            # Sort by rerank score descending
            reranked = sorted(chunks, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
            return reranked[:top_k]

        except Exception as e:
            logger.error(f"Error during re-ranking: {e}. Returning original candidates.")
            return chunks[:top_k]
