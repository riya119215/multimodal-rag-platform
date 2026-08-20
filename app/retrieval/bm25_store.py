import re
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi

from app.ingestion.metadata import DocumentChunk
from app.core.config import settings
from app.core.logging_config import logger

def tokenize_for_bm25(text: str) -> List[str]:
    """Lowercase and extract alphanumeric word tokens."""
    return re.findall(r"\w+", text.lower())

class BM25Index:
    """
    Sparse keyword search index using Okapi BM25 for precise lexical term matching,
    exact code identifiers, and numbers.
    """
    def __init__(self, vector_store_dir: Optional[Path] = None):
        self.vector_store_dir = vector_store_dir or settings.VECTOR_STORE_DIR
        self.bm25_path = self.vector_store_dir / "bm25_index.pkl"
        
        self.bm25: Optional[BM25Okapi] = None
        self.corpus_tokens: List[List[str]] = []
        self.metadata: List[Dict[str, Any]] = []

    def load_index(self) -> bool:
        """Load BM25 index and chunk metadata from disk."""
        if not self.bm25_path.exists():
            return False

        logger.info(f"Loading BM25 index from {self.bm25_path}...")
        with open(self.bm25_path, "rb") as f:
            data = pickle.load(f)
            self.bm25 = data.get("bm25")
            self.corpus_tokens = data.get("corpus_tokens", [])
            self.metadata = data.get("metadata", [])

        logger.info(f"Loaded {len(self.metadata)} chunks from BM25 index.")
        return True

    def build_index(self, chunks: List[DocumentChunk]):
        """Tokenize chunks and build BM25 Okapi index."""
        if not chunks:
            logger.warning("No chunks provided to build BM25 index.")
            return

        self.metadata = [c.to_dict() for c in chunks]
        self.corpus_tokens = [tokenize_for_bm25(c.text) for c in chunks]

        logger.info(f"Building BM25 Okapi index over {len(self.corpus_tokens)} tokenized chunks...")
        self.bm25 = BM25Okapi(self.corpus_tokens)

        # Persist index
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        with open(self.bm25_path, "wb") as f:
            pickle.dump({
                "bm25": self.bm25,
                "corpus_tokens": self.corpus_tokens,
                "metadata": self.metadata
            }, f)

        logger.info(f"BM25 index written to {self.bm25_path}.")

    def search(
        self,
        query: str,
        top_k: int = settings.TOP_K,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Sparse BM25 retrieval with optional metadata filtering."""
        if not query.strip():
            return []

        if self.bm25 is None:
            if not self.load_index():
                logger.warning("BM25 Index is not built yet.")
                return []

        tokenized_query = tokenize_for_bm25(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        
        # Sort indices by score descending
        fetch_k = max(top_k * 4, 30)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:fetch_k]

        results = []
        rank = 1
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0.0:
                continue

            item = dict(self.metadata[idx])

            # Apply metadata filter if specified
            if filter_metadata:
                match = True
                for k, v in filter_metadata.items():
                    if not v:
                        continue
                    item_v = str(item.get(k, "")).lower()
                    target_v = str(v).lower()
                    if target_v not in item_v:
                        match = False
                        break
                if not match:
                    continue

            item["bm25_score"] = round(score, 4)
            item["bm25_rank"] = rank
            results.append(item)
            rank += 1
            if len(results) >= top_k:
                break

        return results

