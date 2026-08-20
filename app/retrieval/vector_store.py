import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.ingestion.metadata import DocumentChunk
from app.core.config import settings
from app.core.logging_config import logger

class VectorStore:
    """
    Dense vector search engine utilizing FAISS (IndexFlatIP) with L2-normalized embeddings
    to achieve exact cosine similarity search.
    """
    def __init__(
        self,
        model_name: str = settings.EMBEDDING_MODEL,
        vector_store_dir: Optional[Path] = None
    ):
        self.model_name = model_name
        self.vector_store_dir = vector_store_dir or settings.VECTOR_STORE_DIR
        self.index_path = self.vector_store_dir / "faiss_index.bin"
        self.metadata_path = self.vector_store_dir / "metadata.pkl"
        
        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: List[Dict[str, Any]] = []

    def _ensure_model(self):
        """Lazy load embedding model."""
        if self.model is None:
            logger.info(f"Loading SentenceTransformer embedding model: '{self.model_name}'...")
            self.model = SentenceTransformer(self.model_name)

    def load_index(self) -> bool:
        """Load FAISS index and metadata from disk, checking fallback legacy paths if needed."""
        # Check primary path
        if self.index_path.exists() and self.metadata_path.exists():
            target_idx = self.index_path
            target_meta = self.metadata_path
        # Check legacy embeddings directory
        elif (settings.LEGACY_EMBEDDINGS_DIR / "faiss_index.bin").exists():
            target_idx = settings.LEGACY_EMBEDDINGS_DIR / "faiss_index.bin"
            target_meta = settings.LEGACY_EMBEDDINGS_DIR / "metadata.pkl"
        else:
            return False

        logger.info(f"Loading FAISS index from {target_idx}...")
        self.index = faiss.read_index(str(target_idx))

        with open(target_meta, "rb") as f:
            self.metadata = pickle.load(f)

        logger.info(f"Loaded {len(self.metadata)} chunks from FAISS vector store.")
        return True

    def build_index(self, chunks: List[DocumentChunk], batch_size: int = 64):
        """Generate normalized embeddings for chunks and persist FAISS index."""
        self._ensure_model()
        if not chunks:
            logger.warning("No chunks provided to build vector index.")
            return

        texts = [c.text for c in chunks]
        self.metadata = [c.to_dict() for c in chunks]

        logger.info(f"Generating dense embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype("float32")

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        # Ensure directory exists and persist
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

        logger.info(f"FAISS index written to {self.index_path} ({len(chunks)} vectors, dim={dimension}).")

    def search(
        self,
        query: str,
        top_k: int = settings.TOP_K,
        min_score: float = 0.0,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Dense cosine similarity retrieval with optional metadata filtering."""
        if not query.strip():
            return []

        if self.index is None:
            if not self.load_index():
                logger.warning("FAISS Index is not built yet.")
                return []

        self._ensure_model()

        query_vector = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype("float32")

        # Fetch extra candidates to account for metadata filtering
        fetch_k = min(max(top_k * 4, 20), self.index.ntotal)
        if fetch_k <= 0:
            return []

        scores, indices = self.index.search(query_vector, fetch_k)

        results = []
        rank = 1
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue
            sim_score = float(score)
            if sim_score < min_score:
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

            item["dense_score"] = round(sim_score, 4)
            item["dense_rank"] = rank
            results.append(item)
            rank += 1
            if len(results) >= top_k:
                break

        return results

