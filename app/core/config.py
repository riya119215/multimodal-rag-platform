import os
from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from dotenv import load_dotenv

# Base project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

class Settings(BaseSettings):
    # Application Info
    APP_NAME: str = "Enterprise RAG System"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False
    
    # API Keys
    GEMINI_API_KEY: Optional[str] = None
    
    # Models
    LLM_MODEL: str = "gemini-3.6-flash"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ENABLE_RERANKER: bool = True
    
    # Ingestion & Chunking
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100
    SUPPORTED_EXTENSIONS: List[str] = [".pdf", ".docx", ".txt", ".md", ".csv", ".json", ".mp3", ".wav", ".m4a"]
    
    # Retrieval Configuration
    TOP_K: int = 5
    RETRIEVAL_MODE: str = "hybrid"  # dense, sparse, hybrid
    MIN_SIMILARITY_SCORE: float = 0.10
    RRF_K: int = 60  # Reciprocal Rank Fusion constant
    DENSE_WEIGHT: float = 0.6
    SPARSE_WEIGHT: float = 0.4
    
    # Whisper Configuration
    WHISPER_MODEL: str = "base"
    WHISPER_LANGUAGE: str = "hi"
    WHISPER_TASK: str = "translate"
    
    # Directory Paths
    DATA_DIR: Path = PROJECT_ROOT / "data"
    DOCUMENTS_DIR: Path = PROJECT_ROOT / "data" / "documents"
    AUDIO_DIR: Path = PROJECT_ROOT / "data" / "audios"
    PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
    VECTOR_STORE_DIR: Path = PROJECT_ROOT / "vectorstore"
    
    # Legacy Directory compatibility (for existing audio & json assets)
    LEGACY_AUDIO_DIR: Path = PROJECT_ROOT / "audios"
    LEGACY_JSON_DIR: Path = PROJECT_ROOT / "jsons"
    LEGACY_EMBEDDINGS_DIR: Path = PROJECT_ROOT / "embeddings"
    
    # Server configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="allow",
        case_sensitive=True
    )

    def ensure_directories(self):
        """Create necessary directories if they do not exist."""
        for path in [
            self.DATA_DIR,
            self.DOCUMENTS_DIR,
            self.AUDIO_DIR,
            self.PROCESSED_DIR,
            self.VECTOR_STORE_DIR
        ]:
            path.mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.ensure_directories()
