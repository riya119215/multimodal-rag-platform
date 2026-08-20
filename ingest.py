import os
import sys
import argparse
from chunk import process_audios
from embed import build_vector_index

# Windows terminal UTF-8 compatibility
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def run_ingestion(
    audio_dir: str = "audios",
    json_dir: str = "jsons",
    embeddings_dir: str = "embeddings",
    whisper_model: str = "base",
    embed_model: str = "all-MiniLM-L6-v2",
    lang: str = "hi",
    task: str = "translate",
    batch_size: int = 64
):
    """
    Unified end-to-end ingestion pipeline:
    1. Transcribe audio files in audio_dir into timestamped JSON chunks in json_dir.
    2. Encode all chunks using SentenceTransformer into FAISS index and metadata.pkl.
    """
    print("\n" + "=" * 65)
    print("      STARTING END-TO-END RAG AUDIO INGESTION PIPELINE")
    print("=" * 65)
    print(f"  • Audio Directory       : {audio_dir}")
    print(f"  • Transcripts JSON Path : {json_dir}")
    print(f"  • Embeddings/FAISS Path : {embeddings_dir}")
    print(f"  • Whisper Model         : {whisper_model} (Language: {lang}, Task: {task})")
    print(f"  • Embedding Model       : {embed_model}")
    print("=" * 65 + "\n")

    # Step 1: Transcribe & Chunk
    print("[STEP 1/2] Transcribing and Chunking Audios...")
    process_audios(
        audio_dir=audio_dir,
        json_dir=json_dir,
        model_name=whisper_model,
        language=lang,
        task=task
    )

    # Step 2: Build Vector Embeddings & Index
    print("\n[STEP 2/2] Generating Vector Embeddings & Building FAISS Index...")
    build_vector_index(
        json_folder=json_dir,
        output_folder=embeddings_dir,
        model_name=embed_model,
        batch_size=batch_size
    )

    print("\n" + "=" * 65)
    print("  ✓ INGESTION PIPELINE COMPLETED SUCCESSFULLY!")
    print("  You can now start querying with 'python rag.py' or launch 'streamlit run app.py'")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end audio ingestion pipeline for RAG.")
    parser.add_argument("--audio_dir", type=str, default="audios", help="Directory containing audio files")
    parser.add_argument("--json_dir", type=str, default="jsons", help="Directory to save JSON transcripts")
    parser.add_argument("--embeddings_dir", type=str, default="embeddings", help="Directory to save FAISS index")
    parser.add_argument("--whisper_model", type=str, default="base", help="Whisper model size")
    parser.add_argument("--embed_model", type=str, default="all-MiniLM-L6-v2", help="Embedding model name")
    parser.add_argument("--lang", type=str, default="hi", help="Audio source language")
    parser.add_argument("--task", type=str, default="translate", help="Whisper task: 'translate' or 'transcribe'")
    parser.add_argument("--batch_size", type=int, default=64, help="Embedding batch size")
    
    args = parser.parse_args()
    run_ingestion(
        audio_dir=args.audio_dir,
        json_dir=args.json_dir,
        embeddings_dir=args.embeddings_dir,
        whisper_model=args.whisper_model,
        embed_model=args.embed_model,
        lang=args.lang,
        task=args.task,
        batch_size=args.batch_size
    )
