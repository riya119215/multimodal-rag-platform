import os
import sys
import argparse
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Windows terminal UTF-8 compatibility
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.core.config import settings
from app.core.logging_config import logger
from app.api.routes import router as api_router
from app.generation.answer_generator import RAGAnswerGenerator
from app.ingestion.pipeline import IngestionPipeline

def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="Production-grade Multi-Modal RAG Platform with Hybrid Search, Re-ranking, and Grounding."
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix="/api/v1")
    return app

api_app = create_app()

def run_interactive_cli():
    """Interactive command-line interface for testing RAG queries."""
    print("=" * 65)
    print("  Enterprise Grounded RAG Platform - Interactive CLI")
    print("=" * 65)
    print(f"• LLM Model       : {settings.LLM_MODEL}")
    print(f"• Embedding Model : {settings.EMBEDDING_MODEL}")
    print(f"• Retrieval Mode  : {settings.RETRIEVAL_MODE}")
    print("=" * 65)

    generator = RAGAnswerGenerator()
    generator.retriever.vector_store.load_index()
    generator.retriever.bm25_index.load_index()

    print("\nRAG System ready! Enter your question (or type 'exit' / 'q' to quit):\n")
    while True:
        try:
            query = input("[Question] > ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("Exiting RAG system. Goodbye!")
                break

            print("\n[*] Retrieving context & synthesizing grounded answer...")
            res = generator.generate_answer(query, top_k=settings.TOP_K)

            print("\n" + "=" * 50)
            print(f"ANSWER (Confidence: {res['confidence']['percentage']} - {res['confidence']['level']}):")
            print("=" * 50)
            print(res["answer"])
            print("\n" + "=" * 50)
            print("SOURCES:")
            print("=" * 50)
            for i, s in enumerate(res["sources"], 1):
                if s.get("doc_type") == "audio_transcript" or "start_formatted" in s:
                    print(f"[{i}] Audio/Video #{s.get('video_number')}: \"{s.get('title')}\" [{s.get('start_formatted')} - {s.get('end_formatted')}]")
                else:
                    print(f"[{i}] Document: \"{s.get('source_file')}\" (Page {s.get('page_number', 'N/A')})")
            print("-" * 50 + "\n")

        except KeyboardInterrupt:
            print("\nExiting. Goodbye!")
            break
        except Exception as e:
            print(f"[!] Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enterprise RAG System CLI & Server Launcher")
    parser.add_argument("--server", action="store_true", help="Start FastAPI REST API server")
    parser.add_argument("--port", type=int, default=settings.PORT, help="Server port")
    parser.add_argument("--host", type=str, default=settings.HOST, help="Server host")
    parser.add_argument("--query", type=str, default="", help="Single question query")
    parser.add_argument("--reindex", action="store_true", help="Run full ingestion and reindex knowledge base")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive CLI")

    args = parser.parse_args()

    if args.reindex:
        pipeline = IngestionPipeline()
        total = pipeline.run()
        print(f"[✓] Knowledge base re-indexed successfully ({total} total chunks).")
    elif args.server:
        logger.info(f"Starting FastAPI server on http://{args.host}:{args.port}...")
        uvicorn.run(api_app, host=args.host, port=args.port)
    elif args.query:
        generator = RAGAnswerGenerator()
        generator.retriever.vector_store.load_index()
        generator.retriever.bm25_index.load_index()
        res = generator.generate_answer(args.query)
        print(f"\nQUESTION: {args.query}\n")
        print(f"ANSWER:\n{res['answer']}\n")
        print(f"CONFIDENCE: {res['confidence']['percentage']} ({res['confidence']['level']})")
    else:
        run_interactive_cli()
