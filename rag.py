import os
import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Windows terminal UTF-8 compatibility
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import load_dotenv
load_dotenv()

from app.generation.answer_generator import RAGAnswerGenerator
from app.core.config import settings
from app.core.logging_config import logger

class RAGPipeline:
    """
    Unified RAG Pipeline adapter providing full backward compatibility
    with the upgraded Hybrid Search + Re-ranking + Grounding engine.
    """
    def __init__(
        self,
        api_key: str = None,
        model_name: str = settings.LLM_MODEL,
        index_path: str = None,
        metadata_path: str = None
    ):
        self.generator = RAGAnswerGenerator()
        if api_key:
            self.generator.llm.api_key = api_key
            self.generator.llm.client_ready = False
        if model_name:
            self.generator.llm.model_name = model_name

        # Load search indices
        self.generator.retriever.vector_store.load_index()
        self.generator.retriever.bm25_index.load_index()

    def query(self, user_query: str, top_k: int = 4, min_score: float = 0.0) -> dict:
        """Execute grounded RAG query."""
        res = self.generator.generate_answer(
            query=user_query,
            top_k=top_k,
            min_score=min_score,
            use_memory=False
        )
        return {
            "query": res["query"],
            "answer": res["answer"],
            "sources": res["sources"],
            "confidence": res["confidence"],
            "code_blocks": res["code_blocks"]
        }

def interactive_cli():
    print("=" * 65)
    print("      Enterprise Grounded RAG Platform (Hybrid FAISS + BM25 + Gemini)")
    print("=" * 65)

    try:
        pipeline = RAGPipeline()
    except Exception as e:
        print(f"[!] Initialization Error: {e}")
        return

    print("\nRAG System is ready! Ask any question about your documents and tutorials.")
    print("Type 'exit', 'quit', or 'q' to stop.\n")

    while True:
        try:
            query = input("[Your Question] > ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("Exiting RAG system. Goodbye!")
                break

            print("\n[*] Searching hybrid knowledge base & generating answer with Gemini...")
            result = pipeline.query(query, top_k=4)

            print("\n" + "=" * 50)
            print(f"ANSWER (Confidence: {result['confidence']['percentage']} - {result['confidence']['level']}):")
            print("=" * 50)
            print(result["answer"])
            print("\n" + "=" * 50)
            print("RETRIEVED SOURCES:")
            print("=" * 50)
            for i, src in enumerate(result["sources"], 1):
                if src.get("doc_type") == "audio_transcript" or "start_formatted" in src:
                    print(f"[{i}] Video #{src.get('video_number')}: \"{src.get('title')}\" [{src.get('start_formatted')} - {src.get('end_formatted')}] (Score: {src.get('score', src.get('dense_score', 'N/A'))})")
                    print(f"    ↳ Source File: {src.get('source_file')}")
                else:
                    print(f"[{i}] Document: \"{src.get('source_file')}\" (Page {src.get('page_number', 'N/A')})")
            print("-" * 50 + "\n")

        except KeyboardInterrupt:
            print("\nExiting RAG system. Goodbye!")
            break
        except Exception as e:
            print(f"[!] Query Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG pipeline with Gemini and Hybrid retrieval.")
    parser.add_argument("--query", type=str, default="", help="Single question to query")
    parser.add_argument("--top_k", type=int, default=4, help="Number of chunks to retrieve")
    parser.add_argument("--model", type=str, default=settings.LLM_MODEL, help="Gemini model name")
    args = parser.parse_args()

    if args.query:
        pipeline = RAGPipeline(model_name=args.model)
        res = pipeline.query(args.query, top_k=args.top_k)
        print("\n" + "=" * 50)
        print(f"QUESTION: {args.query}")
        print("=" * 50)
        print(res["answer"])
        print("\n" + "=" * 50)
        print(f"CONFIDENCE: {res['confidence']['percentage']} ({res['confidence']['level']})")
        print("=" * 50)
        print("SOURCES:")
        for s in res["sources"]:
            if "start_formatted" in s:
                print(f"- Video #{s.get('video_number')}: \"{s.get('title')}\" [{s.get('start_formatted')} - {s.get('end_formatted')}]")
            else:
                print(f"- Document: \"{s.get('source_file')}\"")
    else:
        interactive_cli()