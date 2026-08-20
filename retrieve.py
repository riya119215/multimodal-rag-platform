import os
import sys
import pickle
import argparse
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Windows UTF-8 compatibility
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

class TranscriptRetriever:
    """
    Handles semantic vector search over audio/video transcript chunks using FAISS.
    """
    def __init__(
        self,
        index_path: str = "embeddings/faiss_index.bin",
        metadata_path: str = "embeddings/metadata.pkl",
        model_name: str = "all-MiniLM-L6-v2"
    ):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.model_name = model_name
        self.index = None
        self.metadata = None
        self.model = None
        self._load_resources()

    def _load_resources(self):
        if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
            raise FileNotFoundError(
                f"Embedding files not found. Please run 'python embed.py' first.\n"
                f"Expected: {self.index_path} and {self.metadata_path}"
            )

        print(f"[*] Loading SentenceTransformer model: '{self.model_name}'...")
        self.model = SentenceTransformer(self.model_name)

        print(f"[*] Loading FAISS index from '{self.index_path}'...")
        self.index = faiss.read_index(self.index_path)

        print(f"[*] Loading metadata from '{self.metadata_path}'...")
        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

        print(f"[+] Successfully loaded {len(self.metadata)} chunks from index.")

    def retrieve(self, query: str, top_k: int = 4, min_score: float = 0.0) -> list:
        """
        Search for top_k most semantically relevant transcript chunks.
        Returns a list of matching chunks with metadata and cosine similarity scores.
        """
        if not query.strip():
            return []

        # Encode query with unit normalization for Cosine Similarity
        query_vector = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype("float32")

        # Perform FAISS search
        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx == -1:
                continue

            sim_score = float(score)
            if sim_score < min_score:
                continue

            chunk_meta = self.metadata[idx]
            start_fmt = chunk_meta.get("start_formatted", "00:00")
            end_fmt = chunk_meta.get("end_formatted", "00:00")
            title = chunk_meta.get("title", "Untitled")
            num = chunk_meta.get("video_number", "N/A")

            results.append({
                "rank": rank,
                "score": round(sim_score, 4),
                "video_number": num,
                "title": title,
                "start": chunk_meta.get("start", 0.0),
                "end": chunk_meta.get("end", 0.0),
                "start_formatted": start_fmt,
                "end_formatted": end_fmt,
                "timestamp_label": f"[{start_fmt} - {end_fmt}]",
                "source_file": chunk_meta.get("source_file", ""),
                "chunk_id": chunk_meta.get("chunk_id", idx),
                "text": chunk_meta.get("text", "")
            })

        return results

    def format_context_for_llm(self, results: list) -> str:
        """Format retrieved chunks into a clean context block for LLM prompting."""
        if not results:
            return "No relevant video transcript excerpts found."

        context_lines = []
        for i, item in enumerate(results, 1):
            context_lines.append(
                f"[Source {i}] Video #{item['video_number']}: \"{item['title']}\" | Timestamp: {item['timestamp_label']}\n"
                f"Transcript: \"{item['text']}\"\n"
            )
        return "\n".join(context_lines)

def interactive_cli():
    print("=" * 60)
    print("    Audio/Video Transcript Semantic Search (FAISS)")
    print("=" * 60)

    try:
        retriever = TranscriptRetriever()
    except Exception as e:
        print(f"[!] Error: {e}")
        return

    print("\nSearch is ready! Enter your query below (or type 'exit' to quit):")

    while True:
        try:
            query = input("\n[Search Query] > ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("Exiting search. Goodbye!")
                break

            results = retriever.retrieve(query, top_k=3)

            if not results:
                print("[-] No matching transcript chunks found.")
                continue

            print(f"\n--- Top {len(results)} Matches for: \"{query}\" ---")
            for item in results:
                print(f"\n* Rank #{item['rank']} | Score: {item['score']} (Cosine Similarity)")
                print(f"  Video: Video #{item['video_number']} - {item['title']}")
                print(f"  Time : {item['timestamp_label']} ({item['source_file']})")
                print(f"  Text : \"{item['text']}\"")
                print("-" * 50)

        except KeyboardInterrupt:
            print("\nExiting search. Goodbye!")
            break
        except Exception as e:
            print(f"[!] Search Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query transcript embeddings via FAISS.")
    parser.add_argument("--query", type=str, default="", help="Direct search query string")
    parser.add_argument("--top_k", type=int, default=3, help="Number of chunks to retrieve")
    args = parser.parse_args()

    if args.query:
        retriever = TranscriptRetriever()
        res = retriever.retrieve(args.query, top_k=args.top_k)
        print(f"\nResults for '{args.query}':\n")
        for r in res:
            print(f"[{r['score']}] Video #{r['video_number']} - {r['title']} {r['timestamp_label']}")
            print(f"       {r['text']}\n")
    else:
        interactive_cli()