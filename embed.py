import os
import sys
import json
import pickle
import argparse
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Ensure Windows terminal compatibility with UTF-8
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def format_timestamp(seconds: float) -> str:
    """Convert seconds into MM:SS or HH:MM:SS format."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"

def parse_filename_info(filename: str):
    """Extract default video number and title from filename if missing."""
    base = os.path.splitext(filename)[0]
    if "_" in base:
        parts = base.split("_", 1)
        return parts[0].strip(), parts[1].strip()
    return "N/A", base.strip()

def build_vector_index(
    json_folder: str = "jsons",
    output_folder: str = "embeddings",
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64
):
    """
    Load all transcript chunks, generate normalized dense embeddings,
    and save a FAISS vector index alongside full metadata (timestamps, titles, transcripts).
    """
    os.makedirs(output_folder, exist_ok=True)

    print(f"[*] Loading embedding model: '{model_name}'...")
    model = SentenceTransformer(model_name)
    print("[+] Model loaded successfully.")

    all_texts = []
    metadata = []

    json_files = [f for f in os.listdir(json_folder) if f.endswith(".json")]
    if not json_files:
        print(f"[!] No JSON files found in '{json_folder}'. Run chunk.py first.")
        return

    print(f"[*] Parsing transcript files from '{json_folder}' ({len(json_files)} files)...")

    for file in sorted(json_files):
        json_path = os.path.join(json_folder, file)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        default_num, default_title = parse_filename_info(file)
        source_file = data.get("file_name", file)
        chunks = data.get("chunks", [])

        for idx, chunk in enumerate(chunks):
            text = chunk.get("text", "").strip()
            if not text:
                continue

            start_sec = float(chunk.get("start", 0.0))
            end_sec = float(chunk.get("end", 0.0))
            num = str(chunk.get("number", data.get("video_id", default_num)))
            title = chunk.get("title", data.get("title", default_title))

            all_texts.append(text)
            metadata.append({
                "chunk_id": chunk.get("chunk_id", idx),
                "source_file": source_file,
                "video_number": num,
                "title": title,
                "start": start_sec,
                "end": end_sec,
                "start_formatted": format_timestamp(start_sec),
                "end_formatted": format_timestamp(end_sec),
                "text": text
            })

    print(f"[+] Total transcript chunks collected: {len(all_texts)}")
    print(f"[*] Encoding vectors in batches (batch_size={batch_size})...")

    # Generate L2-normalized embeddings for cosine similarity
    embeddings = model.encode(
        all_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    dimension = embeddings.shape[1]
    print(f"[+] Embeddings generated. Shape: {embeddings.shape} (Dimension: {dimension})")

    # Use Inner Product (IndexFlatIP) on normalized vectors for exact Cosine Similarity
    print("[*] Creating FAISS Index (Cosine Similarity)...")
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    index_path = os.path.join(output_folder, "faiss_index.bin")
    metadata_path = os.path.join(output_folder, "metadata.pkl")

    faiss.write_index(index, index_path)
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    print("\n" + "=" * 50)
    print("[SUCCESS] Vector Indexing Completed Successfully!")
    print(f"  - Total Chunks Indexed : {len(metadata)}")
    print(f"  - Vector Dimensions    : {dimension}")
    print(f"  - Index File           : {index_path}")
    print(f"  - Metadata File        : {metadata_path}")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create FAISS embeddings index for transcripts.")
    parser.add_argument("--json_folder", type=str, default="jsons", help="Directory containing JSON transcript chunks")
    parser.add_argument("--output_folder", type=str, default="embeddings", help="Directory to save index and metadata")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument("--batch_size", type=int, default=64, help="Embedding batch size")
    
    args = parser.parse_args()
    build_vector_index(
        json_folder=args.json_folder,
        output_folder=args.output_folder,
        model_name=args.model,
        batch_size=args.batch_size
    )
