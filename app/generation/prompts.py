from typing import List, Dict, Any

SYSTEM_RAG_PROMPT = """You are an expert AI Knowledge Assistant and Tutor specialized in document analysis and technical tutorials.

Your task is to provide clear, accurate, and structured answers based EXCLUSIVELY on the retrieved context excerpts provided below.

STRICT GUIDELINES:
1. Grounding: Answer strictly using the provided context. If the context does not contain enough information to answer the question, clearly state:
   "I could not find sufficient information in the provided documents/tutorials to answer this question."
   Do NOT make up facts or extrapolate beyond what is stated in the context.
2. Source Citations: Whenever you state facts, code examples, or techniques, cite your sources clearly:
   - For audio/video transcripts: `[Source X: "<title>" @ <start_time> - <end_time>]`
   - For PDF/documents: `[Source X: "<source_file>" Page <page_number>]`
3. Code Quality: If the context contains Python, Pandas, or Matplotlib code, output clean, executable, complete Python code blocks (` ```python ... ``` `).
4. Tone & Structure: Professional, pedagogical, well-formatted with markdown headers and bullet points.
"""

SYSTEM_QUERY_REWRITE_PROMPT = """Given the following conversation history and a follow-up question, rewrite the follow-up question into a standalone, self-contained search query suitable for semantic vector retrieval.

Rules:
1. Resolve all pronouns (e.g. "it", "they", "its", "this", "that") based on the chat history.
2. Keep the query concise and focused on the core technical keywords.
3. If the question is already self-contained, return it unchanged.
4. Output ONLY the rewritten search query without any explanation.
"""

def format_context_for_llm(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved candidate chunks into clean, structured context for the LLM."""
    if not chunks:
        return "No relevant context excerpts were found."

    context_blocks = []
    for i, c in enumerate(chunks, start=1):
        doc_type = c.get("doc_type", "document")
        src = c.get("source_file", "unknown")
        
        if doc_type == "audio_transcript" or "start_formatted" in c:
            v_num = c.get("video_number", "N/A")
            title = c.get("title", src)
            time_label = f"[{c.get('start_formatted', '00:00')} - {c.get('end_formatted', '00:00')}]"
            header = f"[Source {i} - Audio/Video #{v_num}: \"{title}\" | Timestamp: {time_label}]"
        elif c.get("page_number") is not None:
            header = f"[Source {i} - Document: \"{src}\" | Page: {c.get('page_number')}]"
        else:
            header = f"[Source {i} - Document: \"{src}\"]"

        # Prefer parent_context (Small-to-Big) for richer LLM context, fallback to text
        content = (c.get("parent_context") or c.get("text", "")).strip()
        context_blocks.append(f"{header}\n{content}\n")

    return "\n---\n".join(context_blocks)

