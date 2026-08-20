import os
import sys
import glob
import json
from pathlib import Path
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Windows UTF-8 console compatibility
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.core.config import settings
from app.generation.answer_generator import RAGAnswerGenerator
from app.ingestion.pipeline import IngestionPipeline
from app.utils.helpers import extract_python_code, execute_matplotlib_code

# Page Configuration
st.set_page_config(
    page_title="Enterprise RAG System | Multimodal & Grounded",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling with Dark/Light Theme Support
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #2196F3;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        opacity: 0.85;
        margin-bottom: 1.2rem;
        line-height: 1.5;
    }
    .source-card {
        background: rgba(30, 136, 229, 0.08);
        border-left: 4px solid #1E88E5;
        padding: 12px 16px;
        margin: 10px 0px;
        border-radius: 6px;
    }
    .source-card-audio {
        background: rgba(46, 125, 50, 0.08);
        border-left: 4px solid #2E7D32;
        padding: 12px 16px;
        margin: 10px 0px;
        border-radius: 6px;
    }
    .score-badge {
        background-color: rgba(33, 150, 243, 0.15);
        color: #2196F3;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .conf-high {
        background-color: rgba(76, 175, 80, 0.2);
        color: #4CAF50;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 700;
    }
    .conf-med {
        background-color: rgba(255, 152, 0, 0.2);
        color: #FF9800;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 700;
    }
    .conf-low {
        background-color: rgba(244, 67, 54, 0.2);
        color: #F44336;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Cache RAG Generator
@st.cache_resource(show_spinner="Loading Hybrid Retriever, FAISS Index, and Cross-Encoder...")
def get_rag_generator():
    gen = RAGAnswerGenerator()
    gen.retriever.vector_store.load_index()
    gen.retriever.bm25_index.load_index()
    return gen

generator = get_rag_generator()
pipeline = IngestionPipeline()

# Initialize Chat Session State
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# --- Sidebar ---
with st.sidebar:
    st.markdown("### ⚙️ Control & Configuration")
    
    api_key_input = st.text_input(
        "Gemini API Key",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        help="Loads from .env by default. Enter here to override for this session."
    )
    if api_key_input:
        os.environ["GEMINI_API_KEY"] = api_key_input
        generator.llm.api_key = api_key_input
        generator.llm.client_ready = False

    model_choice = st.selectbox(
        "LLM Model",
        options=["gemini-3.6-flash", "gemini-flash-latest", "gemini-2.5-flash", "gemini-1.5-flash"],
        index=0
    )
    generator.llm.model_name = model_choice

    retrieval_mode = st.selectbox(
        "Retrieval Mode",
        options=["hybrid", "dense", "sparse"],
        index=0,
        help="Hybrid combines Dense FAISS + Sparse BM25 via Reciprocal Rank Fusion (RRF)."
    )

    enable_rerank = st.checkbox("Enable Cross-Encoder Re-ranking", value=True)
    generator.retriever.reranker.enabled = enable_rerank

    top_k_val = st.slider("Top Chunks (top_k)", min_value=1, max_value=10, value=5)
    min_score_val = st.slider("Min Similarity Score", min_value=0.0, max_value=1.0, value=0.10, step=0.05)

    allow_fallback = st.checkbox("🌐 Allow General AI Fallback", value=False, help="If context is missing in local documents, generate general answer with disclaimer.")
    
    st.markdown("---")
    st.markdown("### 🎯 Metadata Filtering (Self-Query)")
    filter_doc_type = st.selectbox("Filter by Format", options=["All", "pdf", "docx", "csv", "audio_transcript", "text"], index=0)
    filter_video_num = st.text_input("Filter by Video # (e.g. 2)", value="", help="Leave blank to search all videos")
    
    active_filter_dict = {}
    if filter_doc_type != "All":
        active_filter_dict["doc_type"] = filter_doc_type
    if filter_video_num.strip():
        active_filter_dict["video_number"] = filter_video_num.strip()

    st.markdown("---")
    st.markdown("### 📊 Knowledge Base Stats")
    
    total_vectors = len(generator.retriever.vector_store.metadata) if generator.retriever.vector_store.metadata else 0
    raw_docs = len(list(settings.DOCUMENTS_DIR.glob("*.*"))) if settings.DOCUMENTS_DIR.exists() else 0
    audio_files = len(list(settings.AUDIO_DIR.glob("*.*"))) + len(list(settings.LEGACY_AUDIO_DIR.glob("*.*"))) if settings.LEGACY_AUDIO_DIR.exists() else 0
    json_transcripts = len(list(settings.PROCESSED_DIR.glob("*.json"))) + len(list(settings.LEGACY_JSON_DIR.glob("*.json"))) if settings.LEGACY_JSON_DIR.exists() else 0

    col_s1, col_s2 = st.columns(2)
    col_s1.metric("Indexed Chunks", total_vectors)
    col_s2.metric("Transcripts", json_transcripts)
    
    st.write(f"📄 **Text Docs:** {raw_docs} | 🎙️ **Audio Files:** {audio_files}")

    if st.button("🔄 Rebuild Search Index", use_container_width=True):
        with st.spinner("Re-indexing all documents and transcripts..."):
            count = pipeline.run()
            generator.retriever.vector_store.load_index()
            generator.retriever.bm25_index.load_index()
            st.success(f"✓ Re-indexed {count} chunks successfully!")
            st.rerun()

    st.markdown("---")
    # Export Chat Report
    if st.session_state.chat_messages:
        report_lines = ["# RAG System — Conversation & Citation Report\n"]
        for m in st.session_state.chat_messages:
            role = m['role'].upper()
            report_lines.append(f"### {role}\n{m['content']}\n")
            if m.get('confidence'):
                c = m['confidence']
                report_lines.append(f"*Confidence: {c.get('level')} ({c.get('percentage')})*\n")
        report_md = "\n".join(report_lines)
        st.download_button(
            "📥 Download Chat Report (.md)",
            data=report_md,
            file_name="rag_chat_report.md",
            mime="text/markdown",
            use_container_width=True
        )

# --- Main UI ---
st.markdown('<div class="main-header">🧠 Enterprise Multi-Modal RAG Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Grounded Question Answering across Documents (PDF, DOCX, TXT, CSV) and Audio/Video Tutorials with Hybrid Search, Re-Ranking, and Live Code Execution.</div>', unsafe_allow_html=True)

# Tabs
tab_chat, tab_upload, tab_kb, tab_arch = st.tabs([
    "💬 Ask Assistant (RAG)",
    "📤 Document Upload & Ingestion",
    "🎧 Audio Library & Transcripts",
    "ℹ️ System Architecture"
])

# ================= TAB 1: Chat / RAG =================
with tab_chat:
    col_q1, col_q2, col_q3, col_q4 = st.columns(4)
    preset_query = None
    if col_q1.button("📊 How to make Bar Charts?"):
        preset_query = "How do I create and customize a bar chart in Matplotlib?"
    if col_q2.button("📈 How to create Subplots?"):
        preset_query = "How do I create subplots in Matplotlib with multiple figures?"
    if col_q3.button("🎨 Plot Titles & Labels?"):
        preset_query = "How do I add titles, labels, and customize colors in Matplotlib?"
    if col_q4.button("🧹 Clear Chat History"):
        st.session_state.chat_messages = []
        generator.memory.clear()
        st.rerun()

    # Voice Query Option
    voice_query = None
    with st.expander("🎙️ Voice Query Input (Record from Microphone)"):
        audio_prompt = st.audio_input("Record your question:")
        if audio_prompt is not None:
            with st.spinner("Transcribing your audio with Whisper..."):
                try:
                    import whisper
                    temp_audio_path = settings.DATA_DIR / "temp_voice_query.wav"
                    with open(temp_audio_path, "wb") as f:
                        f.write(audio_prompt.getbuffer())
                    w_model = whisper.load_model("base")
                    w_res = w_model.transcribe(str(temp_audio_path))
                    voice_query = w_res.get("text", "").strip()
                    if voice_query:
                        st.success(f"🗣️ Transcribed: \"{voice_query}\"")
                except Exception as e:
                    st.warning(f"Voice transcription note: {e}")

    # Display conversation history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("confidence"):
                conf = msg["confidence"]
                badge_class = "conf-high" if conf["level"] == "High" else ("conf-med" if conf["level"] == "Medium" else "conf-low")
                st.markdown(f"**Confidence:** <span class='{badge_class}'>{conf['level']} ({conf['percentage']})</span>", unsafe_allow_html=True)

    # Chat Input
    query_input = st.chat_input("Ask any question about your documents, code, or video tutorials...")
    active_query = preset_query or voice_query or query_input

    if active_query:
        # Display user message
        st.session_state.chat_messages.append({"role": "user", "content": active_query})
        with st.chat_message("user"):
            st.markdown(active_query)

        # Assistant generation with streaming
        with st.chat_message("assistant"):
            with st.spinner("🔍 Hybrid Retrieval (FAISS + BM25) → Re-ranking → Synthesis..."):
                token_stream, meta = generator.generate_answer_stream(
                    query=active_query,
                    top_k=top_k_val,
                    retrieval_mode=retrieval_mode,
                    min_score=min_score_val,
                    use_memory=True,
                    filter_metadata=active_filter_dict or None,
                    allow_agentic_fallback=allow_fallback
                )

            # Stream the generated response token-by-token
            answer_text = st.write_stream(token_stream)

            # Confidence Badge
            conf = meta.get("confidence", {})
            badge_class = "conf-high" if conf.get("level") == "High" else ("conf-med" if conf.get("level") == "Medium" else "conf-low")
            st.markdown(f"**Retrieval Confidence:** <span class='{badge_class}'>{conf.get('level', 'N/A')} ({conf.get('percentage', 'N/A')})</span> &nbsp;|&nbsp; <b>Grounded:</b> {'✅ Yes' if conf.get('is_grounded') else '⚠️ Context Insufficient'}", unsafe_allow_html=True)
            
            with st.expander("🔍 View Confidence Signal Breakdown"):
                bd = conf.get("breakdown", {})
                b_col1, b_col2, b_col3, b_col4 = st.columns(4)
                b_col1.metric("Dense Similarity", bd.get("dense_similarity", 0.0))
                b_col2.metric("Re-rank Score", bd.get("rerank_score", 0.0))
                b_col3.metric("Keyword Overlap", bd.get("keyword_overlap", 0.0))
                b_col4.metric("Sources Used", bd.get("num_sources", 0))

            # Code Execution if Python Code exists
            code_blocks = extract_python_code(answer_text)
            if code_blocks:
                st.markdown("---")
                st.markdown("### 📊 Live Code & Matplotlib Execution")
                with st.expander("▶️ Run & Preview Python Visualization", expanded=True):
                    code_to_run = code_blocks[0]
                    st.code(code_to_run, language="python")
                    if st.button("⚡ Execute Python Code & Render Chart"):
                        fig, stdout = execute_matplotlib_code(code_to_run)
                        if fig and fig.get_axes():
                            st.pyplot(fig)
                            st.success("✓ Chart rendered successfully!")
                        if stdout:
                            st.text(f"Console output:\n{stdout}")

            # Retrieved Sources
            sources = meta.get("sources", [])
            st.markdown("---")
            st.markdown(f"### 📚 Retrieved Context Sources ({len(sources)} items)")
            
            if not sources:
                if meta.get("is_agentic_fallback"):
                    st.info("ℹ️ Answer generated using general AI knowledge (not present in indexed local files).")
                else:
                    st.info("No sources exceeded the similarity threshold.")
            else:
                for idx, src in enumerate(sources, start=1):
                    doc_type = src.get("doc_type", "document")
                    score_info = f"RRF: {src.get('rrf_score', 'N/A')} | Dense: {src.get('dense_score', 'N/A')} | Rerank: {src.get('rerank_score', 'N/A')}"
                    
                    if doc_type == "audio_transcript" or "start_formatted" in src:
                        v_num = src.get("video_number", "N/A")
                        title = src.get("title", src.get("source_file", "Untitled"))
                        st.markdown(f"""
                        <div class="source-card-audio">
                            <b>Source #{idx} [Audio/Video #{v_num}]:</b> {title}<br>
                            ⏱️ <b>Timestamp:</b> <code>[{src.get('start_formatted')} - {src.get('end_formatted')}]</code> &nbsp;|&nbsp;
                            <span class="score-badge">{score_info}</span><br>
                            <em>"{src.get('text')}"</em>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Audio sync player
                        src_audio = src.get("source_file", "")
                        possible_paths = [
                            settings.AUDIO_DIR / src_audio,
                            settings.LEGACY_AUDIO_DIR / src_audio
                        ]
                        for p in possible_paths:
                            if p.exists():
                                st.audio(str(p), start_time=int(src.get("start", 0)))
                                break
                    else:
                        page_str = f"Page {src.get('page_number')}" if src.get("page_number") else "Document"
                        st.markdown(f"""
                        <div class="source-card">
                            <b>Source #{idx} [Document]:</b> <code>{src.get('source_file')}</code> ({page_str})<br>
                            <span class="score-badge">{score_info}</span><br>
                            <em>"{src.get('text')}"</em>
                        </div>
                        """, unsafe_allow_html=True)

            # Store in session state
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": answer_text,
                "confidence": conf
            })


# ================= TAB 2: Ingestion & Upload =================
with tab_upload:
    st.markdown("### 📤 Ingest Documents into Knowledge Repository")
    st.write("Upload PDF, Word (DOCX), Text (TXT/MD), CSV, or Audio Transcript files:")

    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=["pdf", "docx", "txt", "md", "csv", "json"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("🚀 Upload & Index Files", type="primary"):
            settings.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
            settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            
            for uf in uploaded_files:
                ext = Path(uf.name).suffix.lower()
                target_dir = settings.PROCESSED_DIR if ext == ".json" else settings.DOCUMENTS_DIR
                target_path = target_dir / uf.name
                with open(target_path, "wb") as f:
                    f.write(uf.getbuffer())
                st.write(f"✓ Saved `{uf.name}`")

            with st.spinner("Re-indexing knowledge base..."):
                count = pipeline.run()
                generator.retriever.vector_store.load_index()
                generator.retriever.bm25_index.load_index()
                st.success(f"🎉 Ingestion complete! {count} total chunks are indexed.")
                st.rerun()

    st.markdown("---")
    st.markdown("### 📂 Stored Document Files")
    docs = list(settings.DOCUMENTS_DIR.glob("*.*")) if settings.DOCUMENTS_DIR.exists() else []
    if docs:
        doc_data = [{"Filename": d.name, "Type": d.suffix, "Size (KB)": round(d.stat().st_size / 1024, 2)} for d in docs]
        st.dataframe(pd.DataFrame(doc_data), use_container_width=True)
    else:
        st.info("No raw document files uploaded yet in `data/documents`.")

# ================= TAB 3: Audio Library =================
with tab_kb:
    st.markdown("### 🎧 Indexed Audio/Video Tutorials")
    json_paths = sorted(glob.glob("data/processed/*.json") + glob.glob("jsons/*.json"))
    seen_jsons = set()
    unique_jsons = []
    for jp in json_paths:
        name = os.path.basename(jp)
        if name not in seen_jsons:
            seen_jsons.add(name)
            unique_jsons.append(jp)

    if not unique_jsons:
        st.warning("No transcript JSON files found.")
    else:
        for jp in unique_jsons:
            with open(jp, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            v_id = data.get("video_id", "N/A")
            title = data.get("title", os.path.basename(jp))
            total_c = data.get("total_chunks", len(data.get("chunks", [])))
            
            with st.expander(f"🎬 Video #{v_id}: {title} ({total_c} chunks)"):
                st.write(f"**Source File:** `{data.get('file_name', '')}`")
                st.write(f"**Full Transcript Snippet:**")
                st.write(data.get("full_text", "")[:350] + "...")
                
                st.markdown("**Sample Segment Chunks:**")
                for c in data.get("chunks", [])[:4]:
                    st.write(f"- `[{c.get('start_formatted')} - {c.get('end_formatted')}]` {c.get('text')}")

# ================= TAB 4: Architecture =================
with tab_arch:
    st.markdown("""
    ### 🌟 Production RAG Architecture
    
    ```text
    ┌─────────────────────────────────────────────────────────────┐
    │                     1. Ingestion Layer                      │
    │  PDF, DOCX, TXT, MD, CSV, Audio Transcripts (OpenAI Whisper)│
    │                           ↓                                 │
    │     Recursive Character Chunking (500 chars / 100 overlap)  │
    └─────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                   2. Dual Indexing Layer                    │
    │      Dense FAISS Index (Cosine)  +   Sparse BM25 Index      │
    │      (all-MiniLM-L6-v2, 384-dim)     (BM25Okapi Lexical)    │
    └─────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                  3. Query & Retrieval Layer                 │
    │               Multi-turn Query Rewriting                    │
    │                           ↓                                 │
    │     Reciprocal Rank Fusion (RRF) Hybrid Search              │
    │                           ↓                                 │
    │     Cross-Encoder Re-Ranking (ms-marco-MiniLM-L-6-v2)       │
    └─────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                 4. Grounding & Generation                   │
    │               Context Relevance Pre-Check                   │
    │                           ↓                                 │
    │         Google Gemini Grounded Answer Generation            │
    │                           ↓                                 │
    │      Source Citations + Confidence Breakdown + Live Code    │
    └─────────────────────────────────────────────────────────────┘
    ```
    """)
