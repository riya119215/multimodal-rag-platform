from typing import Dict, Any, Optional, List
from app.retrieval.hybrid_search import HybridRetriever
from app.generation.llm import GeminiLLM
from app.generation.memory import ConversationMemory, QueryRewriter
from app.generation.grounding import GroundingChecker
from app.generation.prompts import format_context_for_llm, SYSTEM_RAG_PROMPT
from app.utils.helpers import extract_python_code
from app.core.config import settings
from app.core.logging_config import logger

class RAGAnswerGenerator:
    """
    End-to-End RAG Orchestrator:
    1. Query Rewriting (multi-turn context resolution)
    2. Hybrid Retrieval (FAISS Dense + BM25 Sparse + Re-ranking)
    3. Grounding & Sufficiency Verification
    4. LLM Generation via Gemini
    5. Confidence Assessment & Source Attribution
    """
    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        llm: Optional[GeminiLLM] = None,
        memory: Optional[ConversationMemory] = None,
        grounding_checker: Optional[GroundingChecker] = None
    ):
        self.retriever = retriever or HybridRetriever()
        self.llm = llm or GeminiLLM()
        self.memory = memory or ConversationMemory()
        self.rewriter = QueryRewriter(llm=self.llm)
        self.grounding = grounding_checker or GroundingChecker()

    def generate_answer(
        self,
        query: str,
        top_k: int = settings.TOP_K,
        retrieval_mode: str = settings.RETRIEVAL_MODE,
        min_score: float = settings.MIN_SIMILARITY_SCORE,
        use_memory: bool = True,
        filter_metadata: Optional[Dict[str, Any]] = None,
        allow_agentic_fallback: bool = False
    ) -> Dict[str, Any]:
        """Process a query end-to-end and generate a grounded, cited response."""
        clean_query = query.strip()
        if not clean_query:
            return {
                "query": query,
                "rewritten_query": query,
                "answer": "Please provide a valid question.",
                "sources": [],
                "confidence": {"score": 0.0, "percentage": "0%", "level": "Low", "is_grounded": False},
                "code_blocks": [],
                "is_agentic_fallback": False
            }

        # Step 1: Query Rewriting for follow-up turns
        rewritten_query = clean_query
        if use_memory and self.memory.history:
            rewritten_query = self.rewriter.rewrite(clean_query, self.memory)

        # Step 2: Multi-stage Hybrid Retrieval
        logger.info(f"Retrieving context for query: '{rewritten_query}' (mode={retrieval_mode}, top_k={top_k})...")
        sources = self.retriever.retrieve(
            query=rewritten_query,
            top_k=top_k,
            mode=retrieval_mode,
            min_score=min_score,
            filter_metadata=filter_metadata
        )

        # Step 3: Grounding & Sufficiency Pre-check
        is_sufficient, refusal_msg = self.grounding.check_context_sufficiency(rewritten_query, sources)
        confidence_info = self.grounding.calculate_confidence(rewritten_query, sources)

        # Agentic Fallback Handler
        if not is_sufficient:
            if allow_agentic_fallback:
                logger.info("Context insufficient; using Agentic General AI Knowledge Fallback...")
                fallback_prompt = f"""The user asked: "{clean_query}"
This topic was NOT found in the indexed documents. Provide a helpful, clear general technical answer, but prefix the response with a clear disclaimer:
"ℹ️ *[General AI Knowledge - Not found in indexed documents]*"
"""
                general_answer = self.llm.generate(prompt=fallback_prompt, temperature=0.3)
                code_blocks = extract_python_code(general_answer)
                if use_memory:
                    self.memory.add_turn(clean_query, general_answer)
                return {
                    "query": clean_query,
                    "rewritten_query": rewritten_query,
                    "answer": general_answer,
                    "sources": [],
                    "confidence": {"score": 0.35, "percentage": "35%", "level": "Low", "is_grounded": False},
                    "code_blocks": code_blocks,
                    "is_agentic_fallback": True
                }
            else:
                logger.info("Context insufficient for grounded answer generation.")
                return {
                    "query": clean_query,
                    "rewritten_query": rewritten_query,
                    "answer": refusal_msg,
                    "sources": sources,
                    "confidence": confidence_info,
                    "code_blocks": [],
                    "is_agentic_fallback": False
                }

        # Step 4: Context Formatting (Small-to-Big Parent Context)
        context_text = format_context_for_llm(sources)

        user_prompt = f"""USER QUESTION:
{clean_query}

RELEVANT RETRIEVED CONTEXT:
{context_text}

Please provide a clear, accurate, and source-cited answer based strictly on the context above:"""

        # Step 5: LLM Synthesis
        logger.info("Invoking Gemini for grounded answer synthesis...")
        answer = self.llm.generate(
            prompt=user_prompt,
            system_instruction=SYSTEM_RAG_PROMPT,
            temperature=0.2
        )

        # Step 6: Code extraction & Post-processing
        code_blocks = extract_python_code(answer)

        # Step 7: Update Conversation Memory
        if use_memory:
            self.memory.add_turn(clean_query, answer)

        return {
            "query": clean_query,
            "rewritten_query": rewritten_query,
            "answer": answer,
            "sources": sources,
            "confidence": confidence_info,
            "code_blocks": code_blocks,
            "is_agentic_fallback": False
        }

    def generate_answer_stream(
        self,
        query: str,
        top_k: int = settings.TOP_K,
        retrieval_mode: str = settings.RETRIEVAL_MODE,
        min_score: float = settings.MIN_SIMILARITY_SCORE,
        use_memory: bool = True,
        filter_metadata: Optional[Dict[str, Any]] = None,
        allow_agentic_fallback: bool = False
    ):
        """
        Stream answer tokens one-by-one.
        Returns a tuple: (token_generator, metadata_dict)
        """
        clean_query = query.strip()
        if not clean_query:
            def empty_gen():
                yield "Please provide a valid question."
            return empty_gen(), {
                "query": query, "sources": [], "confidence": {"score": 0.0, "percentage": "0%", "level": "Low", "is_grounded": False}, "is_agentic_fallback": False
            }

        rewritten_query = clean_query
        if use_memory and self.memory.history:
            rewritten_query = self.rewriter.rewrite(clean_query, self.memory)

        sources = self.retriever.retrieve(
            query=rewritten_query,
            top_k=top_k,
            mode=retrieval_mode,
            min_score=min_score,
            filter_metadata=filter_metadata
        )

        is_sufficient, refusal_msg = self.grounding.check_context_sufficiency(rewritten_query, sources)
        confidence_info = self.grounding.calculate_confidence(rewritten_query, sources)

        if not is_sufficient:
            if allow_agentic_fallback:
                fallback_prompt = f"""The user asked: "{clean_query}"
This topic was NOT found in the indexed documents. Provide a helpful, clear general technical answer, prefixing with:
"ℹ️ *[General AI Knowledge - Not found in indexed documents]*\n\n"
"""
                meta = {
                    "query": clean_query, "rewritten_query": rewritten_query, "sources": [],
                    "confidence": {"score": 0.35, "percentage": "35%", "level": "Low", "is_grounded": False}, "is_agentic_fallback": True
                }
                return self.llm.generate_stream(fallback_prompt, temperature=0.3), meta
            else:
                def refusal_gen():
                    yield refusal_msg
                meta = {
                    "query": clean_query, "rewritten_query": rewritten_query, "sources": sources,
                    "confidence": confidence_info, "is_agentic_fallback": False
                }
                return refusal_gen(), meta

        context_text = format_context_for_llm(sources)
        user_prompt = f"""USER QUESTION:
{clean_query}

RELEVANT RETRIEVED CONTEXT:
{context_text}

Please provide a clear, accurate, and source-cited answer based strictly on the context above:"""

        meta = {
            "query": clean_query,
            "rewritten_query": rewritten_query,
            "sources": sources,
            "confidence": confidence_info,
            "is_agentic_fallback": False
        }
        return self.llm.generate_stream(user_prompt, system_instruction=SYSTEM_RAG_PROMPT, temperature=0.2), meta

