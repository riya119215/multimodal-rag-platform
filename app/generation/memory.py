from typing import List, Dict, Optional
from app.generation.prompts import SYSTEM_QUERY_REWRITE_PROMPT
from app.core.logging_config import logger

class ConversationMemory:
    """Maintains short-term conversational context for multi-turn dialogues."""
    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self.history: List[Dict[str, str]] = []

    def add_turn(self, user_message: str, assistant_response: str):
        """Add a dialogue turn to the buffer."""
        self.history.append({
            "user": user_message.strip(),
            "assistant": assistant_response.strip()
        })
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]

    def clear(self):
        """Reset conversation history."""
        self.history.clear()

    def get_formatted_history(self) -> str:
        """Format history for prompt ingestion."""
        if not self.history:
            return ""
        lines = []
        for i, turn in enumerate(self.history, start=1):
            lines.append(f"Turn {i}:")
            lines.append(f"User: {turn['user']}")
            lines.append(f"Assistant: {turn['assistant'][:200]}...")
        return "\n".join(lines)

class QueryRewriter:
    """Reformulates conversational follow-up questions into standalone retrieval queries."""
    def __init__(self, llm=None):
        self.llm = llm

    def rewrite(self, query: str, memory: ConversationMemory) -> str:
        """
        Rewrite query if conversational context exists and query looks like a follow-up.
        """
        if not memory.history or not query.strip():
            return query

        # Pronoun & reference heuristics
        indicators = ["it", "its", "they", "them", "these", "those", "this", "that", "explain more", "why", "how", "what about"]
        words = query.lower().split()
        is_followup = any(w in words for w in indicators) or len(words) <= 4

        if not is_followup or not self.llm or not self.llm.client_ready:
            return query

        history_text = memory.get_formatted_history()
        prompt = f"""CONVERSATION HISTORY:
{history_text}

FOLLOW-UP QUESTION:
{query}

REWRITTEN STANDALONE SEARCH QUERY:"""

        try:
            rewritten = self.llm.generate(
                prompt=prompt,
                system_instruction=SYSTEM_QUERY_REWRITE_PROMPT,
                temperature=0.1
            )
            if rewritten and len(rewritten) > 3 and not rewritten.startswith("⚠️"):
                logger.info(f"Query rewritten: '{query}' -> '{rewritten}'")
                return rewritten
        except Exception as e:
            logger.warning(f"Query rewriting failed: {e}. Using original query.")

        return query
