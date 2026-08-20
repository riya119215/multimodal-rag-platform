import pytest
from app.generation.memory import ConversationMemory
from app.generation.grounding import GroundingChecker
from app.generation.prompts import format_context_for_llm

def test_conversation_memory_turns():
    memory = ConversationMemory(max_turns=2)
    memory.add_turn("Question 1", "Answer 1")
    memory.add_turn("Question 2", "Answer 2")
    memory.add_turn("Question 3", "Answer 3")
    
    # Should only retain last 2 turns
    assert len(memory.history) == 2
    assert memory.history[0]["user"] == "Question 2"
    assert memory.history[1]["user"] == "Question 3"

def test_grounding_confidence_calculation():
    checker = GroundingChecker()
    mock_sources = [
        {
            "text": "Matplotlib bar charts are created using plt.bar(x, y).",
            "dense_score": 0.85,
            "rerank_score": 0.90,
            "source_file": "doc1.txt"
        }
    ]
    conf = checker.calculate_confidence("How to plot bar chart in Matplotlib?", mock_sources)
    assert conf["score"] > 0.6
    assert conf["level"] in ["High", "Medium"]
    assert conf["is_grounded"] is True

def test_grounding_rejection_for_empty_sources():
    checker = GroundingChecker()
    is_sufficient, msg = checker.check_context_sufficiency("What is quantum entanglement?", [])
    assert is_sufficient is False
    assert "could not find sufficient information" in msg

def test_format_context_for_llm():
    chunks = [
        {
            "doc_type": "audio_transcript",
            "video_number": "2",
            "title": "Bar Charts",
            "start_formatted": "01:00",
            "end_formatted": "02:00",
            "text": "Here is how you set bar chart colors."
        },
        {
            "doc_type": "pdf",
            "source_file": "report.pdf",
            "page_number": 4,
            "text": "Data visualization best practices."
        }
    ]
    formatted = format_context_for_llm(chunks)
    assert "Audio/Video #2" in formatted
    assert "report.pdf" in formatted
    assert "Page: 4" in formatted
