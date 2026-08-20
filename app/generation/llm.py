import os
from typing import Optional, List, Dict, Any
import google.generativeai as genai

from app.core.config import settings
from app.core.logging_config import logger
from app.generation.prompts import SYSTEM_RAG_PROMPT

class GeminiLLM:
    """
    Client interface for Google Gemini models with model fallback and error resilience.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = settings.LLM_MODEL
    ):
        self.api_key = api_key or settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self.client_ready = False
        self._init_client()

    def _init_client(self):
        if not self.api_key:
            logger.warning("GEMINI_API_KEY is not set. LLM generation will be unavailable until key is provided.")
            return

        try:
            genai.configure(api_key=self.api_key, transport="rest")
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=SYSTEM_RAG_PROMPT
            )
            self.client_ready = True
            logger.info(f"Gemini client initialized with model: '{self.model_name}'")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            self.client_ready = False

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2
    ) -> str:
        """Generate text using Gemini with multi-model fallback."""
        if not self.client_ready:
            self._init_client()

        if not self.client_ready:
            return (
                "⚠️ **LLM Generation Unavailable**: GEMINI_API_KEY is not configured.\n"
                "Please configure your `GEMINI_API_KEY` in the `.env` file or provide it via the settings panel."
            )

        models_to_try = [
            self.model_name,
            "gemini-3.6-flash",
            "gemini-flash-latest",
            "gemini-2.5-flash",
            "gemini-1.5-flash"
        ]
        # Remove duplicates while preserving order
        models_to_try = list(dict.fromkeys(models_to_try))

        last_error = None
        for model_id in models_to_try:
            try:
                model_inst = genai.GenerativeModel(
                    model_name=model_id,
                    system_instruction=system_instruction or SYSTEM_RAG_PROMPT,
                    generation_config=genai.GenerationConfig(temperature=temperature)
                )
                response = model_inst.generate_content(prompt)
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                logger.warning(f"Generation failed with {model_id}: {e}. Trying fallback...")
                last_error = e

        return f"⚠️ Error generating answer with Gemini: {str(last_error)}"

    def generate_stream(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2
    ):
        """Yield tokens streamingly using Gemini with multi-model fallback."""
        if not self.client_ready:
            self._init_client()

        if not self.client_ready:
            yield (
                "⚠️ **LLM Generation Unavailable**: GEMINI_API_KEY is not configured.\n"
                "Please configure your `GEMINI_API_KEY` in the `.env` file or provide it via the settings panel."
            )
            return

        models_to_try = [
            self.model_name,
            "gemini-3.6-flash",
            "gemini-flash-latest",
            "gemini-2.5-flash",
            "gemini-1.5-flash"
        ]
        models_to_try = list(dict.fromkeys(models_to_try))

        last_error = None
        for model_id in models_to_try:
            try:
                model_inst = genai.GenerativeModel(
                    model_name=model_id,
                    system_instruction=system_instruction or SYSTEM_RAG_PROMPT,
                    generation_config=genai.GenerationConfig(temperature=temperature)
                )
                response = model_inst.generate_content(prompt, stream=True)
                for chunk in response:
                    if chunk and chunk.text:
                        yield chunk.text
                return
            except Exception as e:
                logger.warning(f"Streaming failed with {model_id}: {e}. Trying fallback...")
                last_error = e

        yield f"⚠️ Error generating stream with Gemini: {str(last_error)}"

