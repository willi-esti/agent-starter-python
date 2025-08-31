"""Factory for creating plugin instances with configuration."""

import os
from typing import Optional

from .whisper_stt import LocalWhisperSTT
from .ollama_llm import LocalOllamaLLM
from .coqui_tts import LocalCoquiTTS


class PluginFactory:
    """Factory class for creating configured plugin instances."""
    
    @staticmethod
    def create_whisper_stt(model_size: str = "base") -> LocalWhisperSTT:
        """Create a configured Whisper STT instance."""
        return LocalWhisperSTT(model_size=model_size)
    
    @staticmethod
    def create_ollama_llm(
        model: str = "llama3.2:1b", 
        base_url: Optional[str] = None
    ) -> LocalOllamaLLM:
        """Create a configured Ollama LLM instance."""
        if base_url is None:
            base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
        return LocalOllamaLLM(model=model, base_url=base_url)
    
    @staticmethod
    def create_coqui_tts(base_url: Optional[str] = None) -> LocalCoquiTTS:
        """Create a configured Coqui TTS instance."""
        if base_url is None:
            base_url = os.getenv("COQUI_TTS_BASE_URL", "http://coqui-tts:5000")
        return LocalCoquiTTS(base_url=base_url)
