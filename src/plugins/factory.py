"""Factory for creating plugin instances with configuration."""

import os
import asyncio
from typing import Optional

from .whisper_stt import LocalWhisperSTT
from .ollama_llm import LocalOllamaLLM
from .coqui_tts import LocalCoquiTTS


class PluginFactory:
    """Factory class for creating configured plugin instances."""
    
    @staticmethod
    def create_whisper_stt(model_size: Optional[str] = None) -> LocalWhisperSTT:
        """Create a configured Whisper STT instance."""
        if model_size is None:
            model_size = os.getenv("WHISPER_MODEL", "small")
        return LocalWhisperSTT(model_size=model_size)
    
    @staticmethod
    async def prewarm_whisper_stt(model_size: Optional[str] = None) -> LocalWhisperSTT:
        """Create and prewarm a Whisper STT instance."""
        if model_size is None:
            model_size = os.getenv("WHISPER_MODEL", "small")
        stt_instance = LocalWhisperSTT(model_size=model_size)
        await stt_instance._ensure_model_loaded()
        return stt_instance
    
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
