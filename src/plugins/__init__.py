"""Plugin modules for the agent."""

from .whisper_stt import WhisperModelSingleton, LocalWhisperSTT
from .ollama_llm import LocalOllamaLLM
from .coqui_tts import LocalCoquiTTS, CoquiChunkedStream
from .chatter_tts import ChatterTTS, ChatterTTSCustom
from .factory import PluginFactory

__all__ = [
    'WhisperModelSingleton',
    'LocalWhisperSTT', 
    'LocalOllamaLLM',
    'LocalCoquiTTS',
    'CoquiChunkedStream',
    'ChatterTTS',
    'ChatterTTSCustom',
    'PluginFactory'
]
