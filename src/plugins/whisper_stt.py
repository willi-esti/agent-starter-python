"""Whisper Speech-to-Text implementation using local models."""

import logging
import numpy as np
from typing import AsyncIterator

from faster_whisper import WhisperModel
from livekit import rtc
from livekit.agents import stt

logger = logging.getLogger("agent.plugins.whisper_stt")


class WhisperModelSingleton:
    """Singleton class to manage Whisper model instances."""
    _instance = None
    _model = None
    
    def __new__(cls, model_size="base"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_model(self, model_size="base"):
        if self._model is None:
            logger.info(f"Loading Whisper model: {model_size}")
            self._model = WhisperModel(model_size, device="cpu", compute_type="int8")
        return self._model


class LocalWhisperSTT(stt.STT):
    """Local Whisper Speech-to-Text implementation."""
    
    def __init__(self, model_size="base"):
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
            )
        )
        self.model_singleton = WhisperModelSingleton(model_size)
        self.model = self.model_singleton.get_model(model_size)
    
    async def _recognize_impl(self, buffer: rtc.AudioFrame, *, language: str | None = None, **kwargs) -> stt.SpeechEvent:
        """Convert audio frame to text using local Whisper"""
        try:
            # Convert audio frame to numpy array
            audio_data = np.frombuffer(buffer.data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Handle language parameter - use None for auto-detection if not provided or if NOT_GIVEN
            whisper_language = None if not language or language == "NOT_GIVEN" else language
            
            # Transcribe using faster-whisper
            segments, _ = self.model.transcribe(audio_data, language=whisper_language)
            text = " ".join([segment.text for segment in segments])
            
            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[
                    stt.SpeechData(
                        text=text.strip(),
                        language=language or "en",
                        confidence=0.9,
                    )
                ]
            )
        except Exception as e:
            logger.error(f"STT error: {e}")
            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[
                    stt.SpeechData(
                        text="",
                        language=language or "en", 
                        confidence=0.0,
                    )
                ]
            )
