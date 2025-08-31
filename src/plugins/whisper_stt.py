"""Whisper Speech-to-Text implementation using local models."""

import logging
import numpy as np
import asyncio
from typing import AsyncIterator

from faster_whisper import WhisperModel
from livekit import rtc
from livekit.agents import stt

logger = logging.getLogger("agent.plugins.whisper_stt")


class WhisperModelSingleton:
    """Singleton class to manage Whisper model instances."""
    _instance = None
    _model = None
    _loading = False
    
    def __new__(cls, model_size="small"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def get_model(self, model_size="small"):
        if self._model is None and not self._loading:
            self._loading = True
            try:
                logger.info(f"Loading Whisper model: {model_size}")
                # Load model in executor to avoid blocking
                loop = asyncio.get_event_loop()
                self._model = await loop.run_in_executor(
                    None, 
                    lambda: WhisperModel(model_size, device="cpu", compute_type="int8")
                )
                logger.info(f"Whisper model {model_size} loaded successfully")
            finally:
                self._loading = False
        elif self._loading:
            # Wait for loading to complete
            while self._loading:
                await asyncio.sleep(0.1)
        return self._model


class LocalWhisperSTT(stt.STT):
    """Local Whisper Speech-to-Text implementation."""
    
    def __init__(self, model_size="small"):
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
            )
        )
        self.model_singleton = WhisperModelSingleton(model_size)
        self.model_size = model_size
        self.model = None
    
    async def _ensure_model_loaded(self):
        """Ensure the model is loaded before use."""
        if self.model is None:
            self.model = await self.model_singleton.get_model(self.model_size)
    
    async def _recognize_impl(self, buffer: rtc.AudioFrame, *, language: str | None = None, **kwargs) -> stt.SpeechEvent:
        """Convert audio frame to text using local Whisper"""
        try:
            # Ensure model is loaded
            await self._ensure_model_loaded()
            
            # Convert audio frame to numpy array
            audio_data = np.frombuffer(buffer.data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Handle language parameter - use None for auto-detection if not provided or if NOT_GIVEN
            whisper_language = None if not language or language == "NOT_GIVEN" else language
            
            # Transcribe using faster-whisper in executor to avoid blocking
            loop = asyncio.get_event_loop()
            segments, _ = await loop.run_in_executor(
                None,
                lambda: self.model.transcribe(audio_data, language=whisper_language)
            )
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
