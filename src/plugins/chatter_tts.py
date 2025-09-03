"""Chatter TTS implementation using OpenAI-compatible API."""

import logging
import requests
import json
from livekit.agents import tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

logger = logging.getLogger("agent.plugins.chatter_tts")


class ChatterTTS(tts.TTS):
    """Chatter TTS implementation using OpenAI-compatible API."""

    # Best voice, Julian.wav, Connor.wav, Elana.wav
    def __init__(self, base_url="http://chatter:8004", model="tts-1", voice="Connor.wav"):
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=False,  # Non-streaming TTS
            ),
            sample_rate=22050,
            num_channels=1,
        )
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.voice = voice
    
    def synthesize(self, text: str, *, conn_options=None, **kwargs) -> "tts.ChunkedStream":
        """Synthesize text to speech using Chatter TTS OpenAI-compatible API"""
        conn_options = conn_options or DEFAULT_API_CONNECT_OPTIONS
        return ChatterChunkedStream(
            tts=self, 
            input_text=text, 
            conn_options=conn_options, 
            base_url=self.base_url,
            model=self.model,
            voice=self.voice
        )


class ChatterChunkedStream(tts.ChunkedStream):
    """Chunked stream implementation for Chatter TTS."""
    
    def __init__(self, *, tts: ChatterTTS, input_text: str, conn_options, base_url: str, model: str, voice: str):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts = tts
        self.base_url = base_url
        self.model = model
        self.voice = voice
    
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        try:
            # Prepare the request payload for OpenAI-compatible endpoint
            payload = {
                "model": self.model,
                "input": self.input_text,
                "voice": self.voice,
                "response_format": "wav",
                "speed": 1.0
            }
            
            # Call Chatter TTS OpenAI-compatible API
            response = requests.post(
                f"{self.base_url}/v1/audio/speech",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                # Initialize the output emitter
                output_emitter.initialize(
                    request_id="chatter-tts",
                    sample_rate=self._tts.sample_rate,
                    num_channels=self._tts.num_channels,
                    mime_type="audio/wav"
                )
                
                # Get audio data and push it
                audio_data = response.content
                output_emitter.push(audio_data)
                
                # Flush to complete the stream
                output_emitter.flush()
                logger.info(f"Successfully generated TTS audio for text: {self.input_text[:50]}...")
            else:
                logger.error(f"Chatter TTS error: HTTP {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"Chatter TTS error: {e}")


class ChatterTTSCustom(tts.TTS):
    """Chatter TTS implementation using custom /tts endpoint with advanced features."""
    
    def __init__(self, base_url="http://chatter:8004", voice_mode="predefined", predefined_voice_id="default_voice.wav"):
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=False,  # Non-streaming TTS
            ),
            sample_rate=22050,
            num_channels=1,
        )
        self.base_url = base_url.rstrip('/')
        self.voice_mode = voice_mode
        self.predefined_voice_id = predefined_voice_id
    
    def synthesize(self, text: str, *, conn_options=None, **kwargs) -> "tts.ChunkedStream":
        """Synthesize text to speech using Chatter TTS custom API"""
        conn_options = conn_options or DEFAULT_API_CONNECT_OPTIONS
        return ChatterCustomChunkedStream(
            tts=self, 
            input_text=text, 
            conn_options=conn_options, 
            base_url=self.base_url,
            voice_mode=self.voice_mode,
            predefined_voice_id=self.predefined_voice_id
        )


class ChatterCustomChunkedStream(tts.ChunkedStream):
    """Chunked stream implementation for Chatter TTS custom endpoint."""
    
    def __init__(self, *, tts: ChatterTTSCustom, input_text: str, conn_options, base_url: str, voice_mode: str, predefined_voice_id: str):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts = tts
        self.base_url = base_url
        self.voice_mode = voice_mode
        self.predefined_voice_id = predefined_voice_id
    
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        try:
            # Prepare the request payload for custom /tts endpoint
            payload = {
                "text": self.input_text,
                "voice_mode": self.voice_mode,
                "predefined_voice_id": self.predefined_voice_id,
                "output_format": "wav",
                "split_text": True,
                "chunk_size": 120
            }
            
            # Call Chatter TTS custom API
            response = requests.post(
                f"{self.base_url}/tts",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                # Initialize the output emitter
                output_emitter.initialize(
                    request_id="chatter-tts-custom",
                    sample_rate=self._tts.sample_rate,
                    num_channels=self._tts.num_channels,
                    mime_type="audio/wav"
                )
                
                # Get audio data and push it
                audio_data = response.content
                output_emitter.push(audio_data)
                
                # Flush to complete the stream
                output_emitter.flush()
                logger.info(f"Successfully generated custom TTS audio for text: {self.input_text[:50]}...")
            else:
                logger.error(f"Chatter TTS custom error: HTTP {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"Chatter TTS custom error: {e}")
