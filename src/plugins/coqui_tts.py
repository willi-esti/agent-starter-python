"""Coqui Text-to-Speech implementation."""

import logging
import requests
from livekit.agents import tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

logger = logging.getLogger("agent.plugins.coqui_tts")


class LocalCoquiTTS(tts.TTS):
    """Local Coqui Text-to-Speech implementation."""
    
    def __init__(self, base_url="http://coqui-tts:5000"):
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=False,  # Non-streaming TTS
            ),
            sample_rate=22050,
            num_channels=1,
        )
        self.base_url = base_url
    
    def synthesize(self, text: str, *, conn_options=None, **kwargs) -> "tts.ChunkedStream":
        """Synthesize text to speech using Coqui TTS"""
        conn_options = conn_options or DEFAULT_API_CONNECT_OPTIONS
        return CoquiChunkedStream(tts=self, input_text=text, conn_options=conn_options, base_url=self.base_url)


class CoquiChunkedStream(tts.ChunkedStream):
    """Chunked stream implementation for Coqui TTS."""
    
    def __init__(self, *, tts: LocalCoquiTTS, input_text: str, conn_options, base_url: str):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts = tts
        self.base_url = base_url
    
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        try:
            # Call Coqui TTS API
            response = requests.post(
                f"{self.base_url}/synthesize",
                json={"text": self.input_text}
            )
            
            if response.status_code == 200:
                # Initialize the output emitter (this is the missing step!)
                output_emitter.initialize(
                    request_id="coqui-tts",
                    sample_rate=self._tts.sample_rate,
                    num_channels=self._tts.num_channels,
                    mime_type="audio/wav"
                )
                
                # Get audio data and push it
                audio_data = response.content
                output_emitter.push(audio_data)
                
                # Flush to complete the stream
                output_emitter.flush()
            else:
                logger.error(f"TTS error: HTTP {response.status_code}")
                
        except Exception as e:
            logger.error(f"TTS error: {e}")
