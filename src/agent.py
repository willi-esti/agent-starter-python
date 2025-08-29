import logging
import asyncio
import io
import numpy as np
import os
import requests
from typing import AsyncIterator

from dotenv import load_dotenv
from faster_whisper import WhisperModel
import ollama
from livekit import rtc
from livekit.agents import (
    NOT_GIVEN,
    Agent,
    AgentFalseInterruptionEvent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.llm import function_tool, LLM
from livekit.agents import stt, tts
from livekit.plugins import cartesia, deepgram, noise_cancellation, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class WhisperModelSingleton:
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


class LocalOllamaLLM(LLM):
    def __init__(self, model="llama3.2:1b", base_url="http://ollama:11434"):
        super().__init__()
        self.model = model
        self.base_url = base_url
        self.client = ollama.Client(host=base_url)
    
    async def _generate_impl(self, chat_ctx, function_ctx):
        """Generate response using local Ollama"""
        try:
            # Convert chat context to Ollama format
            prompt = ""
            for msg in chat_ctx.messages:
                if hasattr(msg, 'content'):
                    content = msg.content
                elif isinstance(msg, dict):
                    content = msg.get('content', str(msg))
                else:
                    content = str(msg)
                prompt += f"{content}\n"
            
            # Call Ollama
            response = self.client.generate(model=self.model, prompt=prompt)
            
            # Return the response in the expected format
            from livekit.agents.llm import ChatMessage
            return ChatMessage.create(
                text=response.get('response', ''),
                role="assistant"
            )
        except Exception as e:
            logger.error(f"LLM error: {e}")
            from livekit.agents.llm import ChatMessage
            return ChatMessage.create(
                text="I'm sorry, I'm having trouble processing that request.",
                role="assistant"
            )


class LocalWhisperSTT(stt.STT):
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


class LocalCoquiTTS(tts.TTS):
    def __init__(self, base_url="http://coqui-tts:5000"):
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=True,  # Set streaming to True for this approach
            ),
            sample_rate=22050,
            num_channels=1,
        )
        self.base_url = base_url
    
    # Re-implement the required abstract method
    def synthesize(self, text: str, *, conn_options=None, **kwargs) -> "tts.SynthesizeStream":
        """Synthesize text to speech using Coqui TTS and return a stream."""
        return CoquiSynthesizeStream(text, self.base_url, self.sample_rate, self.num_channels, tts=self, conn_options=conn_options)


class CoquiSynthesizeStream(tts.SynthesizeStream):
    def __init__(self, text: str, base_url: str, sample_rate: int, num_channels: int, *, tts, conn_options=None):
        super().__init__(tts=tts, conn_options=conn_options)
        self.text = text
        self.base_url = base_url
        self.sample_rate = sample_rate
        self.num_channels = num_channels
    
    async def _run(self, output_emitter):
        try:
            response = requests.post(
                f"{self.base_url}/synthesize",
                json={"text": self.text}
            )
            
            if response.status_code == 200:
                # Get audio data as a byte stream
                audio_bytes = response.content
                
                # Convert the byte stream to a NumPy array of 16-bit integers
                audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
                
                # Create the AudioFrame
                frame = rtc.AudioFrame.create(
                    sample_rate=self.sample_rate,
                    num_channels=self.num_channels,
                    samples_per_channel=len(audio_np)
                )
                
                # Copy the NumPy array data into the AudioFrame
                frame.data[:] = audio_np.tobytes()
                await output_emitter(frame)
            else:
                logger.error(f"TTS error: HTTP {response.status_code}")
                
        except Exception as e:
            logger.error(f"TTS error: {e}")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice AI assistant.
            You eagerly assist users with their questions by providing information from your extensive knowledge.
            Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            You are curious, friendly, and have a sense of humor.""",
        )

    # all functions annotated with @function_tool will be passed to the LLM when this
    # agent is active
    @function_tool
    async def lookup_weather(self, context: RunContext, location: str):
        """Use this tool to look up current weather information in the given location.

        If the location is not supported by the weather service, the tool will indicate this. You must tell the user the location's weather is unavailable.

        Args:
            location: The location to look up weather information for (e.g. city name)
        """

        logger.info(f"Looking up weather for {location}")

        return "sunny with a temperature of 70 degrees."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, Deepgram, and the LiveKit turn detector
    session = AgentSession(
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all providers at https://docs.livekit.io/agents/integrations/llm/
        llm=openai.LLM(
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")
        ),
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all providers at https://docs.livekit.io/agents/integrations/stt/
        stt=LocalWhisperSTT(),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all providers at https://docs.livekit.io/agents/integrations/tts/
        tts=LocalCoquiTTS(),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead:
    # session = AgentSession(
    #     # See all providers at https://docs.livekit.io/agents/integrations/realtime/
    #     llm=openai.realtime.RealtimeModel()
    # )

    # sometimes background noise could interrupt the agent session, these are considered false positive interruptions
    # when it's detected, you may resume the agent's speech
    @session.on("agent_false_interruption")
    def _on_agent_false_interruption(ev: AgentFalseInterruptionEvent):
        logger.info("false positive interruption, resuming")
        session.generate_reply(instructions=ev.extra_instructions or NOT_GIVEN)

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/integrations/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/integrations/avatar/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # LiveKit Cloud enhanced noise cancellation
            # - If self-hosting, omit this parameter
            # - For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
