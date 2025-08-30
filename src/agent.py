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
                streaming=False,  # Non-streaming TTS
            ),
            sample_rate=22050,
            num_channels=1,
        )
        self.base_url = base_url
    
    def synthesize(self, text: str, *, conn_options=None, **kwargs) -> "tts.ChunkedStream":
        """Synthesize text to speech using Coqui TTS"""
        from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
        conn_options = conn_options or DEFAULT_API_CONNECT_OPTIONS
        return CoquiChunkedStream(tts=self, input_text=text, conn_options=conn_options, base_url=self.base_url)


class CoquiChunkedStream(tts.ChunkedStream):
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


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice AI assistant.
            You eagerly assist users with their questions by providing information from your extensive knowledge.
            Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            You are curious, friendly, and have a sense of humor.
            
            Always respond with plain text. Do not use any function calls or JSON formatting in your responses.""",
        )

    # Temporarily commented out the function tool to debug the issue
    # @function_tool
    # async def lookup_weather(self, context: RunContext, location: str):
    #     """ONLY use this tool when the user explicitly asks about weather conditions for a specific location.
    #     
    #     Examples of when to use this tool:
    #     - "What's the weather like in Paris?"
    #     - "How's the weather in Tokyo today?"
    #     - "Is it raining in London?"
    #     
    #     Do NOT use this tool for:
    #     - General greetings like "hello", "hi"
    #     - General questions not about weather
    #     - Conversations about other topics

    #     Args:
    #         location: The location to look up weather information for (e.g. city name)
    #     """

    #     logger.info(f"Looking up weather for {location}")

    #     return "sunny with a temperature of 70 degrees."


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
