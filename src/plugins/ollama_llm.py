"""Ollama Local LLM implementation."""

import logging
import ollama
from livekit.agents.llm import LLM, ChatMessage

logger = logging.getLogger("agent.plugins.ollama_llm")


class LocalOllamaLLM(LLM):
    """Local Ollama Large Language Model implementation."""
    
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
            return ChatMessage.create(
                text=response.get('response', ''),
                role="assistant"
            )
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return ChatMessage.create(
                text="I'm sorry, I'm having trouble processing that request.",
                role="assistant"
            )
