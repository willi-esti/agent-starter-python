
````
# Install if needed
winget install LiveKit.LiveKitCLI

# List rooms
lk room list --url ws://192.168.255.14:7880 --api-key devkey --api-secret secret --verbose true

# Create and join a room
lk create-room --url ws://192.168.255.14:7880 --api-key devkey --api-secret secret --verbose true --name my-room
```


# Generate a participant token for yourself
lk token create --url ws://192.168.255.14:7880 --api-key devkey --api-secret secret --room-name my-room --identity user1 --valid-for 24h

# This will give you a token that you can use in a web client
lk room join --url ws://192.168.255.14:7880 --api-key devkey --api-secret secret --room my-room --identity user1


````
docker compose exec agent-app python -m src.agent console
```


curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hello!"}]}'


  ollama pull llama3.2:1b



# For coqui-tts

docker run --rm -it -p 5002:5002 --entrypoint /bin/bash ghcr.io/coqui-ai/tts-cpu

python3 TTS/server/server.py --list_models
python3 TTS/server/server.py --model_name "tts_models/en/ljspeech/tacotron2-DDC"


curl -X POST \
  http://localhost:5002/api/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a test from Coqui TTS.",
    "model_name": "tts_models/en/ljspeech/tacotron2-DDC"
  }' \
  --output my_audio.wav

