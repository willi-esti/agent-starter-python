from flask import Flask, request, jsonify
import tempfile
import os
from TTS.api import TTS
import torch

app = Flask(__name__)

# Get model name from environment variable with fallback
model_name = os.getenv("TTS_MODEL", "tts_models/en/ljspeech/vits")
print(f"Loading TTS model: {model_name}")

# Check if CUDA is available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize TTS with GPU support if available
tts = TTS(model_name=model_name, gpu=(device == "cuda"))
print("TTS model loaded successfully!")

@app.route("/synthesize", methods=["POST"])
def synthesize():
    try:
        data = request.get_json()
        text = data.get("text", "")
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Create temp file for audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            # Generate speech using Coqui TTS
            tts.tts_to_file(text=text, file_path=temp_file.name)
            
            # Read the audio file
            with open(temp_file.name, "rb") as f:
                audio_data = f.read()
            
            # Clean up
            os.unlink(temp_file.name)
            
            return audio_data, 200, {"Content-Type": "audio/wav"}
    except Exception as e:
        print(f"TTS Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": model_name, "device": device})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
