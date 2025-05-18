import sys
import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import vosk
import sounddevice as sd
import numpy as np
import queue
import threading
import time
from llama_cpp import Llama

app = Flask(__name__)
CORS(app)

# Global variables
vosk_model = None
llm = None
audio_queue = queue.Queue()
recording = False
audio_data = []
samplerate = 16000
silence_threshold = 0.5
audio_level_threshold = 0.01

def initialize_models():
    global vosk_model, llm
    
    # Initialize Vosk model
    model_path = os.path.join(os.path.dirname(__file__), "vosk-model-small-en-us-0.15")
    if not os.path.exists(model_path):
        raise Exception(f"Vosk model not found at {model_path}")
    vosk_model = vosk.Model(model_path)
    
    # Initialize GGUF model
    gguf_model_path = os.path.join(os.path.dirname(__file__), "model.gguf")
    if not os.path.exists(gguf_model_path):
        raise Exception(f"GGUF model not found at {gguf_model_path}")
    llm = Llama(
        model_path=gguf_model_path,
        n_ctx=2048,
        n_threads=4
    )

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    if recording:
        audio_data.extend(indata[:, 0])
        audio_queue.put(indata[:, 0])

def start_listening():
    global recording, audio_data
    recording = True
    audio_data = []
    
    with sd.InputStream(samplerate=samplerate, channels=1, callback=audio_callback):
        while recording:
            time.sleep(0.1)
            if len(audio_data) > 0:
                audio_level = np.abs(np.array(audio_data[-int(samplerate * 0.1):])).mean()
                if audio_level < audio_level_threshold:
                    time.sleep(silence_threshold)
                    if np.abs(np.array(audio_data[-int(samplerate * 0.1):])).mean() < audio_level_threshold:
                        recording = False

def stop_listening():
    global recording
    recording = False

def process_audio():
    global audio_data
    if len(audio_data) == 0:
        return ""
    
    audio_data_np = np.array(audio_data, dtype=np.float32)
    audio_data_int16 = (audio_data_np * 32768).astype(np.int16)
    
    rec = vosk_model.AcceptWaveform(audio_data_int16.tobytes())
    if rec:
        result = json.loads(rec.Result())
        return result.get("text", "")
    return ""

def get_llm_response(text):
    if not text:
        return ""
    
    response = llm(
        text,
        max_tokens=100,
        temperature=0.7,
        top_p=0.95,
        repeat_penalty=1.1,
    )
    return response['choices'][0]['text'].strip()

@app.route('/initialize', methods=['POST'])
def initialize():
    try:
        initialize_models()
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/start_listening', methods=['POST'])
def api_start_listening():
    threading.Thread(target=start_listening).start()
    return jsonify({"status": "listening"})

@app.route('/stop_listening', methods=['POST'])
def api_stop_listening():
    stop_listening()
    text = process_audio()
    return jsonify({"text": text})

@app.route('/get_response', methods=['POST'])
def api_get_response():
    data = request.json
    text = data.get('text', '')
    response = get_llm_response(text)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000) 