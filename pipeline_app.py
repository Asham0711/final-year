from flask import Flask, render_template, request, jsonify
import subprocess
import threading
import os

# Flask app initialization
app = Flask(__name__)

# Global variable to hold pipeline results
pipeline_results = {
    "recording_status": "Idle",
    "auth_result": None,
    "word_result": None
}

# Function to run the recording script
def record_audio():
    global pipeline_results
    pipeline_results["recording_status"] = "Recording..."

    # Run rec.py
    try:
        subprocess.run(["python", "rec.py"], check=True)
        pipeline_results["recording_status"] = "Recording Completed"
    except subprocess.CalledProcessError as e:
        pipeline_results["recording_status"] = f"Error: {str(e)}"

# Function to authenticate audio using auth.py
def authenticate_audio(file_path):
    result = None
    try:
        process = subprocess.run(["python", "auth.py"], text=True, capture_output=True, check=True)
        output = process.stdout.strip()
        result = output
    except subprocess.CalledProcessError as e:
        result = f"Error: {str(e)}"
    return result

# Function to detect emergency words using word.py
def detect_words(file_path):
    result = None
    try:
        process = subprocess.run(["python", "word.py"], text=True, capture_output=True, check=True)
        output = process.stdout.strip()
        result = output
    except subprocess.CalledProcessError as e:
        result = f"Error: {str(e)}"
    return result

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", status=pipeline_results)

@app.route("/start_pipeline", methods=["POST"])
def start_pipeline():
    global pipeline_results
    pipeline_results["recording_status"] = "Recording..."

    # Start recording in a separate thread
    threading.Thread(target=record_audio).start()

    # Dummy file name for the recorded audio (to integrate with auth.py and word.py)
    audio_file = "chunk_1.wav"

    # Authenticate the audio
    pipeline_results["auth_result"] = authenticate_audio(audio_file)

    # Detect words
    pipeline_results["word_result"] = detect_words(audio_file)

    return jsonify(pipeline_results)

if __name__ == "__main__":
    app.run(debug=True)
