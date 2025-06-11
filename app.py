from flask import Flask, jsonify, request
import subprocess
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])

# Global process holder
rec_process = None

@app.route('/api/start_rec', methods=['POST'])
def start_rec():
    global rec_process
    if rec_process is not None and rec_process.poll() is None:
        return jsonify({"message": "Recording already in progress"}), 400
    rec_process = subprocess.Popen(["python", "rec.py"])
    return jsonify({"message": "Recording started"}), 200

@app.route('/api/stop_rec', methods=['POST'])
def stop_rec():
    global rec_process
    if rec_process is None or rec_process.poll() is not None:
        return jsonify({"message": "No recording in progress"}), 400
    rec_process.terminate()
    rec_process = None
    return jsonify({"message": "Recording stopped"}), 200

@app.route('/api/get_status', methods=['GET'])
def get_status():
    # try:
    #     auth_output = subprocess.check_output(["python", "auth.py"]).decode("utf-8")
    # except Exception as e:
    #     auth_output = f"Auth error: {str(e)}"

    try:
        word_output = subprocess.check_output(["python", "wordrecognition.py"]).decode("utf-8")
    except Exception as e:
        word_output = f"WordRecognition error: {str(e)}"

    return jsonify({
        # "authResult": auth_output,
        "wordRecResult": word_output
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
