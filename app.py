import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai

app = Flask(__name__)
CORS(app)

# OpenAI API 키 로드 (Render Environment Variable에서 불러옴)
openai.api_key = os.environ.get("OPENAI_API_KEY")

@app.route("/")
def home():
    return jsonify({"message": "서버 연결 성공!"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    return jsonify({"result": f"'{text}'를 받았습니다."})

#  실제 STT 엔드포인트
@app.route("/api/stt", methods=["POST"])
def stt():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        audio_file = request.files["file"]

        # OpenAI Whisper API 호출
        transcript = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )

        return jsonify({"text": transcript.text}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
