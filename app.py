# app.py (추가/교체)

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return jsonify({"message": "서버 연결 성공!"})

# ---- 여기부터 추가: STT 엔드포인트 (/api/stt 와 /stt 둘 다 지원) ----

def _handle_stt_request():
    """
    1) multipart/form-data: field 이름이 'file' 또는 'audio'
    2) raw body: Content-Type: audio/wav (또는 audio/*)
    현재는 연결 확인용으로 파일을 임시 저장한 뒤 길이만 확인하고
    더미 텍스트를 반환합니다.
    """
    # 1) 멀티파트 수신
    if request.content_type and "multipart/form-data" in request.content_type:
        file = request.files.get("file") or request.files.get("audio")
        if not file:
            return jsonify({"error": "no_file", "detail": "multipart field 'file' or 'audio' not found"}), 400

        # 임시 저장
        with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as tmp:
            file.save(tmp.name)
            size = os.path.getsize(tmp.name)

        # TODO: 여기에서 실제 STT 호출로 교체 (Faster-Whisper 등)
        return jsonify({"text": f"음성 수신 OK (multipart, {size} bytes)"}), 200

    # 2) RAW (audio/wav)
    if request.data:
        # 간단한 검사: 앞 몇 바이트 확인 (RIFF/WAVE 헤더 여부)
        head = request.data[:12]
        # 임시 저장
        with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as tmp:
            tmp.write(request.data)
            tmp.flush()
            size = os.path.getsize(tmp.name)

        # TODO: 여기에서 실제 STT 호출로 교체
        return jsonify({"text": f"음성 수신 OK (raw, {size} bytes, head={list(head)})"}), 200

    return jsonify({"error": "empty_body", "detail": "no audio payload"}), 400


@app.route("/api/stt", methods=["POST"])
def stt_api():
    return _handle_stt_request()

@app.route("/stt", methods=["POST"])
def stt_root():
    return _handle_stt_request()
# ---- 추가 끝 ----


if __name__ == "__main__":
    # Render 등에서는 환경 변수 PORT를 쓰기도 합니다. 로컬 테스트용.
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
