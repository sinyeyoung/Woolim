# app.py
import os, tempfile, wave
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return jsonify({"message": "서버 연결 성공!"})

# 헬스체크/테스트용
@app.get("/api/hello")
def hello():
    return jsonify({"message": "Hello from Render!"})

# JSON POST 예시 (기존)
@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    return jsonify({"result": f"'{text}'를 받았습니다."})

# === STT 업로드 엔드포인트 (Flutter가 멀티파트 'file' 필드로 WAV 보낸다고 가정) ===
@app.post("/api/stt")
def stt():
    # 'file' 또는 'audio' 둘 다 허용(클라이언트 구현 편의)
    up = request.files.get("file") or request.files.get("audio")
    if not up:
        return jsonify({"error": "missing file field ('file' or 'audio')"}), 400
    if not up.filename:
        return jsonify({"error": "empty filename"}), 400

    # 임시 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        up.save(tmp.name)
        tmp_path = tmp.name

    try:
        # 간단 WAV 검증 + 정보 추출(선택)
        with wave.open(tmp_path, "rb") as wf:
            sr = wf.getframerate()
            ch = wf.getnchannels()
            frames = wf.getnframes()
            dur = frames / float(sr) if sr else 0.0

        # TODO: 여기서 실제 STT 모델/외부 API 호출
        text = "음성 인식 결과(예시)"

        return jsonify({
            "text": text,
            "sample_rate": sr,
            "channels": ch,
            "duration_sec": round(dur, 2)
        }), 200

    except wave.Error as e:
        return jsonify({"error": f"invalid WAV: {e}"}), 400
    except Exception as e:
        return jsonify({"error": f"internal error: {e}"}), 500
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

if __name__ == "__main__":
    # Render는 PORT 환경변수를 줍니다. 반드시 0.0.0.0 로 바인딩
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
