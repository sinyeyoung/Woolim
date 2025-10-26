# app.py
import re
import difflib
import os
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS

# ffmpeg 바이너리 준비 (Render에서 apt 사용 불가 → imageio-ffmpeg 사용)
try:
    import imageio_ffmpeg
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = os.path.dirname(ffmpeg_path)
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
except Exception as _:
    # 실패해도 WAV(PCM)만 다루면 운 좋게 통과할 수 있음. 되도록 위 패키지 설치 권장.
    pass

app = Flask(__name__)
CORS(app)

# 업로드 용량 제한 (Render 무료는 과도한 큰 파일 비추천)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB

# ---- 모델 설정 (환경변수로 조정 가능) ----
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")   # tiny | base | small ...
DEVICE        = os.environ.get("WHISPER_DEVICE", "cpu")   # cpu (Render 무료는 GPU 없음)
COMPUTE_TYPE  = os.environ.get("WHISPER_COMPUTE", "int8") # int8 권장(저메모리)
BEAM_SIZE     = int(os.environ.get("WHISPER_BEAM", "1"))  # 1=greedy
LANGUAGE      = os.environ.get("WHISPER_LANG", "ko")      # 한국어 위주면 "ko"

_model = None  # lazy load

def get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        # 첫 호출 시 로드 (모델은 /opt/render/project/.cache 아래로 캐시됨)
        _model = WhisperModel(WHISPER_MODEL, device=DEVICE, compute_type=COMPUTE_TYPE)
    return _model

@app.route("/")
def home():
    return jsonify({"message": "서버 연결 성공!"})

@app.route("/health")
def health():
    return jsonify({"ok": True})

def _stt_from_file(tmp_path: str) -> str:
    """
    faster-whisper로 transcribe 수행.
    파일은 WAV 16kHz mono 기준(지금 Flutter가 그렇게 보냄).
    """
    model = get_model()

    wav16 = _ensure_16k_mono(tmp_path)
    
    # 간단/저자원 설정 (Render 프리티어 고려)
    segments, info = model.transcribe( 
        tmp_path,
        language="ko",
        task="transcribe",
        beam_size=1,
        temperature=0.0,
        vad_filter=False,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    text = " ".join([seg.text.strip() for seg in segments]).strip()
    return text

def _handle_stt_request():
    # 1) multipart/form-data
    if request.content_type and "multipart/form-data" in request.content_type:
        file = request.files.get("file") or request.files.get("audio")
        if not file:
            return jsonify({"error": "no_file", "detail": "Use field 'file' or 'audio'."}), 400
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        try:
            text = _stt_from_file(tmp_path)
            return jsonify({"text": text}), 200
        except Exception as e:
            return jsonify({"error": "stt_failed", "detail": str(e)}), 500
        finally:
            try: os.remove(tmp_path)
            except: pass

    # 2) RAW (audio/wav)
    if request.data:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(request.data)
            tmp.flush()
            tmp_path = tmp.name
        try:
            text = _stt_from_file(tmp_path)
            return jsonify({"text": text}), 200
        except Exception as e:
            return jsonify({"error": "stt_failed", "detail": str(e)}), 500
        finally:
            try: os.remove(tmp_path)
            except: pass

    return jsonify({"error": "empty_body", "detail": "no audio payload"}), 400

@app.route("/api/stt", methods=["POST"])
def stt_api():
    return _handle_stt_request()

@app.route("/stt", methods=["POST"])
def stt_root():
    return _handle_stt_request()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)

## 추가
# ==== STT 패치: 바로 붙여넣기 ====
from typing import Optional
from fastapi import Request, UploadFile, File, HTTPException

import io

try:
    from pydub import AudioSegment
    USE_PYDUB = True
except Exception:
    USE_PYDUB = False

def _ensure_16k_mono(wav_bytes: bytes) -> bytes:
    # ffmpeg 미설치면 그냥 원본 리턴(에러 방지용)
    if not USE_PYDUB:
        return wav_bytes
    try:
        seg = AudioSegment.from_file(io.BytesIO(wav_bytes), format="wav")
        seg = seg.set_frame_rate(16000).set_channels(1)
        buf = io.BytesIO()
        seg.export(buf, format="wav")
        return buf.getvalue()
    except Exception:
        return wav_bytes

def _transcribe(wav_bytes: bytes) -> str:
    # TODO: 실제 STT 엔진 연결(Whisper 등)
    return "음성을 인식했습니다(데모)."

async def _handle_stt_common(
    file_part: Optional[UploadFile],
    audio_part: Optional[UploadFile],
    raw_wav: Optional[bytes],
):
    data: Optional[bytes] = None

    if file_part is not None:
        data = await file_part.read()
    elif audio_part is not None:
        data = await audio_part.read()
    elif raw_wav is not None:
        data = raw_wav

    if not data:
        raise HTTPException(status_code=400, detail="No audio data")

    fixed = _ensure_16k_mono(data)
    try:
        text = _transcribe(fixed)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"stt_failed: {e}")

@app.post("/api/stt")
async def stt_api(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
    audio: Optional[UploadFile] = File(default=None),
):
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("audio/"):
        raw = await request.body()
        return await _handle_stt_common(None, None, raw)
    return await _handle_stt_common(file, audio, None)

@app.post("/stt")
async def stt_alias(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
    audio: Optional[UploadFile] = File(default=None),
):
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("audio/"):
        raw = await request.body()
        return await _handle_stt_common(None, None, raw)
    return await _handle_stt_common(file, audio, None)
# ==== STT 패치 끝 ====

