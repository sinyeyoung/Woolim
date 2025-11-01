# app.py (Flask 전용, 복붙용)
import os
import io
import tempfile
from typing import Optional

from flask import Flask, request, jsonify
from flask_cors import CORS

# ---- ffmpeg 경로 준비 (pydub 사용 시 필요) ----
try:
    import imageio_ffmpeg
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = os.path.dirname(ffmpeg_path)
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass  # ffmpeg 없어도 no-op 변환으로 동작 가능

# ---- pydub(있으면 실제 16k/mono 변환, 없으면 no-op) ----
try:
    from pydub import AudioSegment
    USE_PYDUB = True
except Exception:
    USE_PYDUB = False

app = Flask(__name__)
CORS(app)

# 업로드 용량 제한
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB

# ---- 모델 설정 (환경변수로 조정 가능) ----
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")   # tiny | base | small ...
DEVICE        = os.environ.get("WHISPER_DEVICE", "cpu")   # cpu (Render 무료는 GPU 없음)
COMPUTE_TYPE  = os.environ.get("WHISPER_COMPUTE", "int8") # int8 권장(저메모리)
LANGUAGE      = os.environ.get("WHISPER_LANG", "ko")      # 한국어
BEAM_SIZE     = int(os.environ.get("WHISPER_BEAM", "1"))  # 1=greedy

_model = None  # lazy load

def get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(WHISPER_MODEL, device=DEVICE, compute_type=COMPUTE_TYPE)
    return _model

@app.route("/")
def home():
    return jsonify({"message": "서버 연결 성공!"})

@app.route("/health")
def health():
    return jsonify({"ok": True})

def _ensure_16k_mono_path(src_wav_path: str) -> str:
    """
    입력: 원본 WAV 경로
    출력: 16kHz/mono 보장된 WAV 임시파일 경로 (no-op이면 원본 경로 그대로 반환)
    - pydub(+ffmpeg) 가능하면 실제 변환
    - 불가하면 원본을 그대로 사용(Flutter가 이미 16k/mono면 충분)
    """
    if not USE_PYDUB:
        return src_wav_path  # no-op

    try:
        seg = AudioSegment.from_file(src_wav_path, format="wav")
        seg = seg.set_frame_rate(16000).set_channels(1)
        fd, out_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        seg.export(out_path, format="wav")
        return out_path
    except Exception:
        # 변환 실패 시 원본 경로 사용
        return src_wav_path

def _stt_from_file(tmp_path: str) -> str:
    """
    faster-whisper로 transcribe.
    필요 시 16k/mono로 변환된 경로를 실제로 사용.
    """
    model = get_model()

    # 16k/mono 보장
    norm_path = _ensure_16k_mono_path(tmp_path)
    cleanup_norm = (norm_path != tmp_path)

    try:
        segments, info = model.transcribe(
            norm_path,
            language=LANGUAGE,
            task="transcribe",
            beam_size=BEAM_SIZE,
            temperature=0.0,
            vad_filter=False,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        text = " ".join([seg.text.strip() for seg in segments]).strip()
        return text
    finally:
        # 변환 파일을 생성했으면 정리
        if cleanup_norm:
            try:
                os.remove(norm_path)
            except Exception:
                pass

def _handle_stt_request():
    # 1) multipart/form-data (field: file | audio)
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
