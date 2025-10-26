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

def _ensure_16k_mono(in_path: str) -> str:
    out_path = in_path + ".16k.wav"
    os.system(f'ffmpeg -y -loglevel error -i "{in_path}" -ac 1 -ar 16000 -c:a pcm_s16le "{out_path}"')
    return out_path

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

    # 임시 파일 정리
    try:
        os.remove(wav16)
    except Exception:
        pass
        
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
    
def _lite_cleanup(text: str) -> str:
    """가벼운 정리: 연속 공백/불필요 공백/기본 문장부호 정돈."""
    t = (text or "").strip()
    if not t:
        return t
    # 연속 공백 제거
    t = re.sub(r"\s+", " ", t)
    # 문장부호 앞 공백 제거
    t = re.sub(r"\s+([,.!?])", r"\1", t)
    # 문장부호 뒤 공백 보장
    t = re.sub(r"([,.!?])(?!\s|$)", r"\1 ", t)
    # 따옴표/괄호 주변 공백 정리
    t = re.sub(r"\(\s+", "(", t)
    t = re.sub(r"\s+\)", ")", t)
    t = re.sub(r"\s+”", "”", t)
    t = re.sub(r"“\s+", "“", t)
    # 끝에 마침표 없으면 가볍게 붙여주기(짧은 문장 시각적 구분용)
    if t and t[-1] not in ".!?…":
        t += "."
    return t

def _too_different(a: str, b: str, thresh: float = 0.55) -> bool:
    """과수정 방지: 전/후 유사도가 너무 떨어지면 교정 취소."""
    try:
        ratio = difflib.SequenceMatcher(a=a, b=b).ratio()
        return ratio < thresh
    except Exception:
        return False

@app.route("/api/correct", methods=["POST"])
def api_correct():
    data = request.get_json(silent=True) or {}
    original = (data.get("text") or "").strip()
    if not original:
        return jsonify({"error": "empty_text"}), 400
    corrected = original.strip()  # (지금은 간단 교정)
    return jsonify({
        "original_text": original,
        "corrected_text": corrected,
        "changed": corrected != original,
        "reason": "lite"
    })


