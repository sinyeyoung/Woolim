# app.py
from __future__ import annotations
from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import io
import os
import tempfile

# ───────────────────────── Flask ─────────────────────────
app = Flask(__name__)
CORS(app)  # 모바일/웹 호출 모두 안전하게 허용

@app.get("/health")
def health():
    return jsonify(ok=True), 200


# ───────────────────────── STT (faster-whisper) ─────────────────────────
# Render 저사양 환경 대비: 가장 가벼운 tiny/int8로 시작
# 필요 시 환경변수로 조절 가능 (WHISPER_MODEL, COMPUTE_TYPE)
from faster_whisper import WhisperModel
import soundfile as sf

MODEL_NAME = os.getenv("WHISPER_MODEL", "tiny")   # tiny / base 등
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")  # int8 / int8_float32 등

# ★ 앱 시작 시 전역 1회 로드(콜드스타트/메모리 안정)
model = WhisperModel(MODEL_NAME, device="cpu", compute_type=COMPUTE_TYPE)

@app.post("/api/stt")
def api_stt():
    """
    기대 형식:
      - multipart/form-data: field name 'file' (Content-Type: audio/wav)
      - 또는 raw audio/* 바디
    입력은 16kHz mono WAV(PCM) 권장
    """
    # 1) 오디오 바이트 수신
    wav_bytes: bytes | None = None
    if "file" in request.files:
        wav_bytes = request.files["file"].read()
    else:
        ctype = request.headers.get("Content-Type", "")
        if ctype.startswith("audio/"):
            wav_bytes = request.get_data()

    if not wav_bytes:
        return jsonify(error="no_audio", detail="audio file not found"), 400

    # 2) WAV 파싱 (가능하면 메모리상에서 처리)
    try:
        # dtype=float32로 바로 변환, 1D(np.float32) 배열 반환
        audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
    except Exception:
        # 드물게 헤더가 이상한 경우 임시파일 경유
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(wav_bytes)
                tmp_path = tmp.name
            audio, sr = sf.read(tmp_path, dtype="float32", always_2d=False)
        except Exception as e2:
            return jsonify(error="decode_failed", detail=str(e2)), 415
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    if sr != 16000:
        # 서버를 단순화: 리샘플 없이 16k만 허용(앱에서 16k로 녹음 중)
        return jsonify(error="sr_mismatch", expect=16000, got=sr), 415

    # 3) 추론(저사양 안전 옵션)
    try:
        segments, info = model.transcribe(
            audio,                       # np.ndarray(float32), mono
            language="ko",               # 한국어 고정
            beam_size=1,                 # 메모리/시간 최소
            vad_filter=True,             # 무음 구간 필터
        )
        text = "".join(s.text for s in segments).strip()
        return jsonify(text=text or ""), 200
    except Exception as e:
        # 클라이언트가 원인을 알 수 있게 상세 전달
        return jsonify(error="stt_failed", detail=str(e)), 500


# ───────────────────────── Correct API (문장 어미 보정) ─────────────────────────
@app.post("/api/correct")
def api_correct():
    data = request.get_json(silent=True) or {}
    text: str = (data.get("text") or "").strip()
    mode: str = (data.get("mode") or "ending").strip()
    style: str = (data.get("style") or "yo").strip()

    if not text:
        return jsonify(error="no text provided"), 400

    try:
        norm = _normalize(text)
        if mode == "ending":
            corrected = correct_ending(norm, style=style)
        elif mode == "formal":
            corrected = correct_ending(norm, style="formal")
        else:
            corrected = norm
        corrected = _post_normalize(corrected)
        return jsonify(original=text, corrected=corrected), 200
    except Exception as e:
        return jsonify(error="internal_error", message=str(e)), 500


# ───────────────────────── Core Logic (어미 보정) ─────────────────────────
def correct_ending(s: str, style: str = "yo") -> str:
    parts = _split_keep_delim(s)
    fixed_parts = []
    for seg, delim in parts:
        seg_strip = seg.strip()
        if not seg_strip:
            fixed_parts.append(seg + delim)
            continue
        seg_strip = _micro_fixes(seg_strip)
        if style == "yo":
            seg_strip = _to_haeyo(seg_strip)
        elif style == "hae":
            seg_strip = _to_hae(seg_strip)
        elif style == "formal":
            seg_strip = _to_formal(seg_strip)
        if delim == "":
            delim = "."
        fixed_parts.append(seg_strip + delim)
    result = "".join(fixed_parts)
    result = re.sub(r"\s+([,.?!])", r"\1", result)
    result = re.sub(r"\s{2,}", " ", result)
    return result.strip()

def _to_haeyo(seg: str) -> str:
    repl = [
        (r"했어\b", "했어요"), (r"했구나\b", "했군요"), (r"했네\b", "했네요"),
        (r"한다\b", "해요"), (r"한다면\b", "하면요"), (r"한다니까\b", "한다니까요"),
        (r"한다니\b", "한다니요"), (r"한다며\b", "한다면서요"), (r"했지\b", "했죠"),
        (r"해\b", "해요"), (r"자\b", "가요"), (r"야\b", "예요"),
        (r"이야\b", None), (r"거야\b", "거예요"), (r"거지\b", "거죠"),
        (r"거네\b", "거리네요"), (r"거든\b", "거든요"),
        (r"자야돼\b", "자야 돼요"), (r"돼\b", "돼요"), (r"돼야\b", "돼야 해요"),
        (r"했단\b", "했다는"), (r"했음\b", "했습니다"),
    ]
    seg = _apply_pairs(seg, repl)
    seg = re.sub(r"([가-힣])이야\b", lambda m: _i_yeyo(m.group(1)), seg)
    return _ensure_polite(seg)

def _to_hae(seg: str) -> str:
    repl = [
        (r"했어요\b", "했어"), (r"합니다\b", "해"), (r"합니다만\b", "하지만"),
        (r"합니까\b", "해?"), (r"해요\b", "해"), (r"이에요\b", "이야"),
        (r"예요\b", "야"), (r"거예요\b", "거야"), (r"거죠\b", "거지"),
        (r"됩니다\b", "돼"), (r"돼요\b", "돼"),
    ]
    return _apply_pairs(seg, repl)

def _to_formal(seg: str) -> str:
    repl = [
        (r"했어요\b", "했습니다"), (r"했어\b", "했습니다"),
        (r"한다\b", "합니다"), (r"해요\b", "합니다"), (r"해\b", "합니다"),
        (r"이에요\b", "입니다"), (r"예요\b", "입니다"),
        (r"거예요\b", "것입니다"), (r"거야\b", "것입니다"),
        (r"돼요\b", "됩니다"), (r"돼\b", "됩니다"),
    ]
    seg = _apply_pairs(seg, repl)
    return re.sub(r"(이다)\b", "입니다", seg)

def _normalize(s: str) -> str:
    s = s.replace("\u200b", "")
    s = re.sub(r"[ ]{2,}", " ", s)
    s = re.sub(r"([.?!]){2,}", r"\1", s)
    return s.strip()

def _post_normalize(s: str) -> str:
    s = re.sub(r"\s+([,.?!])", r"\1", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()

def _micro_fixes(seg: str) -> str:
    seg = re.sub(r"되요\b", "돼요", seg)
    seg = re.sub(r"돼\s?야\b", "돼야", seg)
    seg = seg.replace("자야돼", "자야 돼")
    seg = re.sub(r"\b것 이\b", "것이", seg)
    seg = re.sub(r"\b거 야\b", "거야", seg)
    return seg

def _apply_pairs(seg: str, pairs: list[tuple[str, str | None]]) -> str:
    for pat, rep in pairs:
        if rep is None:
            continue
        seg = re.sub(pat, rep)
    return seg

def _i_yeyo(last_char: str) -> str:
    code = ord(last_char)
    jong = (code - 0xAC00) % 28
    return last_char + ("이에요" if jong != 0 else "예요")

def _ensure_polite(seg: str) -> str:
    if re.search(r"[가-힣]$", seg) and not re.search(r"(요|다|함|임|니다|해|해요|다며|다니)$", seg):
        ch = seg[-1]
        seg = re.sub(r"[가-힣]$", _i_yeyo(ch), seg)
    return seg

def _split_keep_delim(s: str):
    tokens = re.split(r"([.?!])", s)
    out = []
    for i in range(0, len(tokens), 2):
        seg = tokens[i]
        delim = tokens[i + 1] if i + 1 < len(tokens) else ""
        out.append((seg, delim))
    return out


# ───────────── 앱 기동 (로컬 실행용) ─────────────
if __name__ == "__main__":
    # gunicorn 환경에선 무시됨. 로컬 테스트용.
    app.run(host="0.0.0.0", port=5000, debug=False)
