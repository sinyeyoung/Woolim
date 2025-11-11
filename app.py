# app.py
from __future__ import annotations
from flask import Flask, request, jsonify
from flask_cors import CORS
import io, os, re, tempfile
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

app = Flask(__name__)
CORS(app)

@app.get("/health")
def health():
    return jsonify(ok=True), 200

# ───────────────── Whisper Model ─────────────────
MODEL_NAME   = os.getenv("WHISPER_MODEL", "tiny")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")
model = WhisperModel(MODEL_NAME, device="cpu", compute_type=COMPUTE_TYPE)

def _to_mono(x: np.ndarray) -> np.ndarray:
    # x: float32, shape (n,) or (n, ch)
    if x.ndim == 2:  # (n, ch)
        return x.mean(axis=1).astype("float32", copy=False)
    return x.astype("float32", copy=False)

@app.post("/api/stt")
def api_stt():
    """
    기대 형식:
      - multipart/form-data: field 'file' (Content-Type: audio/wav)
      - 또는 raw body with Content-Type: audio/*
    입력 권장: 16kHz mono PCM WAV
    """
    # 1) 입력 수신
    wav_bytes: bytes | None = None
    if "file" in request.files:
        f = request.files["file"]
        wav_bytes = f.read()
        app.logger.info(f"[STT] file upload: name={getattr(f, 'filename', '')}, ctype={getattr(f, 'content_type', '')}, size={len(wav_bytes)}")
    else:
        ctype = request.headers.get("Content-Type", "")
        if ctype.startswith("audio/"):
            wav_bytes = request.get_data()
            app.logger.info(f"[STT] raw audio body: ctype={ctype}, size={len(wav_bytes)}")

    if not wav_bytes:
        return jsonify(error="no_audio", detail="audio file not found"), 400

    # 2) 디코드
    try:
        audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
    except Exception as e1:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(wav_bytes)
                tmp_path = tmp.name
            audio, sr = sf.read(tmp_path, dtype="float32", always_2d=False)
        except Exception as e2:
            return jsonify(error="decode_failed", detail=str(e2)), 415
        finally:
            try: os.remove(tmp_path)
            except Exception: pass

    # 3) 샘플레이트/채널 확인
    if sr != 16000:
        return jsonify(error="sr_mismatch", expect=16000, got=sr), 415
    audio = _to_mono(np.array(audio, dtype="float32", copy=False))

    # 4) 추론
    try:
        segments, info = model.transcribe(
            audio,
            language="ko",
            beam_size=1,
            vad_filter=True,
        )
        text = "".join(s.text for s in segments).strip()
        return jsonify(text=text or ""), 200
    except Exception as e:
        app.logger.exception("[STT] transcribe failed")
        return jsonify(error="stt_failed", detail=str(e)), 500

# ───────────────── Correct API ─────────────────
@app.post("/api/correct")
def api_correct():
    data = request.get_json(silent=True) or {}
    text  = (data.get("text")  or "").strip()
    mode  = (data.get("mode")  or "ending").strip()
    style = (data.get("style") or "yo").strip()
    if not text:
        return jsonify(error="no text provided"), 400
    try:
        norm = _normalize(text)
        if mode in ("ending", "formal", "hae"):
            if mode == "formal": style = "formal"
            elif mode == "hae":  style = "hae"
            corrected = correct_ending(norm, style=style)
        else:
            corrected = norm
        corrected = _post_normalize(corrected)
        return jsonify(original=text, corrected=corrected), 200
    except Exception as e:
        return jsonify(error="internal_error", message=str(e)), 500

# ───────────── Core (어미 보정) ─────────────
def correct_ending(s: str, style: str = "yo") -> str:
    parts = _split_keep_delim(s)
    fixed = []
    for seg, delim in parts:
        seg_strip = seg.strip()
        if not seg_strip:
            fixed.append(seg + delim); continue
        seg_strip = _micro_fixes(seg_strip)
        if style == "yo":     seg_strip = _to_haeyo(seg_strip)
        elif style == "hae":  seg_strip = _to_hae(seg_strip)
        elif style == "formal": seg_strip = _to_formal(seg_strip)
        if delim == "": delim = "."
        fixed.append(seg_strip + delim)
    result = "".join(fixed)
    result = re.sub(r"\s+([,.?!])", r"\1", result)
    result = re.sub(r"\s{2,}", " ", result)
    return result.strip()

def _to_haeyo(seg: str) -> str:
    repl = [
        (r"했어\b", "했어요"), (r"했구나\b", "했군요"), (r"했네\b", "했네요"),
        (r"한다\b", "해요"), (r"한다면\b", "하면요"), (r"한다니까\b", "한다니까요"),
        (r"한다니\b", "한다니요"), (r"한다며\b", "한다면서요"), (r"했지\b", "했죠"),
        (r"해\b", "해요"), (r"자\b", "가요"), (r"야\b", "예요"),
        (r"자야돼\b", "자야 돼요"), (r"돼\b", "돼요"), (r"돼야\b", "돼야 해요"),
        (r"했단\b", "했다는"), (r"했음\b", "했습니다"),
    ]
    seg = _apply_pairs(seg, repl)
    seg = re.sub(r"([가-힣])이야\b", lambda m: _i_yeyo(m.group(1)), seg)
    return _ensure_polite(seg)

def _to_hae(seg: str) -> str:
    repl = [
        (r"했어요\b", "했어"), (r"합니다\b", "해"), (r"합니다만\b", "하지만"),
        (r"합니까\b", "해\?"), (r"해요\b", "해"), (r"이에요\b", "이야"),
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
        if rep is None: continue
        seg = re.sub(pat, rep)
    return seg

def _i_yeyo(last_char: str) -> str:
    code = ord(last_char); jong = (code - 0xAC00) % 28
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
