from __future__ import annotations

from flask import Flask, request, jsonify
from flask_cors import CORS
import io, os, re, tempfile, threading
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

# ───────────────── Flask 기본 ─────────────────
app = Flask(__name__)
CORS(app)

@app.get("/health")
def health():
    return jsonify(ok=True), 200

# ───────────────── Whisper Model ─────────────────
MODEL_NAME   = os.getenv("WHISPER_MODEL", "tiny")   # 필요하면 small / medium 등으로
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")    # Render 무료면 int8 권장

# CPU 기준
model = WhisperModel(MODEL_NAME, device="cpu", compute_type=COMPUTE_TYPE)

_ASR_LOCK   = threading.Lock()
_WARMED     = False
_WARM_LOCK  = threading.Lock()

def _to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        return x.mean(axis=1).astype("float32", copy=False)
    return x.astype("float32", copy=False)

def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x))))

def _do_warm() -> tuple[bool, str | None]:
    """모델 웜업(무음 0.5초 한 번 돌리기)"""
    global _WARMED
    if _WARMED:
        return True, None
    with _WARM_LOCK:
        if _WARMED:
            return True, None
        try:
            dummy = np.zeros(8000, dtype="float32")  # 0.5초 @16k
            with _ASR_LOCK:
                _ = model.transcribe(dummy, language="ko", beam_size=1, vad_filter=True)
            _WARMED = True
            app.logger.info("[WARM] model warmed")
            return True, None
        except Exception as e:
            app.logger.exception("[WARM] failed")
            return False, str(e)

@app.get("/warm")
def warm():
    ok, err = _do_warm()
    if ok:
        return jsonify(ok=True), 200
    return jsonify(ok=False, error=err), 500

# ───────────────── 서버 시작 시 1회 자동 웜업 ─────────────────
def _auto_warm_once_bg():
    try:
        ok, err = _do_warm()
        if ok:
            app.logger.info("[AUTO_WARM] done")
        else:
            app.logger.warning(f"[AUTO_WARM] failed: {err}")
    except Exception:
        app.logger.exception("[AUTO_WARM] unexpected error")

threading.Thread(target=_auto_warm_once_bg, daemon=True).start()

# ───────────────── STT API ─────────────────
@app.post("/api/stt")
def api_stt():
    """
    Flutter 쪽에서:
      - URL: https://woolim.onrender.com/api/stt
      - method: POST
      - headers: Content-Type: application/octet-stream
      - body: 16kHz mono WAV bytes
    """

    # 1) 입력 받기
    wav_bytes: bytes | None = None

    # (1) multipart/form-data 로 file 필드가 온 경우
    if "file" in request.files:
        f = request.files["file"]
        wav_bytes = f.read()
        app.logger.info(
            f"[STT] multipart upload: name={getattr(f, 'filename', '')}, "
            f"ctype={getattr(f, 'content_type', '')}, size={len(wav_bytes)}"
        )
    else:
        # (2) 그 외에는 raw body 그대로 사용 (Content-Type 상관 없음)
        wav_bytes = request.get_data()
        app.logger.info(
            f"[STT] raw body: ctype={request.headers.get('Content-Type','')}, "
            f"size={len(wav_bytes)}"
        )

    if not wav_bytes:
        return jsonify(error="no_audio", detail="audio data not found"), 400

    # 2) 디코드 (메모리 → 실패 시 임시파일)
    try:
        audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
    except Exception:
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
        return jsonify(error="sr_mismatch", expect=16000, got=sr), 415

    audio = _to_mono(np.array(audio, dtype="float32", copy=False))
    dur_s  = len(audio) / 16000.0
    energy = _rms(audio)

    if dur_s < 0.5 or energy < 1e-4:
        return jsonify(
            text="",
            meta={"duration": round(dur_s, 3), "rms": energy, "note": "too_short_or_silent"},
        ), 200

    # 3) Whisper 추론
    try:
        with _ASR_LOCK:
            segments, info = model.transcribe(
                audio,
                language="ko",
                beam_size=1,
                vad_filter=True,
            )
        text = "".join(s.text for s in segments).strip()
        return jsonify(
            text=text or "",
            meta={
                "duration": round(dur_s, 3),
                "rms": energy,
                "lang": getattr(info, "language", "ko"),
            },
        ), 200
    except Exception as e:
        app.logger.exception("[STT] transcribe failed")
        return jsonify(error="stt_failed", detail=str(e)), 500

# ───────────────── Correct API ─────────────────
@app.post("/api/correct")
def api_correct():
    data  = request.get_json(silent=True) or {}
    text  = (data.get("text")  or "").strip()
    mode  = (data.get("mode")  or "ending").strip()
    style = (data.get("style") or "yo").strip()

    if not text:
        return jsonify(error="no text provided"), 400

    try:
        # 1) 정규화 + 어미 보정
        norm = _normalize(text)
        if mode in ("ending", "formal", "hae"):
            if mode == "formal":
                style = "formal"
            elif mode == "hae":
                style = "hae"
            corrected = correct_ending(norm, style=style)
        else:
            corrected = norm
        corrected = _post_normalize(corrected)

        # 2) 디버그 로그
        print("===== /api/correct called =====", flush=True)
        print("original(text):", text, flush=True)
        print("corrected:", corrected, flush=True)

        # 3) 응답
        return jsonify(original=text, corrected=corrected), 200
    except Exception as e:
        print("[/api/correct] error:", str(e), flush=True)
        return jsonify(error="internal_error", message=str(e)), 500


# ───────────── 어미 보정 로직들 ─────────────

def correct_ending(s: str, style: str = "yo") -> str:
    parts = _split_keep_delim(s)
    fixed = []
    for seg, delim in parts:
        seg_strip = seg.strip()
        if not seg_strip:
            fixed.append(seg + delim)
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
        fixed.append(seg_strip + delim)
    result = "".join(fixed)
    result = re.sub(r"\s+([.?!])", r"\1", result)
    result = re.sub(r"\s{2,}", " ", result)
    return result.strip()

def _to_haeyo(seg: str) -> str:
    repl = [
        (r"했어\b", "했어요"), (r"했구나\b", "했군요"), (r"했네\b", "했네요"),
        (r"한다\b", "해요"), (r"한다면\b", "하면요"), (r"한다니까\b", "한다니까요"),
        (r"한다니\b", "한다니요"), (r"한다며\b", "한다면서요"), (r"했지\b", "했죠"),
        (r"해\b", "해요"), (r"줘\b", "줘요"), (r"자\b", "가요"),   # ★ 여기 '줘\b' 추가
        (r"자야돼\b", "자야 돼요"), (r"돼\b", "돼요"), (r"돼야\b", "돼야 해요"),
        (r"했단\b", "했다는"), (r"했음\b", "했습니다"),
    ]
    seg = _apply_pairs(seg, repl)
    seg = re.sub(r"([가-힣])이야\b", lambda m: _i_yeyo(m.group(1)), seg)
    return _ensure_polite(seg)

def _to_hae(seg: str) -> str:
    repl = [
        (r"했어요\b", "했어"), (r"합니다\b", "해"), (r"합니다만\b", "하지만"),
        (r"합니까\b", "해?"), (r"해요\b", "해"), (r"거예요\b", "거야"), (r"거죠\b", "거지"),
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
    # 콤마 제거 (교정 문장에서는 콤마를 아예 안 쓰기)
    s = s.replace(",", " ")
    s = re.sub(r"\s+([.?!])", r"\1", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()

def _micro_fixes(seg: str) -> str:
    # 기본 맞춤법 보정
    seg = re.sub(r"되요\b", "돼요", seg)
    seg = re.sub(r"돼\s?야\b", "돼야", seg)
    seg = seg.replace("자야돼", "자야 돼")
    seg = re.sub(r"\b것 이\b", "것이", seg)
    seg = re.sub(r"\b거 야\b", "거야", seg)

    # 단어 사전 기반 교정 (에너아→엔화, 환유/한유→환율 등)
    for wrong, right in COMMON_WORD_FIXES.items():
        if wrong in seg:
            seg = seg.replace(wrong, right)

    return seg

def _apply_pairs(seg: str, pairs: list[tuple[str, str | None]]) -> str:
    for pat, rep in pairs:
        if rep is None:
            continue
        seg = re.sub(pat, rep, seg)  # 반드시 세 번째 인자로 seg 사용
    return seg

def _i_yeyo(last_char: str) -> str:
    code = ord(last_char)
    jong = (code - 0xAC00) % 28
    return last_char + ("이에요" if jong != 0 else "예요")

def _ensure_polite(seg: str) -> str:
    # 한글로 끝나지 않으면 손대지 않음
    if not re.search(r"[가-힣]$", seg):
        return seg

    # 이미 공손한 끝말들(요, 해요, 입니다, 했어요, 했죠, 죠 등)은 그대로 둔다
    if re.search(r"(요|죠|다|함|임|니다|해요|했어요|했죠|다며|다니)$", seg):
        return seg

    # 그 외에 정말 맨몸 명사/동사로 끝날 때만 '예요/이에요'를 붙인다
    ch = seg[-1]
    return re.sub(r"[가-힣]$", _i_yeyo(ch), seg)


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
