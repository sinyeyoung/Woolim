# app.py
from __future__ import annotations
from flask import Flask, request, jsonify
import re

app = Flask(__name__)

# ───────────────────────── Health ─────────────────────────
@app.get("/health")
def health():
    return jsonify(ok=True), 200

# ───────────────────────── Correct API ─────────────────────────
# 요청 예: { "text":"나는 오늘 너무 피곤하다 그래서 일찍 자야돼", "mode":"ending", "style":"yo" }
@app.post("/api/correct")
def api_correct():
    data = request.get_json(silent=True) or {}
    text: str = (data.get("text") or "").strip()
    mode: str = (data.get("mode") or "ending").strip()
    style: str = (data.get("style") or "yo").strip()

    if not text:
        return jsonify(error="no text provided"), 400

    try:
        # 1) 사전 정리(공백/연속 구두점 등)
        norm = _normalize(text)

        # 2) 모드 분기 (현재는 ending 위주)
        if mode == "ending":
            corrected = correct_ending(norm, style=style)
        elif mode == "formal":
            corrected = correct_ending(norm, style="formal")
        else:
            # 미지원 모드는 보수적으로 원문 반환
            corrected = norm

        # 3) 후처리(연속 공백 제거 등)
        corrected = _post_normalize(corrected)

        return jsonify(original=text, corrected=corrected), 200
    except Exception as e:
        # 실패 시에도 클라 UX를 위해 200대 + 원문 반환보다는
        # 500을 주고 message 키도 포함(클라 파서가 여러 키를 읽음)
        return jsonify(error="internal_error", message=str(e)), 500


# ───────────────────────── Core Logic ─────────────────────────
def correct_ending(s: str, style: str = "yo") -> str:
    """
    어미·말끝 자연화 중심 보정.
    style: 'yo'(해요체), 'hae'(해체), 'formal'(격식체)
    """
    # 문장 분할(문장부호 보존)
    parts = _split_keep_delim(s)

    fixed_parts = []
    for seg, delim in parts:
        seg_strip = seg.strip()
        if not seg_strip:
            fixed_parts.append(seg + delim)
            continue

        # 1) 흔한 구어체·맞춤형 교정(경어/반말/격식 변환 전에 적용)
        seg_strip = _micro_fixes(seg_strip)

        # 2) 스타일 변환
        if style == "yo":
            seg_strip = _to_haeyo(seg_strip)
        elif style == "hae":
            seg_strip = _to_hae(seg_strip)
        elif style == "formal":
            seg_strip = _to_formal(seg_strip)

        # 3) 마침표 없으면 적절히 보완(?,! 유지)
        if delim == "":
            delim = "."

        fixed_parts.append(seg_strip + delim)

    result = "".join(fixed_parts)
    # 쉼표 뒤 공백 등 마무리
    result = re.sub(r"\s+([,.?!])", r"\1", result)
    result = re.sub(r"\s{2,}", " ", result)
    return result.strip()


# ───────────── 스타일 변환 (간단 휴리스틱) ─────────────
def _to_haeyo(seg: str) -> str:
    # 반말/서술체 → 해요체
    repl = [
        (r"했어\b", "했어요"),
        (r"했구나\b", "했군요"),
        (r"했네\b", "했네요"),
        (r"한다\b", "해요"),
        (r"한다면\b", "하면요"),
        (r"한다니까\b", "한다니까요"),
        (r"한다니\b", "한다니요"),
        (r"한다며\b", "한다면서요"),
        (r"했지\b", "했죠"),
        (r"해\b", "해요"),
        (r"자\b", "가요"),           # 가자 → 가요 (권유 문장 단순화)
        (r"야\b", "예요"),          # ~야 → ~예요
        (r"이야\b", None),          # 이야 → 이에요/예요 (받침 규칙)
        (r"거야\b", "거예요"),
        (r"거지\b", "거죠"),
        (r"거네\b", "거리네요"),
        (r"거든\b", "거든요"),
        (r"자야돼\b", "자야 돼요"),
        (r"돼\b", "돼요"),
        (r"돼야\b", "돼야 해요"),
        (r"했단\b", "했다는"),
        (r"했음\b", "했습니다"),
    ]
    seg = _apply_pairs(seg, repl)

    # '이야' 규칙 처리(받침에 따라 이에요/예요)
    seg = re.sub(r"([가-힣])이야\b", lambda m: _i_yeyo(m.group(1)), seg)

    # 종결 미부호 시 마침 어조 정리
    seg = _ensure_polite(seg)
    return seg


def _to_hae(seg: str) -> str:
    # 해요체/격식 → 반말(해체)
    repl = [
        (r"했어요\b", "했어"),
        (r"합니다\b", "해"),
        (r"합니다만\b", "하지만"),
        (r"합니까\b", "해?"),
        (r"해요\b", "해"),
        (r"이에요\b", "이야"),
        (r"예요\b", "야"),
        (r"거예요\b", "거야"),
        (r"거죠\b", "거지"),
        (r"됩니다\b", "돼"),
        (r"돼요\b", "돼"),
    ]
    return _apply_pairs(seg, repl)


def _to_formal(seg: str) -> str:
    # 해요체/반말 → 격식체
    repl = [
        (r"했어요\b", "했습니다"),
        (r"했어\b", "했습니다"),
        (r"한다\b", "합니다"),
        (r"해요\b", "합니다"),
        (r"해\b", "합니다"),
        (r"이에요\b", "입니다"),
        (r"예요\b", "입니다"),
        (r"거예요\b", "것입니다"),
        (r"거야\b", "것입니다"),
        (r"돼요\b", "됩니다"),
        (r"돼\b", "됩니다"),
    ]
    seg = _apply_pairs(seg, repl)

    # 문장 끝 조사·서술어 보정
    seg = re.sub(r"(이다)\b", "입니다", seg)
    return seg


# ───────────── 사전/후처리 ─────────────
def _normalize(s: str) -> str:
    s = s.replace("\u200b", "")  # zero-width space
    s = re.sub(r"[ ]{2,}", " ", s)
    s = re.sub(r"([.?!]){2,}", r"\1", s)  # 연속 구두점 축소
    return s.strip()

def _post_normalize(s: str) -> str:
    s = re.sub(r"\s+([,.?!])", r"\1", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()


def _micro_fixes(seg: str) -> str:
    """
    어미 변환 전에 적용할 가벼운 정리:
    - '그래서'가 두 절을 어색하게 잇는 경우는 유지(과도한 변형 방지)
    - 자주 틀리는 띄어쓰기
    """
    # 자주 틀리는 띄어쓰기(돼/되)
    seg = re.sub(r"되요\b", "돼요", seg)  # 흔한 오탈자
    seg = re.sub(r"돼\s?야\b", "돼야", seg)

    # '자야돼' → '자야 돼'
    seg = seg.replace("자야돼", "자야 돼")

    # 연결 조사/어미 간략 보정
    seg = re.sub(r"\b것 이\b", "것이", seg)
    seg = re.sub(r"\b거 야\b", "거야", seg)

    return seg


# ───────────── 유틸 ─────────────
def _apply_pairs(seg: str, pairs: list[tuple[str, str | None]]) -> str:
    for pat, rep in pairs:
        if rep is None:
            continue
        seg = re.sub(pat, rep)
    return seg

def _i_yeyo(last_char: str) -> str:
    # 받침 유무로 '이에요/예요' 결정
    code = ord(last_char)
    jong = (code - 0xAC00) % 28
    return last_char + ("이에요" if jong != 0 else "예요")

def _ensure_polite(seg: str) -> str:
    """
    어미가 너무 짧거나 반말 느낌이면 해요체로 부드럽게.
    """
    # 끝이 명사/형용사 추정 + 종결 없는 경우 → '이에요/예요' 보정
    if re.search(r"[가-힣]$", seg) and not re.search(r"(요|다|함|임|니다|해|해요|다며|다니)$", seg):
        ch = seg[-1]
        seg = re.sub(r"[가-힣]$", _i_yeyo(ch), seg)
    return seg

def _split_keep_delim(s: str):
    """
    문장을 (본문, 구두점) 튜플로 쪼개되 구두점을 보존.
    예: "안녕? 오늘 어때" -> [("안녕","?"), (" 오늘 어때","")]
    """
    tokens = re.split(r"([.?!])", s)
    out = []
    for i in range(0, len(tokens), 2):
        seg = tokens[i]
        delim = tokens[i + 1] if i + 1 < len(tokens) else ""
        out.append((seg, delim))
    return out


# ───────────── 앱 기동 ─────────────
if __name__ == "__main__":
    # 개발용: python app.py
    # 운영(예: Render/Gunicorn)은 WSGI로 구동
    app.run(host="0.0.0.0", port=5000, debug=False)
