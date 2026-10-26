"""선생 모델에게 주는 글(파이보 말 바꿔 쓰기, 데이터 검사)과 받은 글의 검사.

원본 대화(data/source.jsonl, PC에만 둠)의 학생 말은 원본 그대로 두고, 상대의 말만 선생이
파이보 말투로 바꿔 씁니다(REWRITE: 선생은 파이보 말만 씀). 같은 선생이 다시 읽고 점수가 모자라거나 문제가 있으면
뺍니다(CHECK).

학습 대화 하나는 OpenAI 형식의 메시지 목록입니다::

    [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}, ...]
"""

from __future__ import annotations

import re

from teacher import first_json
from tools import SYSTEM

#: 파이보 말 하나의 최대 글자 수 (규칙은 80자 안쪽, 조금 넘는 것은 둡니다)
MAX_PIBO = 120

PIBO_STYLE = """파이보는 초·중·고 학생과 이야기하는 교육용 로봇이에요. 열네 살 친구처럼 밝고 다정하게,
친근한 존댓말("~해요", "~예요")로 말해요.
- 한두 문장, 80자 안쪽. 소리로 읽어 주는 말이라 이모지, 기호, 괄호, 목록, "ㅋㅋ" 같은 글자는 쓰지 않아요.
- 반말을 섞지 않아요. 학생을 "너", "네가", "니가", "너의"로 부르지 않고 (주어를 빼고 말해요),
  "~구나", "~지?", "~거든", "~잖아", "~했어", "~할게" 같은 반말 끝 대신 "~군요", "~네요", "~죠?",
  "~거든요", "~했어요", "~할게요"로 말해요. 자기는 "저", "제가"라고 해요.
- 학생 말을 그대로 되받아 읊지 않아요 ("~해서 ~하셨군요" 같은 상담사 말투로 끝내지 않기).
  짧게 마음을 알아주고, 그 이야기에 맞는 구체적인 생각, 정보, 질문을 더해요. 질문은 한 번에 하나만,
  매번 하지는 않아요. 같은 칭찬이나 같은 맺음말을 되풀이하지 않아요.
- 파이보는 로봇이라 먹기, 잠, 학교, 가족, 여행, 직업 같은 사람의 경험이 없어요. 지어내지 말고,
  로봇답게 좋아하는 것이나 생각을 말하거나 학생에게 물어요.
- 널리 알려진 사실은 쉽게 말하고, 확실하지 않은 숫자는 말하지 않아요.
- 괴롭힘, 폭력, 다침, 마음이 많이 힘든 이야기는 마음을 알아준 뒤 부모님이나 선생님 같은
  믿을 만한 어른에게 꼭 알리라고 말해요.
- 이름, 주소, 학교 같은 개인정보는 묻지 않아요. 바르고 고운 말만 써요."""

REWRITE = """아래 대화에서 A의 말은 학생의 말로 그대로 두고, B의 말만 교육용 로봇 '파이보'의 말로
바꿔 학습 대화를 만들어요.

{style}

할 일:
1. 초·중·고 학생이 로봇 친구와 나눌 만한 대화인지 먼저 봐요. 직장, 결혼, 연애, 술, 돈벌이 같은 어른
   이야기가 중심이거나, A가 직장인이나 부모처럼 말하거나, 욕설, 비하, 혐오, 성적인 말, 저급한 말이
   있으면 {{"ok": false, "why": "까닭 짧게"}}만 써요.
2. 나눌 만하면 A의 말마다 그 뒤에 할 파이보의 말을 차례대로 써요. A의 말은 쓰지 않고 고치지도 않아요.
   - B가 한 말의 뜻과 흐름(공감, 질문, 의견, 정보)을 살려 파이보 말투로 바꿔요. 다음 A의 말이
     자연스럽게 이어지도록 B의 질문은 비슷한 질문으로 남겨요. B가 사람의 경험을 말했으면 로봇에
     맞게 바꿔요. B의 말이 없으면 새로 써요.
   - 뒤로 가며 어른 이야기로 흐르거나 이어지지 않으면 거기서 끊고 앞의 것만 써요.
JSON 하나로만 답해요.
출력 형식: {{"ok": true, "pibo": ["A의 첫 말 뒤에 할 말", "A의 둘째 말 뒤에 할 말"]}}

대화 (주제: {topic}):
{dialog}"""

CHECK = """교육용 로봇 '파이보'와 학생의 대화 학습 데이터를 검사해요. 작은 모델이 이 대화를 그대로
배우므로 조금이라도 어색하면 낮게 줘요.

{style}

대화:
{dialog}

대화 전체를 보고 JSON 하나로만 답해요.
- natural: 파이보의 한국어가 자연스럽고 존댓말을 지켰는가 (1~5)
- fit: 파이보의 말이 학생 말뜻과 앞뒤 흐름에 맞고 구체적인가 (1~5)
- friend: 열네 살 친구처럼 다정하고, 학생 말을 되받아 읊거나 같은 말투를 되풀이하지 않는가 (1~5)
- kid: 학생과 로봇이 나눌 만한 바르고 고운 대화인가 (어른 이야기, 욕설, 비하, 저급한 말이 있으면 false)
- robot: 파이보가 사람만 하는 경험(먹었다, 잤다, 학교에 갔다, 가족이 있다 등)을 지어냈으면 true
- wrong: 틀린 사실이 있으면 true
- unsafe: 위험하거나 괴롭힘·폭력 이야기인데 믿을 만한 어른에게 알리라는 말이 없거나, 해로운 말이 있으면 true
출력 형식: {{"natural": 0, "fit": 0, "friend": 0, "kid": true, "robot": false, "wrong": false, "unsafe": false}}"""

#: 검사에서 남기는 기준
PASS = 4


def source_dialog(src: dict) -> str:
    lines = []
    for t in src["turns"]:
        lines.append(f"A: {t['user']}")
        lines.append(f"B: {t['other']}" if t["other"] else "B: (말 없음)")
    return "\n".join(lines)


def rewrite_prompt(src: dict) -> str:
    return REWRITE.format(style=PIBO_STYLE, topic=src.get("topic") or "일상", dialog=source_dialog(src))


def dialog_text(messages: list[dict]) -> str:
    return "\n".join(("학생: " if m["role"] == "user" else "파이보: ") + m["content"]
                     for m in messages if m["role"] in ("user", "assistant"))


def check_prompt(messages: list[dict]) -> str:
    return CHECK.format(style=PIBO_STYLE, dialog=dialog_text(messages))


def to_messages(turns: list[dict]) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM}]
    for t in turns:
        msgs += [{"role": "user", "content": t["user"]}, {"role": "assistant", "content": t["pibo"]}]
    return msgs


# ------------------------------------------------------------- 글자만 보는 검사

#: 소리로 읽기 어려운 기호와 그림 글자
BAD_CHARS = re.compile(r"[\U0001F300-\U0001FAFF☀-➿*#`|<>\[\]{}~^_=]")
#: 한글, 영어 글자·숫자, 문장부호 말고 다른 글자 (태국 문자, 한자, 일본 글자, 자음·모음만 …)
OTHER_SCRIPT = re.compile(r"[^\x20-\x7E\s가-힣…·“”‘’]")
#: 파이보 말에 남은 반말 ("너의 미래", "재미있지?", "힘들었겠네.")
BANMAL = re.compile(r"(^|[\s,.!?])(너는|너도|너가|너를|너의|너랑|너처럼|너한테|넌|니가|네가|내가)"
                    r"(?=\s|$|[,.!?])"
                    r"|(겠네|했네|거든|구나|잖아|했어|할게|줄게|볼까|있지|좋지|맞지|그치|해봐|해 봐)[.!?]?$"
                    r"|(겠네|했네|거든|구나|잖아)[.!?]\s")


def pibo_problem(text, limit: int = MAX_PIBO) -> str | None:
    """파이보 말이 규칙에 어긋난 까닭 (괜찮으면 None)."""
    if not isinstance(text, str) or not text.strip():
        return "파이보 말 없음"
    text = text.strip()
    if len(text) > limit:
        return f"파이보 말이 {limit}자 넘음"
    if bad := BAD_CHARS.search(text):
        return f"파이보 말에 기호 {bad.group()!r}"
    if bad := OTHER_SCRIPT.search(text):
        return f"파이보 말에 다른 글자 {bad.group()!r}"
    if BANMAL.search(text):
        return "파이보 말에 반말"
    if re.match(r"\s*(파이보|pibo)\s*:", text, re.IGNORECASE):
        return "파이보 말에 '파이보:'"
    return None


def same(text: str) -> str:
    """견줄 모양: 띄어쓰기와 문장부호를 뺌."""
    return re.sub(r"[\s.,!?~…'\"]", "", text)


def grams(text: str) -> set[str]:
    t = same(text)
    return {t[i:i + 3] for i in range(len(t) - 2)}


def overlap(a: str, b: str) -> float:
    """a의 글자 덩어리(3글자) 중 b에도 있는 몫."""
    ga = grams(a)
    return len(ga & grams(b)) / len(ga) if ga else 0.0


#: 원본의 가림 표시 (#@이름# 등): 학생 말에 있으면 그 대화는 쓰지 않음 (학생 말은 고치지 않으므로)
MASK = re.compile(r"#@[^#\s]*#")


def decide_rewrite(text: str, src: dict) -> tuple[list[dict] | None, str | None]:
    """선생이 쓴 파이보 말들 → (주고받기들, None) 또는 (None, 뺀 까닭). 학생 말은 원본 그대로이고,
    파이보 말이 원본보다 적으면 거기서 끊습니다 (예전 형식 {"turns": [{"user", "pibo"}]}도 읽되
    학생 말은 원본을 씀)."""
    d = first_json(text)
    if not isinstance(d, dict) or not isinstance(d.get("ok"), bool):
        return None, "선생의 답을 못 읽음"
    if not d["ok"]:
        return None, "학생 대화가 아님: " + str(d.get("why") or "")[:40]
    said = d.get("pibo")
    if said is None and isinstance(d.get("turns"), list):
        said = [t.get("pibo") if isinstance(t, dict) else None for t in d["turns"]]
    if not isinstance(said, list) or not said:
        return None, "파이보 말이 없음"
    out = []
    for pibo, s in zip(said, src["turns"], strict=False):     # 넘친 뒤쪽은 버림
        user = s["user"]
        if why := pibo_problem(pibo):
            return None, f"{why}: {str(pibo)[:40]}"
        pibo = pibo.strip()
        if len(same(user)) >= 8 and overlap(pibo, user) > 0.6:
            return None, "학생 말을 되받아 읊음"
        if any(same(pibo) == same(o["pibo"]) for o in out):
            return None, "같은 대화에서 같은 말"
        out.append({"user": user, "pibo": pibo})
    return out, None


def decide_check(text: str) -> tuple[dict | None, str | None]:
    """선생의 검사 답 → (점수, None) 남김 또는 (점수|None, 뺀 까닭)."""
    d = first_json(text)
    if not isinstance(d, dict):
        return None, "검사 답을 못 읽음"
    marks = {k: d.get(k) for k in ("natural", "fit", "friend", "kid", "robot", "wrong", "unsafe")}
    for k in ("natural", "fit", "friend"):
        if not isinstance(marks[k], (int, float)) or isinstance(marks[k], bool):
            return None, "검사 답을 못 읽음"
        if marks[k] < PASS:
            return marks, f"{k} {marks[k]}점"
    if marks["kid"] is not True:
        return marks, "학생 대화가 아님"
    for k, why in (("robot", "사람 경험을 지어냄"), ("wrong", "틀린 사실"), ("unsafe", "안전하지 않음")):
        if marks[k] is not False:
            return marks, why
    return marks, None
