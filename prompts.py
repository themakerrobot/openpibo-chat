"""큰 모델(선생 모델)에게 학습 예시를 써 달라고 하는 글, 그리고 받은 글의 검사.

예시 하나는 OpenAI 형식의 메시지 목록입니다::

    [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}, ...]

도구를 부르는 예시는 assistant 메시지에 ``tool_calls``, 이어서 ``tool`` 메시지
(도구의 결과)와 그걸 보고 하는 마지막 대답이 붙습니다.
"""

from __future__ import annotations

import json
import random
import re

from tools import BY_NAME, COLORS, MOTIONS, SYSTEM, TOOLS, check_call

TOPICS = ["공룡", "우주", "별", "달", "해", "행성", "바다", "물고기", "고래", "상어", "문어", "곤충", "개미", "나비", "벌", "거미", "강아지", "고양이", "토끼", "햄스터", "사자", "호랑이", "코끼리", "기린", "펭귄", "북극곰", "새", "독수리", "앵무새", "꽃", "나무", "씨앗", "과일", "채소", "사과", "바나나", "딸기", "수박", "김치", "밥", "빵", "아이스크림", "초콜릿", "양치", "손", "씻기", "잠", "꿈", "아침", "인사", "저녁", "인사", "생일", "선물", "크리스마스", "설날", "추석", "눈", "비", "바람", "무지개", "천둥", "계절", "봄", "여름", "가을", "겨울", "숫자", "세기", "더하기", "빼기", "모양", "색깔", "알파벳", "한글", "받아쓰기", "동화", "책", "그림", "그리기", "노래", "춤", "피아노", "악기", "축구", "야구", "줄넘기", "자전거", "수영", "놀이터", "학교", "유치원", "선생님", "친구", "싸움과", "화해", "칭찬", "감사", "사과하기", "슬픔", "화남", "무서움", "외로움", "기쁨", "자랑", "걱정", "엄마", "아빠", "할머니", "형", "누나", "동생", "아기", "가족", "소방관", "경찰관", "의사", "요리사", "우주비행사", "로봇", "컴퓨터", "자동차", "기차", "비행기", "배", "신호등", "길", "건너기", "안전", "불", "조심", "낯선", "사람", "병원", "주사", "감기", "이", "빠짐", "키", "크기", "운동", "물", "마시기", "재활용", "쓰레기", "지구", "환경", "수수께끼", "끝말잇기", "스무고개", "숨바꼭질", "가위바위보", "퀴즈", "농담", "이름", "짓기", "장래희망", "소원", "외계인", "마법", "보물", "해적", "왕과", "공주", "미로", "블록", "레고", "퍼즐", "종이접기", "풍선", "비눗방울", "모래성", "캠핑", "소풍", "동물원", "박물관", "시계", "보기", "요일", "달력"]

ASKERS = ["5살 아이", "6살 아이", "7살 아이", "초등학교 1학년", "초등학교 2학년", "초등학교 3학년",
          "수줍은 아이", "장난꾸러기 아이", "질문이 많은 아이", "속상한 아이", "부모님", "선생님"]

STYLE = """파이보의 말투 규칙:
- 친근한 존댓말("~해요", "~예요")로, 한두 문장만. 길어도 70자 안쪽.
- 소리로 읽어 주는 말이라 이모지, 특수기호, 괄호, 목록, 영어 약어를 쓰지 않아요.
- 아이 눈높이의 쉬운 낱말을 쓰고, 가끔 짧게 되묻거나 칭찬해요.
- 모르거나 확실하지 않은 사실은 지어내지 말고 모른다고 말해요.
- 위험한 일, 아픈 곳, 낯선 사람 이야기는 꼭 어른에게 알리라고 말해요.
- 이름, 주소 같은 개인정보는 묻지 않아요."""

TOOL_TEXT = "\n".join(
    f"- {f['name']}: {f['description']}" for f in BY_NAME.values()) + (
    "\n동작 이름: " + ", ".join(f"{k}({v})" for k, v in MOTIONS.items())
    + "\n눈 색 예: " + ", ".join(f"{k} {v}" for k, v in COLORS.items()))

TEACHER_SYSTEM = ("당신은 어린이용 교육 로봇 '파이보'의 대화 학습 데이터를 쓰는 작가입니다. "
                  "요청한 JSON 하나만 출력하고, 다른 말은 쓰지 않습니다.")


def chat_request(rng: random.Random) -> dict:
    topic, asker, turns = rng.choice(TOPICS), rng.choice(ASKERS), rng.choice([1, 1, 2, 2, 3])
    text = f"""{asker}가 로봇 파이보와 '{topic}'에 대해 이야기하는 짧은 대화를 하나 써 주세요.
주고받기는 {turns}번입니다. 아이의 말은 실제 아이가 말하듯 짧고 자연스럽게(맞춤법이 조금 틀려도 됩니다).
{STYLE}
파이보는 이 대화에서 몸을 움직이거나 불빛을 켜지 않습니다.
출력 형식: {{"turns": [{{"user": "...", "pibo": "..."}}]}}"""
    return {"kind": "chat", "topic": topic, "asker": asker, "prompt": text}


def tool_request(rng: random.Random) -> dict:
    fn = rng.choice(list(BY_NAME))
    asker = rng.choice(ASKERS)
    hint = {
        "set_motion": f"동작은 '{rng.choice(list(MOTIONS))}' 쪽으로 정해 주세요",
        "eye_on": f"색은 '{rng.choice(list(COLORS))}' 쪽으로 정해 주세요",
        "get_weather": f"지역은 '{rng.choice(['서울', '부산', '대구', '인천', '광주', '대전', '제주', '전국'])}'",
    }.get(fn, "")
    text = f"""{asker}가 로봇 파이보에게 무언가를 해 달라고 하거나 물어서, 파이보가 도구 '{fn}'을 써야 하는
장면을 하나 써 주세요. {hint}
파이보의 도구:
{TOOL_TEXT}
- user: 아이의 말 (도구 이름을 직접 말하지 않고 자연스럽게. 예: "춤춰 줘", "눈 파랗게 해 봐")
- calls: 파이보가 부르는 도구들 (보통 하나, 꼭 필요할 때만 둘). 인자는 위 목록 그대로
- result: 도구가 돌려준 결과를 짧게 지어서 (예: "완료", "배터리 70%", "컵, 사과", "서울 맑음 18도")
- pibo: 결과를 보고 파이보가 하는 마지막 말
{STYLE}
출력 형식: {{"user": "...", "calls": [{{"name": "...", "arguments": {{...}}}}], "result": "...", "pibo": "..."}}"""
    return {"kind": "tool", "fn": fn, "asker": asker, "prompt": text}


def requests(n_chat: int, n_tool: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    out = [chat_request(rng) for _ in range(n_chat)] + [tool_request(rng) for _ in range(n_tool)]
    rng.shuffle(out)
    return out


# ------------------------------------------------------------- checking replies

BAD_CHARS = re.compile(r"[\U0001F300-\U0001FAFF☀-➿*#`|<>\[\]{}~^_=]")


def pibo_ok(text) -> bool:
    return (isinstance(text, str) and 1 <= len(text.strip()) <= 120
            and not BAD_CHARS.search(text) and "파이보:" not in text)


def first_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        depth += ch == "{"
        depth -= ch == "}"
        if depth == 0:
            try:
                return json.loads(text[start:i + 1])
            except json.JSONDecodeError:
                return None
    return None


def after_think(text: str) -> str:
    """생각을 켜고 받은 답에서 생각 뒤의 글만. 생각이 길어 잘렸으면 빈 글."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1]
    return "" if "<think>" in text else text


def to_example(req: dict, reply: str, rng: random.Random) -> dict | None:
    """선생 모델의 답을 학습 예시로. 규칙에 어긋나면 None."""
    data = first_json(reply)
    if not isinstance(data, dict):
        return None
    msgs = [{"role": "system", "content": SYSTEM}]
    if req["kind"] == "chat":
        turns = data.get("turns")
        if not isinstance(turns, list) or not turns:
            return None
        for t in turns:
            if not (isinstance(t, dict) and isinstance(t.get("user"), str)
                    and t["user"].strip() and pibo_ok(t.get("pibo"))):
                return None
            msgs += [{"role": "user", "content": t["user"].strip()},
                     {"role": "assistant", "content": t["pibo"].strip()}]
        # 도구 목록이 있어도 그냥 대화일 때는 부르지 않는 걸 배우도록 절반은 도구와 함께
        return {"kind": "chat", "tools": rng.random() < 0.5, "messages": msgs}
    user, calls, result, pibo = (data.get(k) for k in ("user", "calls", "result", "pibo"))
    if not (isinstance(user, str) and user.strip() and isinstance(calls, list)
            and 1 <= len(calls) <= 2 and pibo_ok(pibo) and isinstance(result, str)):
        return None
    tool_calls = []
    for c in calls:
        if not isinstance(c, dict):
            return None
        args = c.get("arguments") or {}
        if check_call(c.get("name"), args) is not None:
            return None
        tool_calls.append({"type": "function",
                           "function": {"name": c["name"], "arguments": args}})
    if req["fn"] not in [c["function"]["name"] for c in tool_calls]:
        return None
    msgs += [{"role": "user", "content": user.strip()},
             {"role": "assistant", "content": "", "tool_calls": tool_calls}]
    msgs += [{"role": "tool", "content": result.strip()} for _ in tool_calls]
    msgs.append({"role": "assistant", "content": pibo.strip()})
    return {"kind": "tool", "tools": True, "messages": msgs}


def tools_for(example: dict):
    return TOOLS if example.get("tools") else None
