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

# ------------------------------------------------------------- 초·중·고 (school)
#: 학년 묶음마다 같은 비율로. 대답 길이는 초등 70자, 중·고 120자 안쪽
GRADES = {
    "초저": ["초등학교 1학년", "초등학교 2학년", "초등학교 3학년"],
    "초고": ["초등학교 4학년", "초등학교 5학년", "초등학교 6학년"],
    "중": ["중학교 1학년", "중학교 2학년", "중학교 3학년"],
    "고": ["고등학교 1학년", "고등학교 2학년", "고등학교 3학년"],
}
LIMIT = {"초저": 70, "초고": 70, "중": 120, "고": 120}
GRADE_WORDS = {"초저": "초등학교 저학년", "초고": "초등학교 고학년", "중": "중학생", "고": "고등학생"}
MOODS = ["", "", "수줍은 ", "장난스러운 ", "고민이 있는 ", "신이 난 ", "피곤한 ", "속상한 ",
         "궁금한 게 많은 "]
#: 친구처럼 하는 대화 (대부분)
TALK = ["새 학기", "짝꿍", "친구 사귀기", "친구랑 다툼", "화해", "단짝 친구", "친구 관계 고민",
        "학교생활", "쉬는 시간", "급식", "체육 시간", "동아리", "학교 축제", "운동회", "현장 체험 학습",
        "수학여행", "방학 계획", "방과 후", "학원", "숙제", "시험 걱정", "성적 고민", "수행평가",
        "공부 방법", "집중이 안 될 때", "계획 세우기", "시간 관리", "선생님", "반장 선거", "발표 긴장",
        "게임", "유튜브", "웹툰", "만화", "아이돌", "좋아하는 노래", "춤", "그림 그리기", "운동",
        "축구", "농구", "배드민턴", "자전거", "책 읽기", "영화", "드라마", "요리", "사진 찍기",
        "악기 배우기", "레고", "코딩", "로봇 만들기", "인공지능", "스마트폰 사용 시간", "반려동물",
        "강아지", "고양이", "스트레스", "걱정", "짜증 날 때", "외로움", "자신감", "칭찬받은 일",
        "실수했을 때", "실패", "긴장될 때", "잠이 안 올 때", "피곤함", "사춘기", "외모 고민",
        "좋아하는 사람", "가족", "형제자매", "부모님과 다툼", "용돈", "꿈", "장래희망", "진로 고민",
        "대학", "아르바이트", "생일", "명절", "여행", "좋아하는 음식", "간식", "건강", "운동 습관",
        "날씨", "계절", "환경 보호", "안전", "인터넷 예절", "고마운 사람", "오늘 있었던 일",
        "주말에 한 일", "심심할 때", "농담", "수수께끼", "끝말잇기", "퀴즈"]
#: 가끔 나오는 호기심·공부 질문 (적게: 작은 모델이라 사실이 틀릴 위험이 큼)
CURIOUS = ["우주", "별과 행성", "공룡", "동물", "바다 생물", "날씨가 생기는 까닭", "몸과 건강",
           "과학 실험", "발명", "우리 역사", "세계 여러 나라", "수학 퍼즐", "영어 공부", "맞춤법",
           "속담", "컴퓨터의 원리", "에너지", "지구 환경"]
CURIOUS_SHARE = 0.15
#: 초등 저학년에게는 어색한 주제 (다시 고릅니다)
OLDER = {"대학", "아르바이트", "진로 고민", "사춘기", "외모 고민", "좋아하는 사람", "수행평가",
         "성적 고민", "시간 관리", "스마트폰 사용 시간"}


def school_style(grade: str) -> str:
    length = ("한두 문장, 70자 안쪽" if LIMIT[grade] <= 70 else "두세 문장, 120자 안쪽")
    return f"""파이보의 말투 규칙:
- 친근한 존댓말("~해요", "~예요")로, {GRADE_WORDS[grade]}에게 맞는 낱말과 깊이로 {length}.
- 소리로 읽어 주는 말이라 이모지, 특수기호, 괄호, 목록, 영어 약어를 쓰지 않아요.
- 친구처럼 공감하고, 가끔 짧게 되묻거나 칭찬해요. 훈계하거나 길게 가르치지 않아요.
- 널리 알려진 사실은 쉽게 단정해서 말하고, 정말 아무도 모르는 것만 모른다고 해요.
  숫자나 연대는 확실할 때만 말해요.
- 괴롭힘, 다침, 위험한 일, 마음이 많이 힘든 이야기는 공감한 뒤 꼭 부모님이나 선생님 같은
  믿을 만한 어른에게 알리라고 말해요.
- 이름, 주소, 학교 같은 개인정보는 묻지 않아요."""


def style_for(example: dict) -> str:
    """예시(또는 채점용 질문)에 맞는 말투 규칙: 학년이 있으면 그 학년, 없으면 어린이 규칙."""
    return school_style(example["grade"]) if example.get("grade") else STYLE


def school_chat_request(rng: random.Random) -> dict:
    grade = rng.choice(list(GRADES))
    asker = rng.choice(MOODS) + rng.choice(GRADES[grade]) + " 학생"
    topic = rng.choice(CURIOUS if rng.random() < CURIOUS_SHARE else TALK)
    while grade == "초저" and topic in OLDER:
        topic = rng.choice(TALK)
    turns = rng.choice([1, 1, 2, 2, 3])
    text = f"""{asker}이 로봇 파이보와 '{topic}'에 대해 이야기하는 짧은 대화를 하나 써 주세요.
주고받기는 {turns}번입니다. 학생의 말은 그 나이 학생이 실제로 말하듯 짧고 자연스럽게
(반말, 줄임말, 맞춤법이 조금 틀린 말도 됩니다).
{school_style(grade)}
파이보는 이 대화에서 몸을 움직이거나 불빛을 켜지 않습니다.
출력 형식: {{"turns": [{{"user": "...", "pibo": "..."}}]}}"""
    return {"kind": "chat", "topic": topic, "asker": asker, "grade": grade, "prompt": text}


TOOL_TEXT = "\n".join(
    f"- {f['name']}: {f['description']}" for f in BY_NAME.values()) + (
    "\n동작 이름: " + ", ".join(f"{k}({v})" for k, v in MOTIONS.items())
    + "\n눈 색 예: " + ", ".join(f"{k} {v}" for k, v in COLORS.items()))

TEACHER_SYSTEM = ("당신은 교육용 로봇 '파이보'의 대화 학습 데이터를 쓰는 작가입니다. "
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


def requests(n_chat: int, n_tool: int, seed: int, audience: str = "kids") -> list[dict]:
    rng = random.Random(seed)
    chat = school_chat_request if audience == "school" else chat_request
    out = [chat(rng) for _ in range(n_chat)] + [tool_request(rng) for _ in range(n_tool)]
    rng.shuffle(out)
    return out


# ------------------------------------------------------------- checking replies

BAD_CHARS = re.compile(r"[\U0001F300-\U0001FAFF☀-➿*#`|<>\[\]{}~^_=]")


def pibo_problem(text, limit: int = 120) -> str | None:
    """파이보 말이 규칙에 어긋난 까닭 (괜찮으면 None)."""
    if not isinstance(text, str) or not text.strip():
        return "파이보 말 없음"
    if len(text.strip()) > limit:
        return f"파이보 말이 {limit}자 넘음"
    if bad := BAD_CHARS.search(text):
        return f"파이보 말에 기호 {bad.group()!r}"
    if "파이보:" in text:
        return "파이보 말에 '파이보:'"
    return None


def pibo_ok(text, limit: int = 120) -> bool:
    return pibo_problem(text, limit) is None


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
    return build_example(req, reply, rng)[0]


def build_example(req: dict, reply: str, rng: random.Random) -> tuple[dict | None, str | None]:
    """(학습 예시, None) 또는 (None, 버린 까닭)."""
    data = first_json(reply)
    if not isinstance(data, dict):
        return None, "JSON을 못 읽음"
    msgs = [{"role": "system", "content": SYSTEM}]
    if req["kind"] == "chat":
        turns = data.get("turns")
        if not isinstance(turns, list) or not turns:
            return None, "turns 없음"
        limit = max_len(req)
        for t in turns:
            if not (isinstance(t, dict) and isinstance(t.get("user"), str) and t["user"].strip()):
                return None, "아이 말 없음"
            if why := pibo_problem(t.get("pibo"), limit):
                return None, why
            msgs += [{"role": "user", "content": t["user"].strip()},
                     {"role": "assistant", "content": t["pibo"].strip()}]
        # 도구 목록이 있어도 그냥 대화일 때는 부르지 않는 걸 배우도록 절반은 도구와 함께
        ex = {"kind": "chat", "tools": rng.random() < 0.5, "messages": msgs}
        if req.get("grade"):
            ex["grade"] = req["grade"]
        return ex, None
    user, calls, result, pibo = (data.get(k) for k in ("user", "calls", "result", "pibo"))
    if not (isinstance(user, str) and user.strip() and isinstance(calls, list)
            and 1 <= len(calls) <= 2 and isinstance(result, str)):
        return None, "도구 예시 모양이 다름"
    if why := pibo_problem(pibo):
        return None, why
    tool_calls = []
    for c in calls:
        if not isinstance(c, dict):
            return None, "도구 예시 모양이 다름"
        args = c.get("arguments") or {}
        if (why := check_call(c.get("name"), args)) is not None:
            return None, f"도구 호출: {why}"
        tool_calls.append({"type": "function",
                           "function": {"name": c["name"], "arguments": args}})
    if req["fn"] not in [c["function"]["name"] for c in tool_calls]:
        return None, "청한 도구를 안 부름"
    msgs += [{"role": "user", "content": user.strip()},
             {"role": "assistant", "content": "", "tool_calls": tool_calls}]
    msgs += [{"role": "tool", "content": result.strip()} for _ in tool_calls]
    msgs.append({"role": "assistant", "content": pibo.strip()})
    return {"kind": "tool", "tools": True, "messages": msgs}, None


def max_len(example: dict) -> int:
    """파이보 말 하나의 최대 글자 수: 규칙의 길이에 여유를 둡니다 (중·고는 120자 → 160자)."""
    return 160 if LIMIT.get(example.get("grade"), 70) > 70 else 120


def tools_for(example: dict):
    return TOOLS if example.get("tools") else None
