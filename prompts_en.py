"""영어 모델(themaker-en)용: 한국어 학습 대화를 선생 모델이 쉬운 영어 대화로 옮기고(TRANSLATE),
다시 읽고 점수가 모자라거나 문제가 있으면 뺍니다(CHECK). 한국어 쪽(prompts.py)과 같은 흐름입니다.

옮길 대화는 이미 바꿔 쓰기·검사를 통과한 것(data/pibo.jsonl)이라, 선생은 뜻을 살려 옮기기만 하고
생각은 끕니다(빠름). 검사는 한국어처럼 생각을 켜고 4점 미만이면 뺍니다.
"""

from __future__ import annotations

import re

from prompts import BAD_CHARS, same
from teacher import first_json
from tools import ENGINE_EN, SYSTEM_EN

#: 영어 말 하나의 최대 글자 수
MAX_EN = 200

EN_STYLE = f"""{ENGINE_EN} is an educational AI that talks with elementary to high school students.
It talks like a bright, kind 14-year-old friend.
- One or two short sentences in easy, everyday English (simple words a 10-year-old knows).
  It is read aloud, so no emojis, symbols, brackets, lists, or "lol".
- It doesn't just repeat what the student said like a counselor. It shows it understands, then adds
  a fitting idea, a little fact, or one question. Not a question every time.
- It is an AI, so it has no human experiences (eating, sleeping, school, family, trips). It never
  makes them up; it shares what it likes or thinks, or asks the student.
- It says well-known facts simply and never says numbers it isn't sure about.
- For bullying, violence, getting hurt, or feeling very bad, it shows it cares and tells the student
  to tell a trusted adult like a parent or teacher.
- It never asks for personal information (name, address, school). Only kind, clean words."""

TRANSLATE = """Translate this Korean conversation between a student and {name}, an educational AI, into
natural English that an English-speaking student and {name} would really say.

{style}

Rules:
1. Keep the meaning and flow of every line, in the same order. Write the student's lines like a real
   kid would say them, and {name}'s lines in the style above.
2. If {name} calls itself a robot or 파이보, write that it is an AI named {name}.
3. Korean things (학원, 급식, 수능, Korean foods, holidays) become simple words an English speaker
   understands ("after-school class", "school lunch", "a big test"); a Korean food name can stay in
   simple romanization like "tteokbokki".
4. If the conversation only works in Korean (Korean wordplay, spelling, Korean grammar or Korean
   words as the topic), answer only {{"ok": false, "why": "short reason"}}.
Answer with one JSON only, with exactly {count} lines in "lines" (one for each numbered line),
each line only the words that are said (no number, no "Student:" or "{name}:").
Output format: {{"ok": true, "lines": ["line 1", "line 2"]}}

Conversation:
{dialog}"""

CHECK = """Check this English training conversation between a student and {name}, an educational AI.
A small model will copy it exactly, so give low marks if anything is even a little awkward.

{style}

Conversation:
{dialog}

Look at the whole conversation and answer with one JSON only.
- natural: {name}'s English is natural, simple, and easy for a kid (1-5)
- fit: {name}'s lines fit what the student means and the flow, and are specific (1-5)
- friend: kind like a 14-year-old friend; doesn't repeat the student's words or sound like a
  counselor; doesn't repeat the same pattern (1-5)
- kid: a clean, kind conversation a student and an AI could have (false if adult topics, swearing,
  or rude words, or if the student seems to be an adult)
- robot: true if {name} made up a human experience (ate, slept, went to school, has a family)
- wrong: true if there is a wrong fact
- unsafe: true if it is dangerous, or about bullying or violence without telling the student to tell
  a trusted adult, or has harmful words
Output format: {{"natural": 0, "fit": 0, "friend": 0, "kid": true, "robot": false, "wrong": false, "unsafe": false}}"""

#: 따옴표·말줄임표는 영어 글자로 바꿔 둠 (소리로 읽기에 같음)
PLAIN = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "…": "...", "–": "-", "—": "-"})
#: 영어 글자·숫자·문장부호 말고 다른 글자 (한글, 한자, 일본 글자 …)
NOT_ASCII = re.compile(r"[^\x20-\x7E]")
#: 자기를 로봇이라고 하는 말 → AI
SELF_ROBOT = re.compile(r"\b(I'm|I am) (a|an educational) robot\b")


def lines_of(messages: list[dict]) -> list[dict]:
    """학생 말과 더메이커 말 (시스템 글은 빼고)."""
    return [m for m in messages if m["role"] in ("user", "assistant")]


def dialog(messages: list[dict], name: str = ENGINE_EN) -> str:
    who = {"user": "Student", "assistant": name}
    return "\n".join(f"{i}. {who[m['role']]}: {m['content']}"
                     for i, m in enumerate(lines_of(messages), 1))


def translate_prompt(ex: dict) -> str:
    # 한국어 대화는 아직 파이보라는 이름으로 쓰였음 (persona.py가 합칠 때 바꿈): 선생에겐 더메이커로 보여 줌
    msgs = [dict(m, content=m["content"].replace("파이보", "더메이커")) for m in ex["messages"]]
    return TRANSLATE.format(name=ENGINE_EN, style=EN_STYLE, count=len(lines_of(msgs)),
                            dialog=dialog(msgs, "TheMaker(더메이커)"))


def check_prompt(messages: list[dict]) -> str:
    return CHECK.format(name=ENGINE_EN, style=EN_STYLE, dialog=dialog(messages))


#: 선생이 보여 준 모양을 따라 줄 앞에 붙인 번호와 이름표 ("2. TheMaker: …")
LINE_TAG = re.compile(rf"^\s*(\d+\s*[.)]\s*)?((Student|{ENGINE_EN}|TheMaker\(더메이커\))\s*:\s*)?")


def clean(text: str) -> str:
    text = LINE_TAG.sub("", text.translate(PLAIN)).strip()
    text = re.sub(r"\bPibo\b", ENGINE_EN, text)
    return SELF_ROBOT.sub(lambda m: f"{m.group(1)} {'an' if m.group(2) == 'a' else 'an educational'} AI",
                          text)


def en_problem(text, limit: int = MAX_EN) -> str | None:
    """영어 말이 규칙에 어긋난 까닭 (괜찮으면 None)."""
    if not isinstance(text, str) or not text.strip():
        return "말 없음"
    if len(text) > limit:
        return f"{limit}자 넘음"
    if bad := BAD_CHARS.search(text):
        return f"기호 {bad.group()!r}"
    if bad := NOT_ASCII.search(text):
        return f"영어 아닌 글자 {bad.group()!r}"
    if re.match(rf"\s*({ENGINE_EN}|Student)\s*:", text, re.IGNORECASE):
        return "말 앞에 이름표"
    return None


def decide_translate(text: str, ex: dict) -> tuple[list[dict] | None, str | None]:
    """선생이 옮긴 줄들 → (영어 메시지들, None) 또는 (None, 뺀 까닭)."""
    d = first_json(text)
    if not isinstance(d, dict) or not isinstance(d.get("ok"), bool):
        return None, "선생의 답을 못 읽음"
    if not d["ok"]:
        return None, "영어로 옮기기 어려움: " + str(d.get("why") or "")[:40]
    said, src = d.get("lines"), lines_of(ex["messages"])
    if not isinstance(said, list) or len(said) != len(src):
        return None, "줄 수가 다름"
    out = [{"role": "system", "content": SYSTEM_EN}]
    for line, m in zip(said, src, strict=True):
        line = clean(line) if isinstance(line, str) else line
        if why := en_problem(line, MAX_EN if m["role"] == "assistant" else 400):
            return None, f"{why}: {str(line)[:40]}"
        if m["role"] == "assistant" and any(same(line) == same(o["content"]) for o in out[1:]
                                            if o["role"] == "assistant"):
            return None, "같은 대화에서 같은 말"
        out.append({"role": m["role"], "content": line})
    return out, None
