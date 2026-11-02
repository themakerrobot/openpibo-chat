"""코딩 도우미(themaker-code)용: 블록·함수 설명 하나를 근거로 선생이 학생 질문과 대답을 만들고(GEN),
다시 읽고 설명과 다르거나 어렵거나 지어낸 것이 있으면 뺍니다(CHECK). 대답 속 파이썬 코드는 기기마다 쓸 수 있는
import·객체·메서드 목록(data/code-api.json)과 글자로 대조합니다.
"""

from __future__ import annotations

import ast
import json
import re

from prompts import BANMAL, polite_ends
from teacher import first_json

#: 대답 하나의 최대 글자 수 (코드 포함)
MAX_ANSWER = 700
#: 설명 하나에서 만드는 문답 수
PAIRS = 7

CODE_STYLE = """더메이커는 학생과 친구처럼 이야기하는 교육용 AI 코딩 도우미예요.
- 친근한 존댓말("~해요")로 말해요. 학생을 "-시-"로 높이지 않아요 ("하셨어요" 대신 "했어요").
- 설명은 2~4문장으로 짧고 쉽게. 어려운 말은 풀어서 말해요.
- 코드는 꼭 필요할 때만 ```python 으로 감싸서 10줄 안쪽으로 보여 줘요. 코드 첫머리에 준비 코드(import와 객체 만들기)를 넣어요.
- 이모지는 쓰지 않아요. 블록은 블록 글에서 □를 빼고 「」로 감싸 말해요 (예: 「RGB 눈 불빛 바꾸기」).
- 설명에 없는 함수, 인자, 값은 절대 지어내지 않아요."""

GEN = """아래는 {device} 블록 코딩 도구의 {what} 설명이에요. 이 설명만 근거로, {device}로 블록 코딩과 파이썬을
배우는 초·중·고 학생이 물을 만한 질문과 코딩 도우미 더메이커의 대답을 {n}쌍 만들어요.

{style}

질문을 골고루 섞어요:
1. 이게 무엇을 하는지
2. {how}
3. 하고 싶은 일로 묻기 (예: "눈을 빨갛게 하려면 어떻게 해?") — 이 {what}{ro} 할 수 있는 일만
4. 값의 범위, 넣는 순서, 자주 하는 실수 (설명에 있을 때만)
5. 짧은 예제 코드나 블록 순서
질문은 학생 말투(반말, 짧게)로, 서로 다른 말로 써요. 설명이 모자라 대답할 수 없는 질문은 만들지 않아요.
JSON 하나로만 답해요. 출력 형식: {{"pairs": [{{"q": "학생 질문", "a": "더메이커 대답"}}]}}

[설명]
{doc}"""

CHECK = """아래 설명을 근거로 만든 코딩 도우미 문답을 검사해요. 작은 모델이 그대로 배우므로 조금이라도 틀리거나
어색하면 낮게 줘요.

{style}

[설명]
{doc}

[문답]
{pairs}

문답마다 하나씩, 같은 순서로 JSON 하나로만 답해요.
- grounded: 대답이 설명과 맞고, 설명에 없는 함수·인자·값을 지어내지 않았는가 (1~5)
- clear: 학생이 알아듣게 짧고 쉬운가, 질문에 맞게 대답했는가 (1~5)
- code_ok: 코드가 있으면 그대로 돌릴 수 있게 맞는가 (코드가 없으면 true)
출력 형식: {{"marks": [{{"grounded": 0, "clear": 0, "code_ok": true}}]}}"""

PASS = 4
CODE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)
EMOJI = re.compile(r"[\U0001F300-\U0001FAFF☀-➿]")


def unit_doc(u: dict) -> str:
    """선생에게 보여 줄 설명 글."""
    if u["kind"] == "block":
        lines = [f"블록 카테고리: {u.get('category') or ''}",
                 f"블록 글 (□는 값을 끼우거나 고르는 칸): {u['text']}",
                 f"툴팁: {u.get('tooltip') or ''}"]
        for f in u.get("fields") or []:
            desc = f"- 칸 {f['name']}: {'고르기' if f['type'] == 'field_dropdown' else '값 끼우기'}"
            if f.get("check"):
                desc += f" ({f['check']})"
            if f.get("options"):
                opts = [o[0] if isinstance(o, list) else str(o) for o in f["options"]][:14]
                desc += " 고를 수 있는 값: " + ", ".join(opts)
            lines.append(desc)
        if u.get("code"):
            lines.append("이 블록이 만드는 파이썬 코드 (<이름>은 칸에 넣은 값):\n" + u["code"].strip())
        if u.get("imports"):
            lines.append("준비 코드:\n" + "\n".join(u["imports"]))
        return "\n".join(lines)
    setup = f"from {u['module']} import {u['class']}" + (f" as {u['alias']}" if u["alias"] != u["class"] else "")
    if u.get("instance"):
        setup += f"\n{u['instance']} = {u['alias']}()"
    return (f"파이썬 함수: {u['alias']}.{u['signature']}\n준비 코드:\n{setup}\n"
            f"설명:\n{u.get('doc') or '(설명 없음)'}")


def gen_prompt(u: dict) -> str:
    block = u["kind"] == "block"
    return GEN.format(device=u["device"], what="블록" if block else "파이썬 함수", ro="으로" if block else "로",
                      n=PAIRS, style=CODE_STYLE,
                      how=("블록 상자의 어느 카테고리에 있고, 칸에 무엇을 넣는지, 어떤 파이썬 코드가 되는지" if block else
                           "파이썬에서 어떻게 부르는지 (준비 코드와 함께)"),
                      doc=unit_doc(u))


def check_prompt(u: dict, pairs: list[dict]) -> str:
    text = "\n\n".join(f"{k}. 학생: {p['q']}\n   더메이커: {p['a']}" for k, p in enumerate(pairs, 1))
    return CHECK.format(style=CODE_STYLE, doc=unit_doc(u), pairs=text)


def prose(answer: str) -> str:
    return CODE.sub(" ", answer)


def code_problem(code: str, api: dict) -> str | None:
    """코드가 파이썬으로 읽히고, openpibo는 그 기기에 있는 import·객체·메서드만 쓰는지."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"코드 문법 오류: {exc.msg}"
    classes = {c for _, c in api["imports"].values()}
    modules = {m for m, _ in api["imports"].values()} | {"openpibo.utils"}
    instances = dict(api["instances"])
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("openpibo"):
            if node.module not in modules:
                return f"없는 모듈 {node.module}"
            for n in node.names:
                if n.name not in classes:
                    return f"없는 클래스 {n.name}"
        elif isinstance(node, ast.Import):
            for n in node.names:
                if n.name.startswith("openpibo") and n.name not in modules:
                    return f"없는 모듈 {n.name}"
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and \
                isinstance(node.value.func, ast.Name) and node.value.func.id in api["imports"]:
            for target in node.targets:                       # 학생 코드가 다른 이름으로 만들어도 따라감
                if isinstance(target, ast.Name):
                    instances[target.id] = node.value.func.id
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and \
                isinstance(node.func.value, ast.Name) and node.func.value.id in instances:
            alias = instances[node.func.value.id]
            if node.func.attr not in api["methods"].get(alias, []):
                return f"없는 메서드 {alias}.{node.func.attr}"
    return None


def answer_problem(answer, api: dict) -> str | None:
    if not isinstance(answer, str) or not answer.strip():
        return "대답 없음"
    if len(answer) > MAX_ANSWER:
        return f"대답이 {MAX_ANSWER}자 넘음"
    text = prose(answer)
    if bad := EMOJI.search(text):
        return f"이모지 {bad.group()!r}"
    if BANMAL.search(text) or not polite_ends(text):
        return "반말"
    for code in CODE.findall(answer):
        if why := code_problem(code, api):
            return why
    if answer.count("```") % 2:
        return "코드 블록이 닫히지 않음"
    return None


def decide_gen(text: str, u: dict, api: dict) -> tuple[list[dict], dict]:
    """선생이 만든 문답 → (남은 문답들, 뺀 까닭마다 수)."""
    d = first_json(text)
    why: dict = {}
    if not isinstance(d, dict) or not isinstance(d.get("pairs"), list):
        return [], {"선생의 답을 못 읽음": 1}
    kept, seen = [], set()
    for p in d["pairs"]:
        if not isinstance(p, dict) or not isinstance(p.get("q"), str) or not p["q"].strip():
            why["질문 없음"] = why.get("질문 없음", 0) + 1
            continue
        q, a = p["q"].strip(), p.get("a")
        if q in seen:
            why["같은 질문"] = why.get("같은 질문", 0) + 1
            continue
        if problem := answer_problem(a, api):
            key = " ".join(problem.split(" ")[:2]) if problem.startswith(("없는", "코드")) else problem
            why[key] = why.get(key, 0) + 1
            continue
        seen.add(q)
        kept.append({"q": q, "a": a.strip()})
    return kept, why


def decide_check(text: str, n: int) -> list[dict | None]:
    """검사 답 → 문답마다 점수 (못 읽으면 None)."""
    d = first_json(text)
    marks = d.get("marks") if isinstance(d, dict) else None
    if not isinstance(marks, list) or len(marks) != n:
        return [None] * n
    return [m if isinstance(m, dict) else None for m in marks]


def passes(m: dict | None) -> str | None:
    """점수가 기준을 넘으면 None, 아니면 까닭."""
    if m is None:
        return "검사 답을 못 읽음"
    for k in ("grounded", "clear"):
        v = m.get(k)
        if not isinstance(v, (int, float)) or isinstance(v, bool) or v < PASS:
            return f"{k} {v}점"
    if m.get("code_ok") is not True:
        return "코드가 틀림"
    return None


def dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)
