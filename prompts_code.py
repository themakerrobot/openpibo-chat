"""코딩 도우미(themaker-code)용: 블록·함수 설명 하나를 근거로 선생이 학생 질문과 대답을 만들고(GEN),
다시 읽고 설명과 다르거나 어렵거나 지어낸 것이 있으면 뺍니다(CHECK).

글자 검사 (data/code-api.json과 대조):
- 대답 속 파이썬 코드: 문법, 그 기기에 있는 openpibo 모듈·클래스·메서드만, 표준 모듈(time 등) 말고 다른 import 금지
- 글 속 코드 조각(`device.eye_on(...)`): 같은 규칙
- 「」로 부른 블록 이름: 그 기기의 블록 상자에 있는 블록만
- 반말, 이모지, 길이
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
#: openpibo 말고 써도 되는 파이썬 모듈
STDLIB = {"time", "random", "math", "os", "datetime", "json"}

CODE_STYLE = """더메이커는 학생과 친구처럼 이야기하는 교육용 AI 코딩 도우미예요.
- 친근한 존댓말("~해요")로 말해요. 학생을 "-시-"로 높이지 않아요 ("하셨어요" 대신 "했어요").
- 설명은 2~4문장으로 짧고 쉽게. 어려운 말은 풀어서 말해요.
- 코드는 꼭 필요할 때만 ```python 으로 감싸서 10줄 안쪽으로 보여 줘요. 코드 첫머리에 준비 코드(import와 객체 만들기)를 넣어요.
- 이모지는 쓰지 않아요. 블록은 「」로 감싸 블록 목록의 글자로 말해요. 칸은 빼도 돼요 (예: 「오디오 크기로 재생하기」).
- 설명과 블록 목록에 없는 블록, 함수, 인자, 값, 모듈은 절대 지어내지 않아요."""

GEN = """아래는 {device} 블록 코딩 도구의 {what} 설명이에요. 이 설명만 근거로, {device}로 블록 코딩과 파이썬을
배우는 초·중·고 학생이 물을 만한 질문과 코딩 도우미 더메이커의 대답을 {n}쌍 만들어요.

{style}

질문을 골고루 섞어요:
1. 이게 무엇을 하는지
2. {how}
3. 하고 싶은 일로 묻기 (예: "눈을 빨갛게 하려면 어떻게 해?") — 이 {what}{ro} 할 수 있는 일만
4. 값의 범위, 넣는 순서, 자주 하는 실수 (설명에 있을 때만)
5. 짧은 예제 코드나 블록 순서 (다른 블록을 함께 쓸 때는 아래 블록 목록에 있는 블록만)
질문은 학생 말투(반말, 짧게)로, 서로 다른 말로 써요. 설명이 모자라 대답할 수 없는 질문은 만들지 않아요.
설명이 짧으면 문답을 {n}쌍보다 적게 만들어도 돼요.
JSON 하나로만 답해요. 출력 형식: {{"pairs": [{{"q": "학생 질문", "a": "더메이커 대답"}}]}}

[설명]
{doc}

[{device} 블록 상자에 있는 블록 (카테고리: 블록 이름)]
{blocks}"""

CHECK = """아래 설명을 근거로 만든 코딩 도우미 문답을 검사해요. 작은 모델이 그대로 배우므로 조금이라도 틀리거나
어색하면 낮게 줘요.

{style}

[설명]
{doc}

[{device} 블록 상자에 있는 블록 (카테고리: 블록 이름)]
{blocks}

[문답]
{pairs}

문답마다 하나씩, 같은 순서로 JSON 하나로만 답해요.
- grounded: 대답이 설명과 맞는가 (1~5). 설명에 없는 기능이나, 블록 목록에 없는 블록, 설명에 없는 함수·인자·값·모듈이
  하나라도 나오면 1점
- clear: 학생이 알아듣게 짧고 쉬운가, 질문에 맞게 대답했는가 (1~5)
- code_ok: 코드가 있으면 그대로 돌릴 수 있게 맞는가 (코드가 없으면 true)
출력 형식: {{"marks": [{{"grounded": 0, "clear": 0, "code_ok": true}}]}}"""

PASS = 4
CODE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)
INLINE = re.compile(r"`([^`\n]+)`")
QUOTED = re.compile(r"「([^」]+)」")
EMOJI = re.compile(r"[\U0001F300-\U0001FAFF☀-➿]")


def unit_doc(u: dict) -> str:
    """선생에게 보여 줄 설명 글."""
    if u["kind"] == "block":
        lines = [f"블록 카테고리: {u.get('category') or ''}",
                 f"블록 모양 ([ ▾]는 고르는 칸, ([ ])는 값을 끼우는 칸): 「{u.get('name') or u['text']}」",
                 f"설명: {u.get('guide') or u.get('tooltip') or ''}"]
        if u.get("note"):
            lines.append(f"덧붙임: {u['note']}")
        for f in u.get("fields") or []:
            desc = f"- 칸 {f['name']}: {'고르기' if f['type'] == 'field_dropdown' else '값 끼우기'}"
            if f.get("check"):
                desc += f" ({f['check']})"
            if f.get("options"):
                opts = [o[0] if isinstance(o, list) else str(o) for o in f["options"]][:14]
                desc += " 고를 수 있는 값: " + ", ".join(opts)
            lines.append(desc)
        if u.get("byOption"):
            lines.append(f"고르는 값({u['byOption']['field']})마다 만드는 파이썬 코드 (<이름>은 칸에 넣은 값):")
            lines += [f"  {k}: {v.strip()}" for k, v in u["byOption"]["codes"].items()]
        elif u.get("code"):
            lines.append("이 블록이 만드는 파이썬 코드 (<이름>은 칸에 넣은 값):\n" + u["code"].strip())
        if u.get("imports"):
            lines.append("준비 코드:\n" + "\n".join(u["imports"]))
        return "\n".join(lines)
    setup = f"from {u['module']} import {u['class']}" + (f" as {u['alias']}" if u["alias"] != u["class"] else "")
    if u.get("instance"):
        setup += f"\n{u['instance']} = {u['alias']}()"
    return (f"파이썬 함수: {u['alias']}.{u['signature']}\n준비 코드:\n{setup}\n"
            f"설명:\n{u.get('doc') or '(설명 없음)'}")


def block_list(api: dict) -> str:
    return "\n".join(f"{cat}: " + ", ".join(f"「{n}」" for n in names) for cat, names in api.get("blocks", {}).items())


def gen_prompt(u: dict, api: dict) -> str:
    block = u["kind"] == "block"
    return GEN.format(device=u["device"], what="블록" if block else "파이썬 함수", ro="으로" if block else "로",
                      n=PAIRS, style=CODE_STYLE,
                      how=("블록 상자의 어느 카테고리에 있고, 칸에 무엇을 넣는지, 어떤 파이썬 코드가 되는지" if block else
                           "파이썬에서 어떻게 부르는지 (준비 코드와 함께)"),
                      doc=unit_doc(u), blocks=block_list(api))


def check_prompt(u: dict, pairs: list[dict], api: dict) -> str:
    text = "\n\n".join(f"{k}. 학생: {p['q']}\n   더메이커: {p['a']}" for k, p in enumerate(pairs, 1))
    return CHECK.format(style=CODE_STYLE, doc=unit_doc(u), pairs=text, device=u["device"], blocks=block_list(api))


def prose(answer: str) -> str:
    return CODE.sub(" ", answer)


def norm(name: str, slots: bool = True) -> str:
    """견줄 모양: 띄어쓰기·기호를 빼고, slots가 거짓이면 칸([폴더 선택 ▾], ([80]))도 통째로 뺌."""
    if not slots:
        name = re.sub(r"\[[^\]]*\]|\([^)]*\)", "", name)
    return re.sub(r"[\s□▾⏵\[\]()〈〉<>.,·]", "", name)


def known_block(name: str, api: dict) -> bool:
    """「」 안의 이름이 그 기기 블록 모양과 같거나 그 일부(4글자 이상)인지 (칸을 넣어 말해도, 빼고 말해도)."""
    blocks = [b for bs in api.get("blocks", {}).values() for b in bs]
    forms = {f for b in blocks for f in (norm(b), norm(b, slots=False))}
    for n in {norm(name), norm(name, slots=False)}:
        if n and any(n == f or (len(n) >= 4 and n in f) for f in forms):
            return True
    return False


def code_problem(code: str, api: dict) -> str | None:
    """코드가 파이썬으로 읽히고, openpibo는 그 기기에 있는 import·객체·메서드만, 그 밖에는 표준 모듈만 쓰는지."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"코드 문법 오류: {exc.msg}"
    classes = {c for _, c in api["imports"].values()}
    modules = {m for m, _ in api["imports"].values()} | {"openpibo.utils"}
    instances = dict(api["instances"])
    imported = set(STDLIB)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod.startswith("openpibo"):
                if mod not in modules:
                    return f"없는 모듈 {mod}"
                for n in node.names:
                    if n.name not in classes and mod != "openpibo.utils":
                        return f"없는 클래스 {n.name}"
            elif mod.split(".")[0] not in STDLIB:
                return f"없는 모듈 {mod}"
        elif isinstance(node, ast.Import):
            for n in node.names:
                if n.name.startswith("openpibo"):
                    if n.name not in modules:
                        return f"없는 모듈 {n.name}"
                elif n.name.split(".")[0] not in STDLIB:
                    return f"없는 모듈 {n.name}"
                imported.add(n.asname or n.name.split(".")[0])
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


def inline_problem(snippet: str, api: dict) -> str | None:
    """글 속 코드 조각: `device.eye_on(255, 0, 0)` 같은 부르기와 import만 대조 (낱말 하나는 넘어감)."""
    s = snippet.strip()
    if s.startswith(("import ", "from ")):
        return code_problem(s, api)
    m = re.match(r"^(\w+)\.(\w+)\s*\(", s)
    if not m:
        return None
    obj, method = m.groups()
    if obj in api["instances"]:
        return None if method in api["methods"].get(api["instances"][obj], []) else \
            f"없는 메서드 {api['instances'][obj]}.{method}"
    if obj in STDLIB or obj == "utils":
        return None
    return f"없는 모듈 {obj}"


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
    if answer.count("```") % 2:
        return "코드 블록이 닫히지 않음"
    for code in CODE.findall(answer):
        if why := code_problem(code, api):
            return why
    for snippet in INLINE.findall(text):
        if why := inline_problem(snippet, api):
            return why
    for name in QUOTED.findall(text):
        if not known_block(name, api):
            return f"없는 블록 「{name[:20]}」"
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
