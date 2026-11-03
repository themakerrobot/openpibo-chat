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
#: 선생에게 주는 글의 판: 글을 바꾸면 올려서 기록된 답을 버리고 다시 묻게 함 (만들기·검사 따로)
VERSION = 3
CHECK_VERSION = 5
#: openpibo 말고 써도 되는 파이썬 모듈
STDLIB = {"time", "random", "math", "os", "datetime", "json"}

CODE_STYLE = """더메이커는 학생과 친구처럼 이야기하는 교육용 AI 코딩 도우미예요.
- 친근한 존댓말("~해요")로 말해요. 학생을 "-시-"로 높이지 않아요 ("하셨어요" 대신 "했어요").
- 설명은 2~4문장으로 짧고 쉽게. 어려운 말은 풀어서 말해요.
- 코드는 꼭 필요할 때만 ```python 으로 감싸서 10줄 안쪽으로 보여 줘요. 코드 첫머리에 준비 코드(import와 객체 만들기)를 넣어요.
- 이모지는 쓰지 않아요. 블록은 「」로 감싸 블록 목록의 글자로 말해요. 칸은 빼도 돼요 (예: 「오디오 크기로 재생하기」).
- 설명과 블록 목록에 없는 블록, 함수, 인자, 값, 모듈은 절대 지어내지 않아요.
- 돌려주는 값의 종류(True/False, 숫자, 글자 …)와 값의 범위는 설명에 적혀 있을 때만 말해요. 모르면 "값을 출력해
  확인해 보세요"처럼 말해요.
- 함수를 부를 때는 설명에 있는 인자 이름과 순서, 값의 모양(숫자, 글자, 쉼표로 구분한 글자 …) 그대로 써요."""

GEN = """아래는 {device} 블록 코딩 도구의 {what} 설명이에요. 이 설명만 근거로, {device}로 블록 코딩과 파이썬을
배우는 초·중·고 학생이 물을 만한 질문과 코딩 도우미 더메이커의 대답을 {n}쌍 만들어요.

{style}

질문을 골고루 섞어요:
1. 이게 무엇을 하는지
2. {how}
3. 하고 싶은 일로 묻기 (예: "눈을 빨갛게 하려면 어떻게 해?") — 이 {what}{ro} 할 수 있는 일만
4. 값의 범위, 넣는 순서, 자주 하는 실수 (설명에 있을 때만)
5. 짧은 예제 코드나 블록 순서 (다른 블록을 함께 쓸 때는 아래 블록 목록에 있는 블록만)
질문은 모두 이 {what}에 대한 것만 만들어요 (다른 블록은 함께 쓰는 예에서만, 블록 목록에 있는 것만).
질문은 혼자 읽어도 무엇을 묻는지 알 수 있게 써요: 블록 이름이나 함수 이름, 또는 하고 싶은 일을 꼭 넣어요.
"이 블록", "이거", "그 함수", "여기"처럼 가리키는 말로만 묻지 않아요 (예: "이 블록 뭐야?" 대신 "사진 찍기 블록은 뭐야?").
질문은 학생 말투(반말, 짧게)로, 서로 다른 말로 써요. 설명이 모자라 대답할 수 없는 질문은 만들지 않아요.
설명이 짧으면 문답을 {n}쌍보다 적게 만들어도 돼요.
JSON 하나로만 답해요. 출력 형식: {{"pairs": [{{"q": "학생 질문", "a": "더메이커 대답"}}]}}

[설명]
{doc}

[{device} 블록 상자에 있는 블록 (카테고리: 블록 이름)]
{blocks}"""

CHECK = """아래 설명을 근거로 만든 코딩 도우미 문답 하나를 검사해요. 작은 모델이 그대로 배우므로 틀린 사실이 있으면
낮게 줘요.

{style}

[설명]
{doc}

[문답]
학생: {q}
더메이커: {a}

대답을 설명과 대조해요. 이 블록·함수에만 해당하는 사실(값, 값의 범위·형식, 돌려주는 값, 고를 수 있는 값,
파일·폴더 이름, 동작, 오류가 나는 상황) 가운데 설명에서 찾을 수 없는 것만 unsupported에 짧게 적어요 (한 개에 20자 안쪽).
unsupported에 넣지 않는 것 (모두 맞는 것으로 봐요):
- 일반 상식: RGB 색 섞기(255, 0, 0은 빨강), 숫자·글자, 파이썬 기본 문법(print, 변수, if, for, while, 리스트)
- 다른 블록이나 openpibo 함수를 함께 쓰는 예: 그 블록·함수가 실제로 있는지와 인자 수는 따로 확인했어요
- 「」로 부른 블록 이름, 설명의 칸 이름·칸 수·코드 모양에서 바로 알 수 있는 것
- "확인해 보세요", "연결해서 쓰세요" 같은 일반적인 조언과 격려
JSON 하나로만 답해요.
- unsupported: 설명에서 찾을 수 없는 구체적인 사실들 (없으면 [])
- grounded: 대답이 설명과 맞는가 (1~5). unsupported가 있으면 2점 이하
- clear: 학생이 알아듣게 짧고 쉬운가, 질문에 맞게 대답했는가 (1~5)
- on_topic: 질문이 이 설명의 블록·함수에 대한 것인가 (true/false)
- code_ok: 코드가 있으면 인자의 이름·순서·값 모양까지 설명대로라서 그대로 돌릴 수 있는가 (코드가 없으면 true)
출력 형식: {{"unsupported": [], "grounded": 0, "clear": 0, "on_topic": true, "code_ok": true}}"""

PASS = 4
CODE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)
INLINE = re.compile(r"`([^`\n]+)`")
QUOTED = re.compile(r"「([^」]+)」|〈([^〉]+)〉|『([^』]+)』")
#: 학생을 높이는 말 ("-시-"): 친구 말투가 아님 ("~하세요"처럼 부탁하는 끝은 둠)
HONORIFIC = re.compile(r"[가-힣](시면|셨|십시|시겠|시죠|신다)|하신|보신|주신|계신")
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


def check_prompts(u: dict, pairs: list[dict]) -> list[str]:
    """문답마다 따로 검사 (한꺼번에 하면 생각이 길어져 잘림). 블록 이름은 글자 검사가 대조하므로 목록은 넣지 않음."""
    return [CHECK.format(style=CODE_STYLE, doc=unit_doc(u), q=p["q"], a=p["a"]) for p in pairs]


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
            methods = api["methods"].get(alias, {})
            if node.func.attr not in methods:
                return f"없는 메서드 {alias}.{node.func.attr}"
            if why := args_problem(node, methods[node.func.attr], f"{alias}.{node.func.attr}"):
                return why
    return None


def args_problem(call: ast.Call, params, name: str) -> str | None:
    """부를 때 인자가 그 메서드의 인자와 맞는지 (개수, 이름). params가 없으면(예전 목록) 넘어감."""
    if not isinstance(params, dict):
        return None
    if any(isinstance(a, ast.Starred) for a in call.args) or any(k.arg is None for k in call.keywords):
        return None                                           # *args, **kwargs로 넘기면 셀 수 없음
    if not params["varargs"] and len(call.args) > params["positional"]:
        return f"인자가 많음 {name}"
    for k in call.keywords:
        if k.arg not in params["names"] and not params["varkw"]:
            return f"없는 인자 {name}({k.arg}=)"
    given = {k.arg for k in call.keywords}
    missing = [n for n in params["names"][len(call.args):params["required"]] if n not in given]
    if missing:
        return f"인자가 모자람 {name}({missing[0]})"
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
        try:                                          # 조각이 온전한 부르기면 인자까지 대조
            ast.parse(s)
            return code_problem(s, api)
        except SyntaxError:
            return None if method in api["methods"].get(api["instances"][obj], {}) else \
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
    if bad := HONORIFIC.search(text):
        return f"학생을 높임 '{bad.group()}'"
    for groups in QUOTED.findall(text):
        name = next(g for g in groups if g)
        if not known_block(name, api):
            return f"없는 블록 「{name[:20]}」"
    return None


#: 앞뒤 맥락이 있어야 알 수 있는 질문 ("이 블록 뭐야?"): 쓸 때는 질문으로 설명을 찾으므로 못 씀
VAGUE = re.compile(r"(이|그|저)\s*(블록|함수|코드|칸|거|것|기능)|이거|이건|이걸|그거|그건|여기|이게")


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
        if VAGUE.search(q):
            why["가리키는 말로 된 질문"] = why.get("가리키는 말로 된 질문", 0) + 1
            continue
        if problem := answer_problem(a, api):
            key = " ".join(problem.split(" ")[:2]) if problem.startswith(("없는", "코드", "인자", "학생을")) else problem
            why[key] = why.get(key, 0) + 1
            continue
        seen.add(q)
        kept.append({"q": q, "a": a.strip()})
    return kept, why


def decide_check_one(text: str) -> dict | None:
    """문답 하나의 검사 답 → 점수 (못 읽으면 None, 깨진 JSON은 값마다 읽어 냄)."""
    d = first_json(text)
    if isinstance(d, dict) and "grounded" in d:
        return d
    return decide_check('{"marks": [' + text, 1)[0] if '"unsupported"' in text else None


def decide_check(text: str, n: int) -> list[dict | None]:
    """검사 답 → 문답마다 점수 (못 읽으면 None). JSON 전체가 깨졌으면(설명에 없는 말 안의 따옴표 등) 문답마다
    하나씩 읽어 냄: 앞에서부터 읽힌 만큼은 쓰고, 끝이 잘린 뒤쪽만 None."""
    d = first_json(text)
    marks = d.get("marks") if isinstance(d, dict) else None
    if isinstance(marks, list):                  # 온전한 JSON인데 수가 다르면 어느 문답 것인지 몰라 모두 버림
        return [m if isinstance(m, dict) else None for m in marks] if len(marks) == n else [None] * n
    from teacher import after_think

    body = after_think(text) or text
    parts = re.split(r'(?=\{\s*"unsupported")', body)[1:]
    out: list[dict | None] = []
    for part in parts[:n]:
        nums = {k: re.search(rf'"{k}"\s*:\s*(\d)', part) for k in ("grounded", "clear")}
        bools = {k: re.search(rf'"{k}"\s*:\s*(true|false)', part) for k in ("on_topic", "code_ok")}
        if not all(nums.values()) or not bools["code_ok"]:
            out.append(None)
            continue
        empty = re.match(r'\{\s*"unsupported"\s*:\s*\[\s*\]', part) is not None
        out.append({"unsupported": [] if empty else ["(읽어 낸 답: 설명에 없는 말 있음)"],
                    **{k: int(v.group(1)) for k, v in nums.items()},
                    **{k: v.group(1) == "true" for k, v in bools.items() if v}})
    return out + [None] * (n - len(out))


def passes(m: dict | None) -> str | None:
    """점수가 기준을 넘으면 None, 아니면 까닭."""
    if m is None:
        return "검사 답을 못 읽음"
    if m.get("unsupported"):
        return "설명에 없는 말"
    if m.get("on_topic") is False:
        return "다른 블록 이야기"
    for k in ("grounded", "clear"):
        v = m.get(k)
        if not isinstance(v, (int, float)) or isinstance(v, bool) or v < PASS:
            return f"{k} {v}점"
    if m.get("code_ok") is not True:
        return "코드가 틀림"
    return None


def dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)
