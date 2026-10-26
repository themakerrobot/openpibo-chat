"""모델마다 같은 채점용 질문에 답하게 하고, 선생 모델이 대화를 채점합니다.

    python evaluate.py gen --model runs/qwen3-0.6b/final --name qwen3-0.6b-파이보
    python evaluate.py gguf --gguf out/pibo-qwen3-0.6b-Q8_0.gguf --name qwen3-0.6b-Q8_0
    python evaluate.py judge          # 답이 모인 모델 전부 채점
    python evaluate.py report         # results/eval.md
    python evaluate.py ... --set school   # 초·중·고 질문으로 (results-school/)
    python evaluate.py ... --set tools    # 초·중·고 도구 질문으로 (results-tools/, 채점 없이 셈)
    python evaluate.py ... --set english  # 영어 대화·영어 도구·영어 공부 질문으로 (results-english/)

대화: 아이의 첫마디에 대한 대답을 선생 모델이 네 가지로 1~5점 채점합니다. 따로
선생이 생각을 켜고 틀린 사실과 정설을 얼버무린 대답을 찾습니다 (생각을 끈 채점은
선생 자신이 데이터에 넣은 것과 같은 실수를 못 잡았습니다).
도구: 불러야 할 도구와 인자를 맞혔는지(정답률), 그냥 대화인데 도구를 부른 비율
(잘못 부름)을 셉니다.
tools 묶음은 초·중·고 학생 말로 첫마디 도구, 이야기하다가 중간에 부탁한 도구, 도구를
부른 뒤 이어지는 이야기(부르면 안 됨), 도구와 비슷한 말(부르면 안 됨)을 따로 셉니다.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path

from make_data import load_teacher, thinking_params
from prompts import after_think, first_json, style_for, tools_for
from tools import SYSTEM, TOOLS, check_call, same_call, system_for, tool_list

HELDOUT = Path("data/heldout.jsonl")
RESULTS = Path("results")
#: filter_tools.py가 적은 채점용 도구 문제 표시: bad(억지라 뺌), names_only(도구 이름만 봄)
MARKS = Path("data/heldout-marks.json")
#: 채점용 질문 묶음: (답과 점수를 두는 곳, 대화 질문을 바꿔 넣을 파일).
#: school은 어린이 채점용 질문의 대화 질문을 같은 자리에서 초·중·고 대화로 바꾸고 (도구
#: 문제와 그 표시는 그대로), 시스템 글은 지금 것(tools.SYSTEM)으로 묻습니다.
SETS = {"kids": (Path("results"), None),
        "school": (Path("results-school"), Path("data/heldout-school.jsonl")),
        "tools": (Path("results-tools"), None),
        "english": (Path("results-english"), None)}
SCHOOL_CHAT: Path | None = None
#: english 묶음의 질문 (make_data.py --english --english-tool --learn)과 filter_tools.py의 표시
EN_HELD = Path("data/heldout-english.jsonl")
EN_MARKS = Path("data/heldout-english-marks.json")
KIDS_MARKS = MARKS
ENGLISH = False
HANGUL = re.compile(r"[가-힣]")
#: tools 묶음의 질문 (make_data.py --tool --mixed --near)과 filter_tools.py가 적은 표시
TOOL_HELD = Path("data/heldout-school-tools.jsonl")
TOOL_MARKS = Path("data/heldout-school-tools-marks.json")
TOOL_SET = False
#: tools 묶음의 질문 종류
TOOL_KINDS = {"first": "첫마디 도구 정답률", "mid": "대화 중 도구 정답률",
              "after": "도구 뒤 잘못 부름", "near": "비슷한 말에 잘못 부름",
              "unavail": "없는 도구 부름", "name": "이름 맞힘"}
#: 맞힌 비율인 종류 (나머지는 부르면 안 되는데 부른 비율이라 낮을수록 좋음)
TOOL_RIGHT = {"first", "mid", "name"}
#: 다른 기기 이름으로 물을 때 (시스템 글의 이름으로 대답하는지)
NAME_CHECK = [("파이브레인", "PiBrain"), ("코코", "Coco"), ("다온", "Daon")]
NAME_QUESTIONS = ["너 누구야?", "이름이 뭐야?", "What's your name?"]
CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
#: Qwen3.5의 도구 호출: <function=이름> 안에 <parameter=인자>값</parameter>
FUNC = re.compile(r"<function=([^>\s]+)>(.*?)</function>", re.DOTALL)
PARAM = re.compile(r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", re.DOTALL)


def use_set(name: str) -> None:
    global RESULTS, SCHOOL_CHAT, TOOL_SET, ENGLISH, MARKS
    RESULTS, SCHOOL_CHAT = SETS[name]
    TOOL_SET, ENGLISH = name == "tools", name == "english"
    MARKS = EN_MARKS if ENGLISH else KIDS_MARKS


def same_language(q: dict, reply: str) -> bool:
    """물은 말로 대답했는지: 영어 대화 질문이면 한글 없이 영어로, 그 밖에는 한국어로."""
    text = say(reply)
    if q.get("lang") == "en":
        return bool(re.search(r"[A-Za-z]", text)) and not HANGUL.search(text)
    return bool(HANGUL.search(text))


def tool_questions():
    """tools 묶음: 파이보가 말하는 차례마다 그때까지의 대화를 묻습니다. kind는 first(도구 예시의
    첫마디), mid(대화 중 도구를 부탁한 차례), after(도구를 부른 뒤의 이야기 차례),
    near(도구와 비슷한 이야기). 선생이 억지라고 본 줄은 빼고, 인자가 말로 안 정해지는 줄은
    도구 이름만 봅니다."""
    marks = json.loads(TOOL_MARKS.read_text()) if TOOL_MARKS.exists() else {}
    bad, loose = set(marks.get("bad", [])), set(marks.get("names_only", []))
    out = []
    for n, line in enumerate(TOOL_HELD.read_text().splitlines()):
        if n in bad:
            continue
        ex = json.loads(line)
        msgs, called = ex["messages"], False
        for j, m in enumerate(msgs):
            if m["role"] != "assistant" or msgs[j - 1]["role"] != "user":
                continue
            calls = [c["function"] for c in m.get("tool_calls") or []]
            kind = (("first" if ex["kind"] == "tool" else "mid") if calls else
                    "near" if ex.get("near") else "after" if called else None)
            if kind:
                out.append({"kind": kind, "tools": True, "messages": msgs[:j], "calls": calls,
                            "names_only": n in loose, "reference": m.get("content", ""),
                            "grade": ex.get("grade")})
            called = called or bool(calls)
    # 그 도구가 없는 기기: 첫마디 도구 질문에서 부를 도구를 목록에서 빼고 묻기 (부르면 틀림)
    for q in [q for q in out if q["kind"] == "first" and len(q["calls"]) == 1]:
        name = q["calls"][0]["name"]
        out.append(dict(q, kind="unavail", calls=[],
                        tool_list=tool_list([n for n in TOOLS_NAMES if n != name])))
    for ko, en in NAME_CHECK:
        for text in NAME_QUESTIONS:
            out.append({"kind": "name", "tools": True, "calls": [], "names_only": False,
                        "reference": "", "grade": None, "expect_name": [ko, en],
                        "tool_list": tool_list(device=ko),
                        "messages": [{"role": "system", "content": system_for(ko)},
                                     {"role": "user", "content": text}]})
    return out


TOOLS_NAMES = [t["function"]["name"] for t in TOOLS]


def questions():
    if TOOL_SET:
        return tool_questions()
    school = iter(SCHOOL_CHAT.read_text().splitlines()) if SCHOOL_CHAT else None
    out = []
    for line in (EN_HELD if ENGLISH else HELDOUT).read_text().splitlines():
        ex = json.loads(line)
        if school is not None:
            if ex["kind"] == "chat":
                ex = json.loads(next(school))
            for m in ex["messages"]:
                if m["role"] == "system":
                    m["content"] = SYSTEM
        msgs = ex["messages"]
        user_at = next(i for i, m in enumerate(msgs) if m["role"] == "user")
        expect = msgs[user_at + 1]
        out.append({"kind": ex["kind"], "tools": tools_for(ex) is not None,
                    "messages": msgs[:user_at + 1],
                    "calls": [c["function"] for c in expect.get("tool_calls", [])],
                    "reference": expect.get("content", ""), "grade": ex.get("grade"),
                    "lang": ex.get("lang")})
    return out


def parse_calls(text: str) -> list[dict]:
    calls = []
    for m in CALL.finditer(text):
        body = m.group(1)
        if f := FUNC.search(body):           # Qwen3.5 (XML)
            args = {}
            for k, v in PARAM.findall(f.group(2)):
                try:
                    args[k] = json.loads(v)
                except json.JSONDecodeError:
                    args[k] = v
            calls.append({"name": f.group(1), "arguments": args})
            continue
        try:
            c = json.loads(body)
        except json.JSONDecodeError:
            calls.append({"name": "?", "arguments": {}})
            continue
        args = c.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        calls.append({"name": c.get("name"), "arguments": args})
    return calls


def tool_match(got: list[dict], want: list[dict], names_only: bool = False):
    """(맞았는지, 춤에 덤으로 눈 색을 바꿨는지). 정답에 눈 색이 없는 동작 문제에서
    덤으로 부른 eye_on은 빼고 보고 따로 셉니다. names_only면 도구 이름만 봅니다."""
    extra = False
    if any(c["name"] == "set_motion" for c in want) and \
            not any(c["name"] == "eye_on" for c in want):
        kept = [c for c in got if c["name"] != "eye_on"]
        extra, got = len(kept) < len(got), kept
    if not got or len(got) != len(want):
        return False, False
    ok = all(a["name"] == b["name"] if names_only else same_call(a, b)
             for a, b in zip(got, want, strict=True))
    return ok, ok and extra


def openai_messages(messages: list[dict]) -> list[dict]:
    """학습 데이터의 메시지 → OpenAI 형식 (도구 인자는 JSON 글, 호출마다 id): llama-server는
    인자가 글이 아니면 받지 않습니다. vLLM도 같은 형식을 받아 템플릿에 맞게 되돌립니다."""
    out, ids = [], []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            ids = [f"call_{len(out)}_{k}" for k in range(len(m["tool_calls"]))]
            m["tool_calls"] = [{"id": i, "type": "function", "function": {
                "name": c["function"]["name"],
                "arguments": json.dumps(c["function"].get("arguments") or {}, ensure_ascii=False)}}
                for i, c in zip(ids, m["tool_calls"], strict=True)]
        elif m["role"] == "tool" and ids:
            m["tool_call_id"] = ids.pop(0)
        out.append(m)
    return out


def say(text: str) -> str:
    """도구 호출과 생각 표시를 뺀, 소리로 나갈 말."""
    text = CALL.sub("", text)
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def gen(model: str, name: str, no_tools: bool):
    from vllm import LLM, SamplingParams

    qs = questions()
    llm = LLM(model, max_model_len=4096, gpu_memory_utilization=0.85)
    params = SamplingParams(temperature=0.0, max_tokens=200)
    answers = [None] * len(qs)
    # 같은 도구 목록끼리 묶어 묻습니다 (도구 없음, 전체, 기기마다 일부)
    groups: dict = {}
    for i, q in enumerate(qs):
        tools = (q.get("tool_list") or TOOLS) if q["tools"] else None
        groups.setdefault(json.dumps(tools, ensure_ascii=False), (tools, []))[1].append(i)
    for tools, idx in groups.values():
        # 도구 형식이 없는 모델(젬마3)은 도구 없이 묻습니다
        outs = llm.chat([openai_messages(qs[i]["messages"]) for i in idx], params,
                        tools=tools if not no_tools else None,
                        chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
        for i, o in zip(idx, outs, strict=True):
            answers[i] = o.outputs[0].text
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"answers-{name}.json").write_text(json.dumps(
        {"model": model, "no_tools": no_tools, "answers": answers}, ensure_ascii=False,
        indent=1))


def gen_gguf(gguf: str, name: str, kv: str | None = None, no_tools: bool = False):
    """GGUF를 PC의 llama-server(CPU)로 같은 질문에 답하게 합니다: 파이보와 같은 길
    (llama.cpp의 템플릿과 도구 호출 읽기)로 묻고, 양자화로 얼마나 나빠지는지 봅니다."""
    from export import post, server

    import urllib.error

    qs = questions()
    answers = [""] * len(qs)
    broken = 0
    with server(Path(gguf), 8397, kv) as url:
        # 도구 목록이 있는 질문끼리 이어서 물어야 서버가 앞부분을 다시 읽지 않습니다
        order = sorted(range(len(qs)), key=lambda i: qs[i]["tools"])
        for n, i in enumerate(order, 1):
            if n % 100 == 0:
                print(f"{n}/{len(qs)}", flush=True)
            body = {"messages": openai_messages(qs[i]["messages"]), "temperature": 0,
                    "max_tokens": 200}
            if qs[i]["tools"] and not no_tools:   # 도구 없는 대화 모델은 도구 목록 없이
                body["tools"] = qs[i].get("tool_list") or TOOLS
            try:
                msg = post(f"{url}/v1/chat/completions", body)["choices"][0]["message"]
            except urllib.error.HTTPError:   # 서버가 모델의 도구 호출을 못 읽음: 빈 답으로
                broken += 1
                continue
            text = msg.get("content") or ""
            for c in msg.get("tool_calls") or []:
                f = c["function"]
                args = f.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        pass
                text += "\n<tool_call>\n" + json.dumps(
                    {"name": f.get("name"), "arguments": args}, ensure_ascii=False) + \
                    "\n</tool_call>"
            answers[i] = text
    if broken:
        print(f"서버가 읽지 못한 답 {broken}개는 빈 답으로 둠", flush=True)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"answers-{name}.json").write_text(json.dumps(
        {"model": gguf, "no_tools": no_tools, "answers": answers}, ensure_ascii=False, indent=1))


JUDGE = """아래는 어린이용 교육 로봇 '파이보'에게 아이가 한 말과 파이보의 대답입니다.
{style}

아이: {user}
파이보: {reply}

파이보의 대답을 네 가지로 1~5점 채점해 JSON 하나로만 답하세요.
- korean: 한국어가 자연스러운가
- fit: 아이에게 알맞고 말뜻에 맞게 대답했는가
- short: 짧고 소리로 읽기 좋은가 (규칙을 지켰는가)
- true: 틀린 사실을 지어내지 않았는가
출력 형식: {{"korean": 0, "fit": 0, "short": 0, "true": 0}}"""


FACT = """아래는 어린이용 교육 로봇 '파이보'에게 아이가 한 말과 파이보의 대답입니다.
아이들은 파이보의 말을 그대로 믿고 배웁니다.

아이: {user}
파이보: {reply}

두 가지를 판단해 JSON 하나로만 답하세요.
- wrong: 틀린 사실(과학, 자연, 역사, 생활 상식, 숫자, 연대)이 하나라도 있으면 true
- dodged: 널리 받아들여진 사실이나 정설을 "아직 몰라요", "여러 가지 설이 있어요",
  "연구하고 있어요"로 피했으면 true. 정말 아무도 모르는 것을 모른다고 한 것은 false
사실을 말하지 않는 대답(인사, 위로, 되묻기)은 둘 다 false입니다.
출력 형식: {{"wrong": false, "dodged": false}}"""


def for_school(text: str) -> str:
    """채점 글을 초·중·고 질문용으로 (어린이 질문은 원래 글 그대로 채점해 점수가 이어지게)."""
    return (text.replace("어린이용 교육 로봇 '파이보'에게 아이가", "교육용 로봇 '파이보'에게 학생이")
            .replace("아이들은", "학생들은").replace("아이: {user}", "학생: {user}"))


def for_english(text: str) -> str:
    """영어 묶음의 채점 글: 한국어 대신 물은 말로 자연스럽게 대답했는지를 봅니다."""
    return for_school(text).replace(
        "- korean: 한국어가 자연스러운가",
        "- korean: 말이 자연스러운가 (학생이 영어로 말했으면 쉬운 영어로, 한국어로 물었으면 한국어로 "
        "대답해야 하며, 다른 말로 대답했으면 2점 이하)")


def fact_marks(texts: list[str]) -> list[dict]:
    """선생의 사실 판단 → [{"wrong": bool|None, "dodged": bool|None}] (못 읽으면 None)."""
    out = []
    for t in texts:
        d = first_json(after_think(t)) or {}
        out.append({k: d.get(k) if isinstance(d.get(k), bool) else None
                    for k in ("wrong", "dodged")})
    return out


def judge():
    from vllm import SamplingParams

    qs = questions()
    files = sorted(RESULTS.glob("answers-*.json"))
    llm = load_teacher()
    params = SamplingParams(temperature=0.0, max_tokens=80)
    idx = [i for i, q in enumerate(qs) if q["kind"] == "chat"]
    judge_text, fact_text = ((for_english(JUDGE), for_school(FACT)) if ENGLISH
                             else (for_school(JUDGE), for_school(FACT)) if SCHOOL_CHAT
                             else (JUDGE, FACT))
    for f in files:
        answers = json.loads(f.read_text())["answers"]
        pairs = [(qs[i]["messages"][-1]["content"], say(answers[i]) or "(말 없음)") for i in idx]
        scored = f.with_name(f.name.replace("answers-", "scores-"))
        if not scored.exists():
            convs = [[{"role": "user", "content": judge_text.format(
                style=style_for(qs[i]), user=u, reply=r)}] for i, (u, r) in zip(idx, pairs)]
            outs = llm.chat(convs, params, chat_template_kwargs={"enable_thinking": False},
                            use_tqdm=False)
            marks = {}
            for i, o in zip(idx, outs, strict=True):
                d = first_json(o.outputs[0].text) or {}
                marks[i] = {k: d.get(k) for k in ("korean", "fit", "short", "true")}
            scored.write_text(json.dumps(marks, ensure_ascii=False))
            print(f"채점: {f.name}", flush=True)
        facts = f.with_name(f.name.replace("answers-", "facts-"))
        if not facts.exists():
            convs = [[{"role": "user", "content": fact_text.format(user=u, reply=r)}]
                     for u, r in pairs]
            outs = llm.chat(convs, thinking_params(1536),
                            chat_template_kwargs={"enable_thinking": True}, use_tqdm=False)
            marks = fact_marks([o.outputs[0].text for o in outs])
            facts.write_text(json.dumps(dict(zip(idx, marks, strict=True)), ensure_ascii=False))
            print(f"사실 검사: {f.name}", flush=True)


def tool_report() -> str:
    """tools 묶음의 표: 종류마다 맞힌 비율 (after, near는 도구를 부른 비율이라 낮을수록 좋음)."""
    qs = questions()
    head = ["모델", *TOOL_KINDS.values(), "덤으로 더 부름", "도구 형식 오류"]
    rows = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    for f in sorted(RESULTS.glob("answers-*.json")):
        answers = json.loads(f.read_text())["answers"]
        if len(answers) != len(qs):          # 질문이 바뀐 뒤의 옛 답
            continue
        got = [parse_calls(a or "") for a in answers]

        def rate(kinds, test, got=got):
            idx = [i for i, q in enumerate(qs) if q["kind"] in kinds]
            return f"{sum(test(i) for i in idx) / len(idx):.0%}" if idx else "-"

        def right(i, got=got):
            return tool_match(got[i], qs[i]["calls"], qs[i]["names_only"])[0]

        def named(i):
            return any(n.lower() in say(answers[i]).lower() for n in qs[i]["expect_name"])

        cells = [rate({k}, right if k in ("first", "mid") else named if k == "name"
                      else (lambda i, got=got: bool(got[i]))) for k in TOOL_KINDS]
        cells.append(rate({"first", "mid"}, lambda i, got=got: len(got[i]) > len(qs[i]["calls"])))
        cells.append(rate({"first", "mid"}, lambda i, got=got: not got[i] or any(
            check_call(c["name"], c["arguments"]) for c in got[i])))
        rows.append(f"| {f.stem.removeprefix('answers-')} | " + " | ".join(cells) + " |")
    count = {k: sum(q["kind"] == k for q in qs) for k in TOOL_KINDS}
    text = "\n".join(rows) + "\n\n" + ", ".join(
        f"{TOOL_KINDS[k]}: 질문 {v}개" for k, v in count.items()) + (
        ". 도구 뒤·비슷한 말·없는 도구는 도구를 부른 비율이라 낮을수록 좋음 (없는 도구: 첫마디 도구"
        " 질문에서 그 도구를 목록에서 빼고 물음). 이름 맞힘: 다른 기기 이름(파이브레인 등)을 시스템 글에"
        " 넣고 '너 누구야?'를 물었을 때 그 이름으로 대답한 비율. 덤으로 더 부름: 청한 것보다 많이"
        " 부른 비율 (춤에 덤으로 부른 눈 색은 정답률에서는 빼고 봄).\n")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "eval.md").write_text(text)
    return text


def report() -> str:
    if TOOL_SET:
        return tool_report()
    qs = questions()
    marked = json.loads(MARKS.read_text()) if MARKS.exists() else {}
    bad, loose = set(marked.get("bad", [])), set(marked.get("names_only", []))
    rows = [(f"| 모델 | 대화 점수 (5점 만점) | {'자연스러움' if ENGLISH else '한국어'} | 알맞음 | 짧음 | "
             "사실 | 틀린 사실 | 얼버무림 | 도구 정답률 | 춤에 덤 눈 색 | 도구 형식 오류 | 잘못 부름 | "
             "평균 글자 수 | 같은 말로 대답 | 같은 대답 |"),
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for f in sorted(RESULTS.glob("answers-*.json")):
        name = f.stem.removeprefix("answers-")
        data = json.loads(f.read_text())
        answers = data["answers"]
        sf = f.with_name(f.name.replace("answers-", "scores-"))
        marks = json.loads(sf.read_text()) if sf.exists() else {}

        def avg(key, marks=marks):
            v = [m[key] for m in marks.values() if isinstance(m.get(key), (int, float))]
            return statistics.mean(v) if v else float("nan")

        total = statistics.mean([avg(k) for k in ("korean", "fit", "short", "true")]) \
            if marks else float("nan")
        ff = f.with_name(f.name.replace("answers-", "facts-"))
        facts = json.loads(ff.read_text()) if ff.exists() else {}

        def rate(key, facts=facts):
            v = [m[key] for m in facts.values() if isinstance(m.get(key), bool)]
            return f"{sum(v) / len(v):.0%}" if v else "-"

        tool_q = [i for i, q in enumerate(qs) if q["kind"] == "tool" and i not in bad]
        chat_tools = [i for i, q in enumerate(qs) if q["kind"] == "chat" and q["tools"]]
        if data.get("no_tools"):
            right = extra = wrong_fmt = false_call = "도구 없음"
        else:
            hits = fmt = extras = 0
            for i in tool_q:
                got = parse_calls(answers[i])
                fmt += any(check_call(c["name"], c["arguments"]) for c in got) or not got
                ok, more = tool_match(got, qs[i]["calls"], i in loose)
                hits += ok
                extras += more
            fc = sum(bool(parse_calls(answers[i])) for i in chat_tools)
            right = f"{hits / max(len(tool_q), 1):.0%}"
            extra = f"{extras / max(len(tool_q), 1):.0%}"
            wrong_fmt = f"{fmt / max(len(tool_q), 1):.0%}"
            false_call = f"{fc / max(len(chat_tools), 1):.0%}"
        chat_q = [i for i, q in enumerate(qs) if q["kind"] == "chat"]
        length = statistics.mean(len(say(answers[i])) for i in chat_q) if chat_q else 0
        same = (f"{sum(same_language(qs[i], answers[i]) for i in chat_q) / len(chat_q):.0%}"
                if chat_q else "-")
        # 서로 다른 질문에 글자 그대로 같은 대답을 한 비율 (늘 하던 말로 때우는지)
        shapes = Counter(re.sub(r"[\s.,!?~…'\"]", "", say(answers[i])) for i in chat_q)
        dup = (f"{sum(shapes[re.sub(r'[\s.,!?~…\'\"]', '', say(answers[i]))] > 1 for i in chat_q) / len(chat_q):.0%}"
               if chat_q else "-")
        rows.append(f"| {name} | {total:.2f} | {avg('korean'):.2f} | {avg('fit'):.2f} | "
                    f"{avg('short'):.2f} | {avg('true'):.2f} | {rate('wrong')} | "
                    f"{rate('dodged')} | {right} | {extra} | "
                    f"{wrong_fmt} | {false_call} | {length:.0f} | {same} | {dup} |")
    text = "\n".join(rows) + "\n"
    n_tool = sum(q["kind"] == "tool" for q in qs)
    text += ("\n틀린 사실·얼버무림: 선생이 생각을 켜고 본 대화 대답의 비율 (낮을수록 좋음)."
             " 사실 칸(1~5점)은 생각을 끈 채점이라 이런 실수를 잘 못 잡습니다.\n")
    text += ("\n같은 말로 대답: 영어로 말을 건 대화 질문에 한글 없이 영어로, 그 밖의 대화 질문에 한국어로"
             " 대답한 비율. 같은 대답: 서로 다른 질문에 글자 그대로 같은 대답을 한 비율 (낮을수록 좋음).\n")
    text += (f"\n도구 정답률: 춤 번호는 따지지 않음. 춤에 덤으로 부른 눈 색은 빼고 보고 따로 셈."
             + (f" 도구 문제 {n_tool}개 중 선생이 억지라고 본 {len(bad)}개는 빼고, 동작·색·"
                f"지역이 아이 말로 안 정해지는 {len(loose)}개는 도구 이름만 봄." if marked else "")
             + "\n")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "eval.md").write_text(text)
    return text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["gen", "gguf", "judge", "report"])
    p.add_argument("--model")
    p.add_argument("--gguf")
    p.add_argument("--kv", help="KV 캐시 양자화 (q8_0 등): 파이보에서 KV 캐시 메모리를 줄일 때")
    p.add_argument("--name")
    p.add_argument("--no-tools", action="store_true",
                   help="도구 목록 없이 묻기: 도구 형식이 없는 모델(젬마3), 도구를 뺀 대화 모델")
    p.add_argument("--set", choices=list(SETS), default="kids")
    args = p.parse_args()
    use_set(args.set)
    if args.what == "gen":
        gen(args.model, args.name, args.no_tools)
    elif args.what == "gguf":
        gen_gguf(args.gguf, args.name, args.kv, args.no_tools)
    elif args.what == "judge":
        judge()
    else:
        print(report())


if __name__ == "__main__":
    main()
