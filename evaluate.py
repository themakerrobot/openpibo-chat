"""모델마다 같은 채점용 질문에 답하게 하고, 선생 모델이 대화를 채점합니다.

    python evaluate.py gen --model runs/qwen3-0.6b/final --name qwen3-0.6b-파이보
    python evaluate.py gguf --gguf out/pibo-qwen3-0.6b-Q8_0.gguf --name qwen3-0.6b-Q8_0
    python evaluate.py judge          # 답이 모인 모델 전부 채점
    python evaluate.py report         # results/eval.md
    python evaluate.py ... --set school   # 초·중·고 질문으로 (results-school/)

대화: 아이의 첫마디에 대한 대답을 선생 모델이 네 가지로 1~5점 채점합니다. 따로
선생이 생각을 켜고 틀린 사실과 정설을 얼버무린 대답을 찾습니다 (생각을 끈 채점은
선생 자신이 데이터에 넣은 것과 같은 실수를 못 잡았습니다).
도구: 불러야 할 도구와 인자를 맞혔는지(정답률), 그냥 대화인데 도구를 부른 비율
(잘못 부름)을 셉니다.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

from make_data import load_teacher, thinking_params
from prompts import after_think, first_json, style_for, tools_for
from tools import SYSTEM, TOOLS, check_call, same_call

HELDOUT = Path("data/heldout.jsonl")
RESULTS = Path("results")
#: filter_tools.py가 적은 채점용 도구 문제 표시: bad(억지라 뺌), names_only(도구 이름만 봄)
MARKS = Path("data/heldout-marks.json")
#: 채점용 질문 묶음: (답과 점수를 두는 곳, 대화 질문을 바꿔 넣을 파일).
#: school은 어린이 채점용 질문의 대화 질문을 같은 자리에서 초·중·고 대화로 바꾸고 (도구
#: 문제와 그 표시는 그대로), 시스템 글은 지금 것(tools.SYSTEM)으로 묻습니다.
SETS = {"kids": (Path("results"), None),
        "school": (Path("results-school"), Path("data/heldout-school.jsonl"))}
SCHOOL_CHAT: Path | None = None
CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
#: Qwen3.5의 도구 호출: <function=이름> 안에 <parameter=인자>값</parameter>
FUNC = re.compile(r"<function=([^>\s]+)>(.*?)</function>", re.DOTALL)
PARAM = re.compile(r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", re.DOTALL)


def use_set(name: str) -> None:
    global RESULTS, SCHOOL_CHAT
    RESULTS, SCHOOL_CHAT = SETS[name]


def questions():
    school = iter(SCHOOL_CHAT.read_text().splitlines()) if SCHOOL_CHAT else None
    out = []
    for line in HELDOUT.read_text().splitlines():
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
                    "reference": expect.get("content", ""), "grade": ex.get("grade")})
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
    for with_tools in (False, True):
        idx = [i for i, q in enumerate(qs) if q["tools"] == with_tools]
        if not idx:
            continue
        # 도구 형식이 없는 모델(젬마3)은 도구 없이 묻습니다
        outs = llm.chat([qs[i]["messages"] for i in idx], params,
                        tools=TOOLS if with_tools and not no_tools else None,
                        chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
        for i, o in zip(idx, outs, strict=True):
            answers[i] = o.outputs[0].text
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"answers-{name}.json").write_text(json.dumps(
        {"model": model, "no_tools": no_tools, "answers": answers}, ensure_ascii=False,
        indent=1))


def gen_gguf(gguf: str, name: str, kv: str | None = None):
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
            body = {"messages": qs[i]["messages"], "temperature": 0, "max_tokens": 200}
            if qs[i]["tools"]:
                body["tools"] = TOOLS
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
        {"model": gguf, "no_tools": False, "answers": answers}, ensure_ascii=False, indent=1))


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
    judge_text, fact_text = ((for_school(JUDGE), for_school(FACT)) if SCHOOL_CHAT
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


def report() -> str:
    qs = questions()
    marked = json.loads(MARKS.read_text()) if MARKS.exists() else {}
    bad, loose = set(marked.get("bad", [])), set(marked.get("names_only", []))
    rows = [("| 모델 | 대화 점수 (5점 만점) | 한국어 | 알맞음 | 짧음 | 사실 | 틀린 사실 | 얼버무림 | "
             "도구 정답률 | 춤에 덤 눈 색 | 도구 형식 오류 | 잘못 부름 | 평균 글자 수 |"),
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
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
        rows.append(f"| {name} | {total:.2f} | {avg('korean'):.2f} | {avg('fit'):.2f} | "
                    f"{avg('short'):.2f} | {avg('true'):.2f} | {rate('wrong')} | "
                    f"{rate('dodged')} | {right} | {extra} | "
                    f"{wrong_fmt} | {false_call} | {length:.0f} |")
    text = "\n".join(rows) + "\n"
    n_tool = sum(q["kind"] == "tool" for q in qs)
    text += ("\n틀린 사실·얼버무림: 선생이 생각을 켜고 본 대화 대답의 비율 (낮을수록 좋음)."
             " 사실 칸(1~5점)은 생각을 끈 채점이라 이런 실수를 잘 못 잡습니다.\n")
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
    p.add_argument("--no-tools", action="store_true", help="도구 형식이 없는 모델 (젬마3)")
    p.add_argument("--set", choices=list(SETS), default="kids")
    args = p.parse_args()
    use_set(args.set)
    if args.what == "gen":
        gen(args.model, args.name, args.no_tools)
    elif args.what == "gguf":
        gen_gguf(args.gguf, args.name, args.kv)
    elif args.what == "judge":
        judge()
    else:
        print(report())


if __name__ == "__main__":
    main()
