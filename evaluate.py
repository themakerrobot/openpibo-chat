"""모델마다 같은 채점용 질문에 답하게 하고, 선생 모델이 대화를 채점합니다.

    python evaluate.py gen --model runs/qwen3-0.6b/final --name qwen3-0.6b-파이보
    python evaluate.py gguf --gguf out/pibo-qwen3-0.6b-Q8_0.gguf --name qwen3-0.6b-Q8_0
    python evaluate.py judge          # 답이 모인 모델 전부 채점
    python evaluate.py report         # results/eval.md

대화: 아이의 첫마디에 대한 대답을 선생 모델이 네 가지로 1~5점 채점합니다.
도구: 불러야 할 도구와 인자를 맞혔는지(정답률), 그냥 대화인데 도구를 부른 비율
(잘못 부름)을 셉니다.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

from make_data import load_teacher
from prompts import STYLE, first_json, tools_for
from tools import TOOLS, check_call, same_call

HELDOUT = Path("data/heldout.jsonl")
RESULTS = Path("results")
CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def questions():
    out = []
    for line in HELDOUT.read_text().splitlines():
        ex = json.loads(line)
        msgs = ex["messages"]
        user_at = next(i for i, m in enumerate(msgs) if m["role"] == "user")
        expect = msgs[user_at + 1]
        out.append({"kind": ex["kind"], "tools": tools_for(ex) is not None,
                    "messages": msgs[:user_at + 1],
                    "calls": [c["function"] for c in expect.get("tool_calls", [])],
                    "reference": expect.get("content", "")})
    return out


def parse_calls(text: str) -> list[dict]:
    calls = []
    for m in CALL.finditer(text):
        try:
            c = json.loads(m.group(1))
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


def gen_gguf(gguf: str, name: str):
    """GGUF를 PC의 llama-server(CPU)로 같은 질문에 답하게 합니다: 파이보와 같은 길
    (llama.cpp의 템플릿과 도구 호출 읽기)로 묻고, 양자화로 얼마나 나빠지는지 봅니다."""
    from export import post, server

    import urllib.error

    qs = questions()
    answers = [""] * len(qs)
    broken = 0
    with server(Path(gguf), 8397) as url:
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


def judge():
    from vllm import SamplingParams

    qs = questions()
    files = sorted(RESULTS.glob("answers-*.json"))
    llm = load_teacher()
    params = SamplingParams(temperature=0.0, max_tokens=80)
    for f in files:
        scored = f.with_name(f.name.replace("answers-", "scores-"))
        if scored.exists():
            continue
        answers = json.loads(f.read_text())["answers"]
        idx = [i for i, q in enumerate(qs) if q["kind"] == "chat"]
        convs = [[{"role": "user", "content": JUDGE.format(
            style=STYLE, user=qs[i]["messages"][-1]["content"], reply=say(answers[i]) or "(말 없음)")}]
            for i in idx]
        outs = llm.chat(convs, params, chat_template_kwargs={"enable_thinking": False},
                        use_tqdm=False)
        marks = {}
        for i, o in zip(idx, outs, strict=True):
            d = first_json(o.outputs[0].text) or {}
            marks[i] = {k: d.get(k) for k in ("korean", "fit", "short", "true")}
        scored.write_text(json.dumps(marks, ensure_ascii=False))
        print(f"채점: {f.name}", flush=True)


def report() -> str:
    qs = questions()
    rows = [("| 모델 | 대화 점수 (5점 만점) | 한국어 | 알맞음 | 짧음 | 사실 | 도구 정답률 | "
             "도구 형식 오류 | 잘못 부름 | 평균 글자 수 |"),
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
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
        tool_q = [i for i, q in enumerate(qs) if q["kind"] == "tool"]
        chat_tools = [i for i, q in enumerate(qs) if q["kind"] == "chat" and q["tools"]]
        if data.get("no_tools"):
            right = wrong_fmt = false_call = "도구 없음"
        else:
            hits = fmt = 0
            for i in tool_q:
                got = parse_calls(answers[i])
                fmt += any(check_call(c["name"], c["arguments"]) for c in got) or not got
                hits += bool(got) and len(got) == len(qs[i]["calls"]) and all(
                    same_call(a, b) for a, b in zip(got, qs[i]["calls"], strict=True))
            fc = sum(bool(parse_calls(answers[i])) for i in chat_tools)
            right = f"{hits / max(len(tool_q), 1):.0%}"
            wrong_fmt = f"{fmt / max(len(tool_q), 1):.0%}"
            false_call = f"{fc / max(len(chat_tools), 1):.0%}"
        chat_q = [i for i, q in enumerate(qs) if q["kind"] == "chat"]
        length = statistics.mean(len(say(answers[i])) for i in chat_q) if chat_q else 0
        rows.append(f"| {name} | {total:.2f} | {avg('korean'):.2f} | {avg('fit'):.2f} | "
                    f"{avg('short'):.2f} | {avg('true'):.2f} | {right} | {wrong_fmt} | "
                    f"{false_call} | {length:.0f} |")
    text = "\n".join(rows) + "\n"
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "eval.md").write_text(text)
    return text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["gen", "gguf", "judge", "report"])
    p.add_argument("--model")
    p.add_argument("--gguf")
    p.add_argument("--name")
    p.add_argument("--no-tools", action="store_true", help="도구 형식이 없는 모델 (젬마3)")
    args = p.parse_args()
    if args.what == "gen":
        gen(args.model, args.name, args.no_tools)
    elif args.what == "gguf":
        gen_gguf(args.gguf, args.name)
    elif args.what == "judge":
        judge()
    else:
        print(report())


if __name__ == "__main__":
    main()
