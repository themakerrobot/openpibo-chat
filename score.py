"""최종 모델과 예전 모델을 같은 질문으로 채점합니다 (results/score.md).

    python score.py answer --gguf out/themaker-0.8b-Q4_K_M.gguf --name themaker
    python score.py answer --gguf baseline/before-Q4_K_M.gguf --name before
    python score.py judge      # 모은 답을 선생 모델이 채점, 표를 씀
    python score.py report     # 표만 다시

질문 (학습에 쓰지 않은 것): data/heldout.jsonl (채점용 원본 data/source-heldout.jsonl을 학습 데이터와
같은 방법으로 바꿔 쓴 것, rewrite.py)
- 첫마디: 대화마다 학생의 첫마디
- 이어지는 대화: 학생 말이 세 번 이상인 대화를 중간에서 끊고, 앞 대화를 붙여 물음

GGUF를 PC의 llama-server로, 파이보 앱과 같은 대답 설정(pibo_chat.SAMPLING, seed 고정)으로
묻습니다. 선생 모델이 생각을 켜고 대답마다 매깁니다: 한국어·존댓말, 알맞음(말뜻과 흐름에
맞고 구체적인가), 짧음, 이어 가기(1~5점), 틀린 사실, 앞 말 되풀이. 같은 대답 비율, 평균 글자
수, 기호·반말이 섞인 비율은 글자만 보고 셉니다.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from prompts import PIBO_STYLE, grams, pibo_problem, same
from teacher import after_think, first_json

RESULTS = Path("results")
HELDOUT = Path("data/heldout.jsonl")
TOOLS_HELDOUT = Path("data/heldout-tools.jsonl")
LANG = "ko"

JUDGE = """아래는 교육용 로봇 '파이보'와 학생의 대화이고, 마지막 파이보의 대답을 채점합니다.
{style}

{dialog}
파이보(채점할 대답): {reply}

마지막 대답을 판단해 JSON 하나로만 답하세요.
- korean: 한국어가 자연스럽고 존댓말을 지켰는가 (1~5)
- fit: 학생 말뜻과 앞 대화 흐름에 맞고, 물은 것에 구체적으로 답했는가 (1~5)
- short: 짧고 소리로 읽기 좋은가 (1~5)
- engage: 친구처럼 반응하고 대화를 이어 가는가 (1~5)
- wrong: 틀린 사실(과학, 상식, 숫자)이 하나라도 있으면 true
- repeat: 앞 대화에서 파이보가 한 말을 거의 그대로 되풀이했으면 true (앞 대화가 없으면 false)
출력 형식: {{"korean": 0, "fit": 0, "short": 0, "engage": 0, "wrong": false, "repeat": false}}"""

#: 영어 모델(themaker-en): 같은 항목을 영어로 (korean 칸에 영어의 자연스러움을 매김)
JUDGE_EN = """Below is a conversation between a student and TheMaker, an educational AI. Score TheMaker's
last reply.
{style}

{dialog}
TheMaker (reply to score): {reply}

Judge the last reply and answer with one JSON only.
- korean: the reply is in natural, easy English (1-5; 1 if it is not in English)
- fit: it fits what the student means and the flow, and answers specifically (1-5)
- short: it is short and easy to read aloud (1-5)
- engage: it reacts like a friend and keeps the conversation going (1-5)
- wrong: true if there is any wrong fact (science, common sense, numbers)
- repeat: true if it almost repeats something TheMaker said before (false if there is no earlier talk)
Output format: {{"korean": 0, "fit": 0, "short": 0, "engage": 0, "wrong": false, "repeat": false}}"""


def set_lang(lang: str) -> None:
    """en이면 영어 모델의 채점 대화(data/heldout-en.jsonl 등)와 결과 폴더(results-en)로."""
    global RESULTS, HELDOUT, TOOLS_HELDOUT, LANG
    LANG = lang
    if lang == "en":
        RESULTS, HELDOUT = Path("results-en"), Path("data/heldout-en.jsonl")
        TOOLS_HELDOUT = Path("data/heldout-tools-en.jsonl")


def problem(text: str) -> str | None:
    if LANG == "en":
        from prompts_en import en_problem
        return en_problem(text, 400)
    return pibo_problem(text, 400)


def with_system(messages: list[dict]) -> list[dict]:
    """시스템 글은 지금 것(tools.SYSTEM)으로: 채점 대화를 만든 뒤 시스템 글을 바꿨어도 기기 앱과 같게."""
    from tools import SYSTEMS

    return [dict(m, content=SYSTEMS[LANG]) if m["role"] == "system" else m for m in messages]


def questions() -> list[dict]:
    """[{"set": "첫마디"|"이어지는 대화", "messages": 물을 메시지들}]."""
    out, later = [], []
    for line in HELDOUT.read_text().splitlines():
        ex = json.loads(line)
        ex["messages"] = with_system(ex["messages"])
        out.append({"set": "첫마디", "messages": ex["messages"][:2]})
        users = [j for j, m in enumerate(ex["messages"]) if m["role"] == "user"]
        if len(users) >= 3:
            cut = users[(len(users) + 1) // 2]      # 가운데쯤의 학생 말까지 (앞 대화는 그대로)
            later.append({"set": "이어지는 대화", "messages": ex["messages"][:cut + 1]})
    return out + later


def tool_questions() -> list[dict]:
    """도구 채점용: 마지막 학생 말까지와 정답 호출 (부르면 안 되는 말이면 None)."""
    if not TOOLS_HELDOUT.exists():
        return []
    out = []
    for line in TOOLS_HELDOUT.read_text().splitlines():
        ex = json.loads(line)
        last = ex["messages"][-1]
        want = None
        if last.get("tool_calls"):
            fn = last["tool_calls"][0]["function"]
            want = [fn["name"], fn["arguments"]]
        out.append({"topic": ex.get("topic"), "messages": with_system(ex["messages"][:-1]),
                    "want": want})
    return out


def tool_answer(url: str) -> list:
    """도구 목록을 보내고 물어 첫 호출 [이름, 인자] 또는 None."""
    from export import post
    from pibo_chat import SAMPLING
    from tools import TOOLS, calls_of

    got = []
    for q in tool_questions():
        msg = post(f"{url}/v1/chat/completions", {"messages": q["messages"], "tools": TOOLS,
                                                  **SAMPLING, "seed": 0, "max_tokens": 100}
                   )["choices"][0]["message"]
        calls = calls_of(msg)
        got.append([calls[0][0], calls[0][1]] if calls else None)
    return got


def answer(gguf: str, name: str) -> None:
    from export import post, server
    from pibo_chat import SAMPLING

    if not Path(gguf).exists():
        print(f"{gguf}이 없어 건너뜁니다", flush=True)
        return
    qs = questions()
    answers = []
    with server(Path(gguf), 8396) as url:
        for n, q in enumerate(qs, 1):
            if n % 100 == 0:
                print(f"{n}/{len(qs)}", flush=True)
            msg = post(f"{url}/v1/chat/completions",
                       {"messages": q["messages"], **SAMPLING, "seed": 0, "max_tokens": 200}
                       )["choices"][0]["message"]
            answers.append((msg.get("content") or "").strip())
        tools = tool_answer(url)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"answers-{name}.json").write_text(json.dumps(
        {"gguf": gguf, "answers": answers, "tools": tools}, ensure_ascii=False, indent=1))
    print(f"답 {len(answers)}개 → {RESULTS / f'answers-{name}.json'}", flush=True)


def dialog(q: dict) -> str:
    lines = []
    for m in q["messages"][1:]:
        who = ("Student: ", "TheMaker: ") if LANG == "en" else ("학생: ", "파이보: ")
        lines.append(who[m["role"] != "user"] + m["content"])
    return "\n".join(lines)


def marks(text: str) -> dict:
    d = first_json(after_think(text)) or {}
    out = {k: d.get(k) if isinstance(d.get(k), (int, float)) and not isinstance(d.get(k), bool)
           else None for k in ("korean", "fit", "short", "engage")}
    out.update({k: d.get(k) if isinstance(d.get(k), bool) else None for k in ("wrong", "repeat")})
    return out


def judge() -> None:
    from prompts_en import EN_STYLE
    from teacher import load_teacher, thinking_params

    qs = questions()
    llm = None
    for f in sorted(RESULTS.glob("answers-*.json")):
        scored = f.with_name(f.name.replace("answers-", "scores-"))
        if scored.exists():
            continue
        answers = json.loads(f.read_text())["answers"]
        llm = llm or load_teacher()
        judge_, style = (JUDGE_EN, EN_STYLE) if LANG == "en" else (JUDGE, PIBO_STYLE)
        convs = [[{"role": "user", "content": judge_.format(style=style, dialog=dialog(q),
                                                            reply=a or "(말 없음)")}]
                 for q, a in zip(qs, answers, strict=True)]
        outs = llm.chat(convs, thinking_params(1536), chat_template_kwargs={"enable_thinking": True},
                        use_tqdm=False)
        scored.write_text(json.dumps([marks(o.outputs[0].text) for o in outs], ensure_ascii=False))
        print(f"채점: {f.name}", flush=True)
    print(report())


def echo(q: dict, reply: str) -> bool:
    """앞 대화의 파이보 말과 글자 덩어리(3글자)가 절반 넘게 겹치는지."""
    mine = grams(reply)
    return any(mine and len(mine & grams(m["content"])) / len(mine) > 0.5
               for m in q["messages"] if m["role"] == "assistant")


def report() -> str:
    qs = questions()
    head = ["모델", "묶음", "대화 점수", "한국어", "알맞음", "짧음", "이어 가기", "틀린 사실",
            "앞 말 되풀이 (선생)", "앞 말 되풀이 (글자)", "같은 대답", "기호·반말", "평균 글자 수"]
    if LANG == "en":
        head[3], head[11] = "영어", "기호·영어 아닌 글자"
    rows = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    for f in sorted(RESULTS.glob("answers-*.json")):
        name = f.stem.removeprefix("answers-")
        answers = json.loads(f.read_text())["answers"]
        sf = f.with_name(f.name.replace("answers-", "scores-"))
        scores = json.loads(sf.read_text()) if sf.exists() else [{}] * len(answers)
        for which in ("첫마디", "이어지는 대화"):
            idx = [i for i, q in enumerate(qs) if q["set"] == which]
            if not idx:
                continue

            def avg(key, idx=idx, scores=scores):
                v = [scores[i][key] for i in idx if isinstance(scores[i].get(key), (int, float))]
                return f"{statistics.mean(v):.2f}" if v else "-"

            def rate(key, idx=idx, scores=scores):
                v = [scores[i][key] for i in idx if isinstance(scores[i].get(key), bool)]
                return f"{sum(v) / len(v):.0%}" if v else "-"

            parts = [avg(k) for k in ("korean", "fit", "short", "engage")]
            nums = [float(x) for x in parts if x != "-"]
            total = f"{statistics.mean(nums):.2f}" if nums else "-"
            shapes = Counter(same(answers[i]) for i in idx)
            dup = sum(shapes[same(answers[i])] > 1 for i in idx) / len(idx)
            echoed = sum(echo(qs[i], answers[i]) for i in idx) / len(idx)
            bad = sum(bool(problem(answers[i])) for i in idx) / len(idx)
            length = statistics.mean(len(answers[i]) for i in idx)
            rows.append(f"| {name} | {which} ({len(idx)}) | {total} | {' | '.join(parts)} | "
                        f"{rate('wrong')} | {rate('repeat')} | {echoed:.0%} | {dup:.0%} | {bad:.0%} | "
                        f"{length:.0f} |")
    text = "\n".join(rows) + (
        "\n\n대화 점수: 한국어·알맞음·짧음·이어 가기의 평균 (5점 만점, 선생 모델이 생각을 켜고 채점). "
        "틀린 사실·되풀이·같은 대답·기호·반말은 낮을수록 좋음. 같은 대답: 서로 다른 질문에 글자 그대로 "
        "같은 대답. 앞 말 되풀이 (글자): 앞 대화의 파이보 말과 3글자 덩어리가 절반 넘게 겹침.\n")
    text += tool_report()
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "score.md").write_text(text)
    return text


def tool_report() -> str:
    """도구 정답률: 불러야 할 때 맞게 불렀나(이름과 인자), 부르면 안 될 때 안 불렀나."""
    qs = tool_questions()
    if not qs:
        return ""
    rows = ["", "| 모델 | 도구를 맞게 부름 | 부르면 안 될 때 안 부름 | 불빛 | 보기 | 움직이기 |",
            "| --- | --- | --- | --- | --- | --- |"]
    for f in sorted(RESULTS.glob("answers-*.json")):
        got = json.loads(f.read_text()).get("tools")
        if not got or len(got) != len(qs):
            continue

        def rate(idx, got=got):
            ok = [got[i] == qs[i]["want"] for i in idx]
            return f"{sum(ok) / len(ok):.0%} ({len(ok)})" if ok else "-"

        call = [i for i, q in enumerate(qs) if q["want"]]
        by = {k: [i for i in call if qs[i]["want"][0] == k] for k in ("light", "look", "move")}
        rows.append(f"| {f.stem.removeprefix('answers-')} | {rate(call)} | "
                    f"{rate([i for i, q in enumerate(qs) if not q['want']])} | {rate(by['light'])} | "
                    f"{rate(by['look'])} | {rate(by['move'])} |")
    return "\n".join(rows) + "\n\n도구: 학습에 안 쓴 도구 대화(heldout-tools)에 도구 목록을 함께 보내 첫 호출이 정답과 " \
        "같은지 (이름과 인자 모두).\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["answer", "judge", "report"])
    p.add_argument("--gguf")
    p.add_argument("--name")
    p.add_argument("--lang", choices=["ko", "en"], default="ko", help="en이면 영어 모델 채점")
    args = p.parse_args()
    set_lang(args.lang)
    if args.what == "answer":
        answer(args.gguf, args.name)
    elif args.what == "judge":
        judge()
    else:
        print(report())


if __name__ == "__main__":
    main()
