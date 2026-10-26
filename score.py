"""최종 모델과 예전 모델을 같은 질문으로 채점합니다 (results/score.md).

    python score.py answer --gguf out/themaker-0.8b-Q4_K_M.gguf --name themaker
    python score.py answer --gguf baseline/before-Q4_K_M.gguf --name before
    python score.py judge      # 모은 답을 선생 모델이 채점, 표를 씀
    python score.py report     # 표만 다시

질문 (학습에 쓰지 않은 것):
- 첫마디: data/heldout-school.jsonl의 학생 첫마디 400개
- 이어지는 대화: data/heldout-lead.jsonl의 대화 200개를 중간에서 끊고, 앞 대화를 붙여 물음

GGUF를 PC의 llama-server로, 파이보 앱과 같은 대답 설정(pibo_chat.SAMPLING, seed 고정)으로
묻습니다. 선생 모델이 생각을 켜고 대답마다 매깁니다: 한국어·존댓말, 알맞음(말뜻과 흐름에
맞고 구체적인가), 짧음, 이어 가기(1~5점), 틀린 사실, 앞 말 되풀이. 같은 대답 비율, 평균 글자
수, 기호·반말이 섞인 비율은 글자만 보고 셉니다.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path

from filter_chat import BANMAL
from prompts import after_think, first_json, pibo_problem, style_for

RESULTS = Path("results")
SCHOOL = Path("data/heldout-school.jsonl")
LEAD = Path("data/heldout-lead.jsonl")

JUDGE = """아래는 교육용 로봇 '파이보'와 학생의 대화이고, 마지막 파이보의 대답을 채점합니다.
{style}

{dialog}
파이보(채점할 대답): {reply}

마지막 대답을 판단해 JSON 하나로만 답하세요.
- korean: 한국어가 자연스럽고 존댓말을 지켰는가 (1~5, 학생이 영어로 말했으면 자연스러운 영어인가)
- fit: 학생 말뜻과 앞 대화 흐름에 맞고, 물은 것에 구체적으로 답했는가 (1~5)
- short: 짧고 소리로 읽기 좋은가 (1~5)
- engage: 친구처럼 반응하고 대화를 이어 가는가 (1~5)
- wrong: 틀린 사실(과학, 상식, 숫자)이 하나라도 있으면 true
- repeat: 앞 대화에서 파이보가 한 말을 거의 그대로 되풀이했으면 true (앞 대화가 없으면 false)
출력 형식: {{"korean": 0, "fit": 0, "short": 0, "engage": 0, "wrong": false, "repeat": false}}"""


def questions() -> list[dict]:
    """[{"set": "첫마디"|"이어지는 대화", "messages": 물을 메시지들, "grade", "lang"}]."""
    out = []
    for line in SCHOOL.read_text().splitlines():
        ex = json.loads(line)
        out.append({"set": "첫마디", "messages": ex["messages"][:2], "grade": ex.get("grade"),
                    "lang": ex.get("lang")})
    for line in LEAD.read_text().splitlines():
        ex = json.loads(line)
        users = [j for j, m in enumerate(ex["messages"]) if m["role"] == "user"]
        if len(users) < 3:
            continue
        cut = users[(len(users) + 1) // 2]      # 가운데쯤의 학생 말까지 (앞 대화는 그대로)
        out.append({"set": "이어지는 대화", "messages": ex["messages"][:cut + 1],
                    "grade": ex.get("grade"), "lang": ex.get("lang")})
    return out


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
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"answers-{name}.json").write_text(json.dumps(
        {"gguf": gguf, "answers": answers}, ensure_ascii=False, indent=1))
    print(f"답 {len(answers)}개 → {RESULTS / f'answers-{name}.json'}", flush=True)


def dialog(q: dict) -> str:
    lines = []
    for m in q["messages"][1:]:
        lines.append(("학생: " if m["role"] == "user" else "파이보: ") + m["content"])
    return "\n".join(lines)


def marks(text: str) -> dict:
    d = first_json(after_think(text)) or {}
    out = {k: d.get(k) if isinstance(d.get(k), (int, float)) and not isinstance(d.get(k), bool)
           else None for k in ("korean", "fit", "short", "engage")}
    out.update({k: d.get(k) if isinstance(d.get(k), bool) else None for k in ("wrong", "repeat")})
    return out


def judge() -> None:
    from make_data import load_teacher, thinking_params

    qs = questions()
    llm = None
    for f in sorted(RESULTS.glob("answers-*.json")):
        scored = f.with_name(f.name.replace("answers-", "scores-"))
        if scored.exists():
            continue
        answers = json.loads(f.read_text())["answers"]
        llm = llm or load_teacher()
        convs = [[{"role": "user", "content": JUDGE.format(style=style_for(q), dialog=dialog(q),
                                                           reply=a or "(말 없음)")}]
                 for q, a in zip(qs, answers, strict=True)]
        outs = llm.chat(convs, thinking_params(1536), chat_template_kwargs={"enable_thinking": True},
                        use_tqdm=False)
        scored.write_text(json.dumps([marks(o.outputs[0].text) for o in outs], ensure_ascii=False))
        print(f"채점: {f.name}", flush=True)
    print(report())


def same(text: str) -> str:
    return re.sub(r"[\s.,!?~…'\"]", "", text)


def echo(q: dict, reply: str) -> bool:
    """앞 대화의 파이보 말과 글자 덩어리(3글자)가 절반 넘게 겹치는지."""
    def grams(t):
        t = same(t)
        return {t[i:i + 3] for i in range(len(t) - 2)}
    mine = grams(reply)
    return any(mine and len(mine & grams(m["content"])) / len(mine) > 0.5
               for m in q["messages"] if m["role"] == "assistant")


def report() -> str:
    qs = questions()
    head = ["모델", "묶음", "대화 점수", "한국어", "알맞음", "짧음", "이어 가기", "틀린 사실",
            "앞 말 되풀이 (선생)", "앞 말 되풀이 (글자)", "같은 대답", "기호·반말", "평균 글자 수"]
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
            bad = sum(bool(pibo_problem(answers[i], 400)) or bool(BANMAL.search(answers[i]))
                      for i in idx) / len(idx)
            length = statistics.mean(len(answers[i]) for i in idx)
            rows.append(f"| {name} | {which} ({len(idx)}) | {total} | {' | '.join(parts)} | "
                        f"{rate('wrong')} | {rate('repeat')} | {echoed:.0%} | {dup:.0%} | {bad:.0%} | "
                        f"{length:.0f} |")
    text = "\n".join(rows) + (
        "\n\n대화 점수: 한국어·알맞음·짧음·이어 가기의 평균 (5점 만점, 선생 모델이 생각을 켜고 채점). "
        "틀린 사실·되풀이·같은 대답·기호·반말은 낮을수록 좋음. 같은 대답: 서로 다른 질문에 글자 그대로 "
        "같은 대답. 앞 말 되풀이 (글자): 앞 대화의 파이보 말과 3글자 덩어리가 절반 넘게 겹침.\n")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "score.md").write_text(text)
    return text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["answer", "judge", "report"])
    p.add_argument("--gguf")
    p.add_argument("--name")
    args = p.parse_args()
    if args.what == "answer":
        answer(args.gguf, args.name)
    elif args.what == "judge":
        judge()
    else:
        print(report())


if __name__ == "__main__":
    main()
