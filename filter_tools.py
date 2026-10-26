"""도구 예시 거르기: 선생 모델에게 "이 말에 이 도구를 부르는 게 자연스러운가"를 다시 묻습니다.

    python filter_tools.py --train data/train.jsonl --heldout data/heldout.jsonl \\
        --out data/train-clean.jsonl --marks data/heldout-marks.json

데이터를 만들 때 도구를 먼저 정하고 "이 도구를 써야 하는 장면"을 쓰게 해서, 뽑힌
도구에 따라 억지 장면이 섞였습니다 (예: "내가 화났어" → find_objects).

- 학습 데이터: 억지라고 본 도구 예시를 뺀 것을 --out에 씁니다 (대화 예시는 그대로,
  줄은 원래 글자 그대로라 증류 결과(data/kd)도 그대로 맞습니다).
- 채점용 데이터: 파일은 그대로 두고(이미 모은 답의 순서가 바뀌지 않게) 번호만
  --marks에 적습니다. bad: 채점에서 뺄 억지 문제, names_only: 동작·색·지역이 아이
  말로 정해지지 않아("춤춰 줘") 도구 이름만 볼 문제.
- 판단 하나하나는 data/tool-check.jsonl(--log)에 남깁니다 (사람이 훑어보기용).
- 이야기하다가 중간에 도구를 부탁하는 대화(mixed)는 도구를 부른 차례마다 봅니다:
  하나라도 억지면 그 대화를 빼고(채점용이면 bad), 인자가 말로 안 정해지면 names_only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from make_data import load_teacher
from prompts import TOOL_TEXT, first_json

CHECK = """어린이용 교육 로봇 '파이보'의 학습 데이터를 검사합니다.

파이보의 도구:
{tools}

아이의 말: "{user}"
파이보가 부른 도구: {calls}

두 가지를 판단해 JSON 하나로만 답하세요.
- natural: 이 말을 들은 로봇이 이 도구들을 부르는 게 자연스럽고 아이가 바란 일에 꼭 맞으면 true.
  아이가 기분이나 이야기를 말했을 뿐이라 도구가 필요 없거나, 다른 도구가 맞거나,
  아이가 바라지 않은 도구를 덤으로 부르면 false.
- args_from_words: 동작 이름, 색, 지역 같은 인자가 아이의 말만으로 정해지면 true.
  예: "춤춰 줘"는 어떤 춤인지 안 정해지므로 false, "눈 파랗게 해 줘"는 true, 인자가 없는 도구는 true.
출력 형식: {{"natural": true, "args_from_words": true}}"""


def tool_examples(lines: list[str]):
    """(줄 번호, 아이 말, 부른 도구들) — 도구 예시와, 대화 중 도구를 부른 차례마다."""
    for i, line in enumerate(lines):
        ex = json.loads(line)
        if ex["kind"] not in ("tool", "mixed"):
            continue
        msgs = ex["messages"]
        for j, m in enumerate(msgs):
            if m["role"] == "assistant" and m.get("tool_calls"):
                yield i, msgs[j - 1]["content"], [c["function"] for c in m["tool_calls"]]


def prompt(user: str, calls: list[dict]) -> str:
    shown = ", ".join(f"{c['name']}({json.dumps(c['arguments'], ensure_ascii=False)})"
                      for c in calls)
    return CHECK.format(tools=TOOL_TEXT, user=user, calls=shown)


def verdict(text: str) -> dict:
    """선생의 답 → {"natural": bool|None, "args_from_words": bool|None} (못 읽으면 None)."""
    d = first_json(text) or {}
    return {k: d.get(k) if isinstance(d.get(k), bool) else None
            for k in ("natural", "args_from_words")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--heldout", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--marks", required=True)
    p.add_argument("--log", help="판단 기록 (기본: --out 옆의 tool-check.jsonl)")
    args = p.parse_args()

    from vllm import SamplingParams

    sets = {"train": Path(args.train).read_text().splitlines(),
            "heldout": Path(args.heldout).read_text().splitlines()}
    items = [(name, i, user, calls) for name, lines in sets.items()
             for i, user, calls in tool_examples(lines)]
    llm = load_teacher()
    outs = llm.chat([[{"role": "user", "content": prompt(u, c)}] for _, _, u, c in items],
                    SamplingParams(temperature=0.0, max_tokens=40),
                    chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
    drop, bad, names_only, unread = set(), set(), set(), 0
    with open(args.log or Path(args.out).with_name("tool-check.jsonl"), "w") as log:
        for (name, i, user, calls), o in zip(items, outs, strict=True):
            v = verdict(o.outputs[0].text)
            unread += v["natural"] is None
            log.write(json.dumps({"set": name, "line": i, "user": user, "calls": calls, **v},
                                 ensure_ascii=False) + "\n")
            if v["natural"] is False and name == "train":   # 못 읽은 답은 두고 갑니다
                drop.add(i)
            elif v["natural"] is False:
                bad.add(i)
            elif name == "heldout" and v["args_from_words"] is False:
                names_only.add(i)
    names_only -= bad
    bad, names_only = sorted(bad), sorted(names_only)
    kept = [line for i, line in enumerate(sets["train"]) if i not in drop]
    Path(args.out).write_text("\n".join(kept) + "\n")
    n_train = sum(n == "train" for n, *_ in items)
    n_held = len({i for n, i, *_ in items if n == "heldout"})
    Path(args.marks).write_text(json.dumps(
        {"bad": bad, "names_only": names_only, "checked": n_held}, ensure_ascii=False))
    print(f"학습용 도구 예시 {n_train}개 중 {len(drop)}개 뺌 → {args.out} ({len(kept)}줄)",
          flush=True)
    print(f"채점용 도구 문제(줄) {n_held}개 중 억지 {len(bad)}개, 이름만 볼 것 {len(names_only)}개",
          flush=True)
    if unread:
        print(f"선생의 답을 못 읽어 그대로 둔 것 {unread}개", flush=True)


if __name__ == "__main__":
    main()
