"""대화 예시 사실 검사: 선생 모델이 생각을 켜고 파이보의 말을 하나하나 다시 봅니다.

    python filter_chat.py --data data/school.jsonl --tools-from data/train-clean.jsonl \\
        --out data/train-school.jsonl

선생이 생각을 끄고 쓴 대화에 틀린 사실("공룡은 수천 년 전에 살았어요"), 정설을
얼버무린 말("공룡이 왜 사라졌는지는 아직 몰라요"), 어색한 한국어가 섞였고, 학생
모델이 그대로 배웠습니다.

- 문제가 없으면 그대로, 있으면 선생이 고쳐 쓴 말로 바꿉니다. 고친 말이 규칙
  (pibo_ok)에 어긋나거나 선생의 답을 못 읽으면(생각이 길어 잘림 등) 그 예시를 뺍니다.
  빼기만 하면 공룡처럼 아이들이 자주 묻는 주제가 데이터에서 사라지므로 고쳐 씁니다.
- 도구 예시는 그대로 둡니다 (filter_tools.py가 이미 봄). --tools-from을 주면 그 파일의
  도구 예시를 붙이고 시스템 글을 지금 것(tools.SYSTEM)으로 맞춥니다: 초·중·고 대화
  데이터는 대화만 만들고 도구 예시는 거른 어린이 데이터의 것을 다시 씁니다.
- 학년이 적힌 예시(초·중·고)는 그 학년의 말투 규칙과 길이로 봅니다.
- 판단은 data/chat-check.jsonl에 하나씩 남기고 (사람이 훑어보기용), 중간에 끊기면
  거기서 이어 갑니다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from make_data import load_teacher, thinking_params
from prompts import after_think, first_json, max_len, pibo_ok, style_for
from tools import SYSTEM

CHECK = """교육용 로봇 '파이보'의 대화 학습 데이터를 검사합니다.
학생들은 파이보의 말을 그대로 믿고 배우므로 사실이 정확해야 합니다.

{style}

대화:
{dialog}

파이보의 말 {n}개를 하나씩 보고 아래 문제가 있는지 판단하세요.
- 틀린 사실: 과학, 자연, 역사, 생활 상식이 틀렸거나 숫자, 연대가 틀림
  (예: "공룡은 수천 년 전에 살았어요" - 공룡은 약 6600만 년 전에 사라졌습니다)
- 얼버무림: 널리 받아들여진 사실이나 정설을 "아직 몰라요", "여러 가지 설이 있어요",
  "과학자들이 연구하고 있어요"로 피함 (예: 공룡이 사라진 까닭은 큰 소행성이 떨어진 것이 정설)
- 어색한 한국어: 문법이 틀렸거나 한국 사람이 쓰지 않는 말, 같은 말 되풀이, 아이 말에 맞지 않는 대답
- 반말 섞임: 파이보는 늘 존댓말("~해요")입니다. "힘들겠네", "재미있지?", "너랑"처럼 반말이 섞이거나
  자기를 "나", "내가"라고 하면("저", "제가"로) 고칩니다
- 모르는 걸 아는 척: 파이보는 학생이 말한 것만 압니다. 학생이 말하지 않은 날씨, 때, 장소, 기분을
  아는 듯이 말하면("비 오는 날이라 피곤한가요?") 고칩니다
정말 아무도 모르는 것(외계인이 있는지 등)을 모른다고 한 것은 문제가 아닙니다.
아이나 학생의 말은 맞춤법이 틀려도 따지지 않습니다.

문제가 없으면 {{"ok": true}}로 답하세요.
하나라도 있으면 파이보의 말 {n}개를 차례대로 모두 적되 문제 있는 말만 고쳐 쓰세요.
고친 말도 위 말투 규칙을 지키고, 앞뒤 말과 이어져야 합니다.
JSON 하나로만 답하세요.
출력 형식: {{"ok": false, "problem": "무엇이 문제인지 짧게", "fixed": ["...", "..."]}}"""

#: 이만큼씩 선생에게 보내고 결과를 적습니다 (진행률이 로그에 남고, 끊겨도 이어 감)
CHUNK = 1000


def chat_examples(lines: list[str]):
    """(줄 번호, 예시, 아이 말들, 파이보 말들) — 대화 예시만."""
    for i, line in enumerate(lines):
        ex = json.loads(line)
        if ex["kind"] != "chat":
            continue
        msgs = ex["messages"]
        yield (i, ex, [m["content"] for m in msgs if m["role"] == "user"],
               [m["content"] for m in msgs if m["role"] == "assistant"])


def prompt(ex: dict, users: list[str], pibos: list[str]) -> str:
    who = "학생" if ex.get("grade") else "아이"
    dialog = "\n".join(f"{who}: {u}\n파이보 {k}: {p}"
                       for k, (u, p) in enumerate(zip(users, pibos, strict=True), 1))
    return CHECK.format(style=style_for(ex), dialog=dialog, n=len(pibos))


def decide(text: str, pibos: list[str], limit: int = 120) -> dict:
    """선생의 답 → {"verdict": "ok"|"fixed"|"drop", "after": 고친 말들, "problem": ...}."""
    d = first_json(after_think(text))
    if not isinstance(d, dict) or not isinstance(d.get("ok"), bool):
        return {"verdict": "drop", "after": None, "problem": "선생의 답을 못 읽음"}
    if d["ok"]:
        return {"verdict": "ok", "after": pibos, "problem": None}
    fixed = d.get("fixed")
    problem = str(d.get("problem") or "")
    if not (isinstance(fixed, list) and len(fixed) == len(pibos)
            and all(pibo_ok(f, limit) for f in fixed)):
        return {"verdict": "drop", "after": None, "problem": problem or "고친 말이 규칙에 어긋남"}
    fixed = [f.strip() for f in fixed]
    return {"verdict": "fixed" if fixed != pibos else "ok", "after": fixed, "problem": problem}


def with_system(line: str) -> str:
    """예시의 시스템 글을 지금 것으로."""
    ex = json.loads(line)
    for m in ex["messages"]:
        if m["role"] == "system":
            m["content"] = SYSTEM
    return json.dumps(ex, ensure_ascii=False)


def rewrite(lines: list[str], checked: dict[int, dict], tools_from: list[str] = (),
            repeat: int = 1) -> list[str]:
    """검사 결과대로 고치거나 뺀 줄들 (+ tools_from의 도구 예시). 검사하지 않은 대화
    줄은 없어야 합니다."""
    out = []
    for i, line in enumerate(lines):
        ex = json.loads(line)
        if ex["kind"] != "chat":
            out.append(line)
            continue
        c = checked[i]
        if c["verdict"] == "drop":
            continue
        if c["verdict"] == "fixed":
            said = iter(c["after"])
            for m in ex["messages"]:
                if m["role"] == "assistant":
                    m["content"] = next(said)
            line = json.dumps(ex, ensure_ascii=False)
        out.append(line)
    out += [with_system(line) for line in tools_from if json.loads(line)["kind"] == "tool"] * repeat
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tools-from", help="이 파일의 도구 예시를 붙임 (시스템 글은 지금 것으로)")
    p.add_argument("--tools-repeat", type=int, default=1,
                   help="붙인 도구 예시를 몇 번씩 넣을지: 대화가 많을 때 도구 비율을 지키려고")
    args = p.parse_args()

    lines = Path(args.data).read_text().splitlines()
    items = list(chat_examples(lines))
    log_path = Path(args.out).with_name("chat-check.jsonl")
    checked: dict[int, dict] = {}
    if log_path.exists():
        for row in log_path.read_text().splitlines():
            r = json.loads(row)
            checked[r["line"]] = r
    left = [it for it in items if it[0] not in checked]
    print(f"대화 예시 {len(items)}개 중 {len(items) - len(left)}개는 이미 검사함", flush=True)
    if left:
        llm = load_teacher()
        params = thinking_params(2048)
        for start in range(0, len(left), CHUNK):
            part = left[start:start + CHUNK]
            outs = llm.chat([[{"role": "user", "content": prompt(ex, u, b)}]
                             for _, ex, u, b in part],
                            params, chat_template_kwargs={"enable_thinking": True},
                            use_tqdm=False)
            with log_path.open("a") as log:
                for (i, ex, users, pibos), o in zip(part, outs, strict=True):
                    r = {"line": i, "users": users, "before": pibos,
                         **decide(o.outputs[0].text, pibos, max_len(ex))}
                    checked[i] = r
                    log.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"검사 {len(items) - len(left) + start + len(part)}/{len(items)}", flush=True)
    tools = Path(args.tools_from).read_text().splitlines() if args.tools_from else []
    kept = rewrite(lines, checked, tools, args.tools_repeat)
    Path(args.out).write_text("\n".join(kept) + "\n")
    count = {v: sum(checked[i]["verdict"] == v for i, _, _, _ in items)
             for v in ("ok", "fixed", "drop")}
    print(f"대화 예시 {len(items)}개: 그대로 {count['ok']}, 고침 {count['fixed']}, "
          f"뺌 {count['drop']} → {args.out} ({len(kept)}줄)", flush=True)


if __name__ == "__main__":
    main()
