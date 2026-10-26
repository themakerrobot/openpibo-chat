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
- 이야기하다가 중간에 도구를 부탁하는 대화(mixed)도 파이보의 말을 봅니다: 도구를 부른
  차례는 그 도구와 결과를 함께 보여 줘 아는 척으로 보지 않게 하고, 도구 호출은 그대로 둡니다.
- 판단은 data/chat-check.jsonl(--log)에 하나씩 남기고 (사람이 훑어보기용), 중간에 끊기면
  거기서 이어 갑니다.
"""

from __future__ import annotations

import argparse
import json
import re
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

#: 파이보 말에 남은 반말 (사실 검사가 절반만 고친 것: "너는 강해서요. 괜찮겠네.")
BANMAL = re.compile(r"(^|[\s,.!?])(너는|너도|너가|너를|너랑|너처럼|너한테|넌|니가|네가|내가)(?=\s|$|[,.!?])"
                    r"|(겠네|했네|거든|구나|잖아|했어|할게|줄게|볼까|있지|좋지|맞지|그치)[.!?]?$"
                    r"|(겠네|했네|거든|구나|잖아)[.!?]\s")

POLITE = """교육용 로봇 '파이보'의 대답을 다듬습니다. 파이보는 늘 친근한 존댓말("~해요", "~예요")로
말하고, 학생을 "너"라고 부르지 않으며, 자기를 "저", "제가"라고 합니다.

{dialog}

파이보의 말 {n}개에서 반말("~했네", "~지?", "~거든", "너는", "내가" 등)을 존댓말로만 바꾸세요.
뜻과 길이는 그대로 두고, 반말이 없는 말은 글자 하나 바꾸지 마세요.
JSON 하나로만 답하세요. 출력 형식: {{"fixed": ["...", "..."]}}"""


#: 파이보의 말을 검사하는 예시: 대화, 대화 중 도구 (도구 예시는 filter_tools.py가 봄)
CHECKED = ("chat", "mixed")


def said(m: dict) -> bool:
    """파이보가 소리 내 하는 말인지 (도구만 부른 차례는 아님)."""
    return m["role"] == "assistant" and not m.get("tool_calls")


def chat_examples(lines: list[str]):
    """(줄 번호, 예시, 아이 말들, 파이보 말들) — 대화와 대화 중 도구 예시."""
    for i, line in enumerate(lines):
        ex = json.loads(line)
        if ex["kind"] not in CHECKED:
            continue
        msgs = ex["messages"]
        yield (i, ex, [m["content"] for m in msgs if m["role"] == "user"],
               [m["content"] for m in msgs if said(m)])


def tool_notes(ex: dict) -> list[str]:
    """주고받기마다 파이보가 쓴 도구와 결과 ("" = 도구 없음)."""
    notes = []
    for m in ex["messages"]:
        if m["role"] == "user":
            notes.append("")
        elif m["role"] == "assistant" and m.get("tool_calls"):
            names = ", ".join(c["function"]["name"] for c in m["tool_calls"])
            notes[-1] = f"(파이보가 도구 {names}을 쓰고 받은 결과: " + "{result})"
        elif m["role"] == "tool" and "{result}" in notes[-1]:
            notes[-1] = notes[-1].replace("{result}", m["content"])
    return notes


TOOL_NOTE = ("괄호 안은 파이보가 실제로 쓴 도구와 그 결과입니다. 결과에 나온 것을 말하는 건 "
             "아는 척이 아닙니다.\n")


def prompt(ex: dict, users: list[str], pibos: list[str]) -> str:
    who = "학생" if ex.get("grade") else "아이"
    notes = tool_notes(ex)
    dialog = "\n".join(f"{who}: {u}\n" + (f"{n}\n" if n else "") + f"파이보 {k}: {p}"
                       for k, (u, p, n) in enumerate(zip(users, pibos, notes, strict=True), 1))
    if any(notes):
        dialog = TOOL_NOTE + dialog
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
        if ex["kind"] not in CHECKED:
            out.append(line)
            continue
        c = checked[i]
        if c["verdict"] == "drop":
            continue
        if c["verdict"] == "fixed":
            after = iter(c["after"])
            for m in ex["messages"]:
                if said(m):
                    m["content"] = next(after)
            line = json.dumps(ex, ensure_ascii=False)
        out.append(line)
    out += [with_system(line) for line in tools_from if json.loads(line)["kind"] == "tool"] * repeat
    return out


def matches(r: dict, users: list[str], pibos: list[str]) -> bool:
    """기록이 지금 데이터의 그 줄과 같은 대화인지: 같은 이름의 기록 파일을 다른 데이터
    (어린이 데이터 검사)가 썼으면 줄 번호만 같고 대화가 달라, 말 개수가 안 맞습니다."""
    return r.get("users") == users and r.get("before") == pibos


def polite(llm, todo: list, checked: dict, log_path: Path, limit_of) -> int:
    """반말이 남은 파이보 말을 선생이 존댓말로만 고칩니다 (생각 없이, 빠름). 고친 말이 규칙에
    어긋나거나 반말이 그대로면 그 대화를 뺍니다. 고친 수를 돌려줍니다."""
    from vllm import SamplingParams

    convs = []
    for i, ex, users, _ in todo:
        who = "학생" if ex.get("grade") else "아이"
        dialog = "\n".join(f"{who}: {u}\n파이보 {k}: {p}" for k, (u, p) in
                           enumerate(zip(users, checked[i]["after"], strict=True), 1))
        convs.append([{"role": "user", "content": POLITE.format(dialog=dialog,
                                                                n=len(users))}])
    outs = llm.chat(convs, SamplingParams(temperature=0.0, max_tokens=800),
                    chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
    done = 0
    with log_path.open("a") as log:
        for (i, ex, _, pibos), o in zip(todo, outs, strict=True):
            d = first_json(o.outputs[0].text) or {}
            fixed = d.get("fixed")
            r = dict(checked[i], polite=True)
            if (isinstance(fixed, list) and len(fixed) == len(pibos)
                    and all(pibo_ok(f, limit_of(ex)) for f in fixed)
                    and not any(BANMAL.search(f.strip()) for f in fixed)):
                r.update(verdict="fixed", after=[f.strip() for f in fixed])
                done += 1
            else:
                r.update(verdict="drop", after=None,
                         problem=(r.get("problem") or "") + " / 반말을 못 고침")
            checked[i] = r
            log.write(json.dumps(r, ensure_ascii=False) + "\n")
    return done


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tools-from", help="이 파일의 도구 예시를 붙임 (시스템 글은 지금 것으로)")
    p.add_argument("--tools-repeat", type=int, default=1,
                   help="붙인 도구 예시를 몇 번씩 넣을지: 대화가 많을 때 도구 비율을 지키려고")
    p.add_argument("--log", help="판단 기록 (기본: --out 옆의 chat-check.jsonl)")
    args = p.parse_args()

    lines = Path(args.data).read_text().splitlines()
    items = list(chat_examples(lines))
    log_path = Path(args.log) if args.log else Path(args.out).with_name("chat-check.jsonl")
    by_line = {i: (users, pibos) for i, _, users, pibos in items}
    checked: dict[int, dict] = {}
    if log_path.exists():
        rows, kept_rows, other = log_path.read_text().splitlines(), [], 0
        for row in rows:
            try:
                r = json.loads(row)
            except json.JSONDecodeError:     # 쓰다가 끊긴 마지막 줄: 그 예시는 다시 봅니다
                continue
            if r.get("line") not in by_line or not matches(r, *by_line[r["line"]]):
                other += 1                   # 다른 데이터의 기록: 버리고 다시 봅니다
                continue
            checked[r["line"]] = r
            kept_rows.append(row)
        if len(kept_rows) != len(rows):
            print(f"기록 {len(rows)}줄 중 이 데이터와 안 맞는 {other}줄을 버림", flush=True)
            log_path.write_text("".join(row + "\n" for row in kept_rows))
    left = [it for it in items if it[0] not in checked]
    print(f"대화 예시 {len(items)}개 중 {len(items) - len(left)}개는 이미 검사함", flush=True)
    llm = None
    if left:
        llm = llm or load_teacher()
        params = thinking_params(3072)   # 일곱 번 주고받는 대화도 생각이 잘리지 않게
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
    rude = [it for it in items if checked[it[0]]["verdict"] != "drop"
            and not checked[it[0]].get("polite")
            and any(BANMAL.search(s) for s in checked[it[0]]["after"])]
    n_said = sum(len(checked[i]["after"]) for i, _, _, _ in items if checked[i]["verdict"] != "drop")
    print(f"반말이 남은 대화 {len(rude)}개 (파이보 말 {n_said}개 중)", flush=True)
    for start in range(0, len(rude), CHUNK * 4):
        llm = llm or load_teacher()
        part = rude[start:start + CHUNK * 4]
        n = polite(llm, part, checked, log_path, max_len)
        print(f"존댓말로 고침 {start + len(part)}/{len(rude)} (이번에 고친 것 {n})", flush=True)
    tools = Path(args.tools_from).read_text().splitlines() if args.tools_from else []
    kept = rewrite(lines, checked, tools, args.tools_repeat)
    Path(args.out).write_text("\n".join(kept) + "\n")
    count = {v: sum(checked[i]["verdict"] == v for i, _, _, _ in items)
             for v in ("ok", "fixed", "drop")}
    print(f"대화 예시 {len(items)}개: 그대로 {count['ok']}, 고침 {count['fixed']}, "
          f"뺌 {count['drop']} → {args.out} ({len(kept)}줄)", flush=True)


if __name__ == "__main__":
    main()
