"""영어 모델용 학습 대화: 바꿔 쓰기·검사를 통과한 한국어 대화를 선생 모델이 쉬운 영어로 옮기고,
다시 검사해 좋은 것만 남깁니다 (prompts_en.py).

    python translate.py --data data/pibo.jsonl --out data/pibo-en.jsonl
    python translate.py --data data/pibo.jsonl --out data/try-en.jsonl --limit 200   # 먼저 조금
    python translate.py --data data/pibo.jsonl --limit 200 --dropped natural > samples/drop-en.txt

1. 옮기기 (생각 끔): 뜻과 흐름을 살려 학생 말과 더메이커 말을 함께 옮깁니다. 한국어로만 되는
   대화(말장난, 맞춤법 …)는 선생이 뺍니다. 글자 검사(영어 아닌 글자, 기호, 길이)에 걸리면 뺍니다.
2. 검사 (생각 켬): 자연스러움·알맞음·친구 말투가 4점 미만이거나 어른 이야기, 지어낸 사람 경험,
   틀린 사실, 안전 문제가 있으면 뺍니다.

판단과 선생의 답 원문은 --log에 대화 id마다 남기고, 다시 돌리면 이미 한 것은 건너뜁니다 (규칙을
고쳐도 남은 원문으로 다시 판단). 데이터와 기록은 PC 밖으로 내보내지 않습니다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from prompts import decide_check
from prompts_en import check_prompt, decide_translate, dialog, translate_prompt

CHUNK = 3000
#: 단계마다 (최대 토큰, 생각할지)
STAGES = {"translate": (1024, False), "check": (2048, True)}


def key_of(ex: dict) -> str:
    return hashlib.md5(json.dumps(ex["messages"], ensure_ascii=False).encode()).hexdigest()[:12]


def id_of(ex: dict, n: int) -> str:
    return str(ex.get("id") or f"line{n}")


def decide(stage: str, text: str, ex: dict) -> dict:
    if stage == "translate":
        msgs, why = decide_translate(text, ex)
        return {"messages": msgs, "why": why}
    marks, why = decide_check(text)
    return {"marks": marks, "why": why}


def load_log(path: Path, exs: dict[str, dict]) -> dict[tuple[str, str], dict]:
    """{(id, stage): 기록}. 원본이 바뀐 기록은 버리고, 남은 원문으로 지금 규칙대로 다시 판단합니다.
    옮긴 대화가 달라졌으면 그 검사는 다시 합니다."""
    done = {}
    if path.exists():
        for row in path.read_text().splitlines():
            try:
                r = json.loads(row)
            except json.JSONDecodeError:      # 쓰다가 끊긴 마지막 줄
                continue
            ex = exs.get(r.get("id"))
            if ex is not None and key_of(ex) == r.get("key"):
                done[r["id"], r["stage"]] = r
    for (i, stage), r in list(done.items()):
        if stage == "translate":
            r.update(decide(stage, r["raw"], exs[i]))
    for (i, stage), r in list(done.items()):
        if stage != "check":
            continue
        tr = done.get((i, "translate"))
        if tr is None or not tr.get("messages") or r.get("for") != msgs_key(tr["messages"]):
            del done[i, stage]
        else:
            r.update(decide(stage, r["raw"], exs[i]))
    return done


def msgs_key(messages) -> str:
    return hashlib.md5(json.dumps(messages, ensure_ascii=False).encode()).hexdigest()[:12]


def run(llm_ask, exs: dict[str, dict], log_path: Path, timing: dict | None = None) -> dict:
    """llm_ask(글 목록, 최대 토큰, 생각할지) → 답 목록."""
    done = load_log(log_path, exs)
    for stage, (tokens, think) in STAGES.items():
        todo = [i for i in exs if (i, stage) not in done
                and (stage == "translate" or done[i, "translate"]["messages"])]
        print(f"{stage}: {len(todo)}개 할 차례", flush=True)
        for start in range(0, len(todo), CHUNK):
            part = todo[start:start + CHUNK]
            prompts = [translate_prompt(exs[i]) if stage == "translate" else
                       check_prompt(done[i, "translate"]["messages"]) for i in part]
            t = time.time()
            answers = llm_ask(prompts, tokens, think)
            if timing is not None:
                n, sec = timing.get(stage, [0, 0.0])
                timing[stage] = [n + len(part), sec + time.time() - t]
            with log_path.open("a") as log:
                for i, text in zip(part, answers, strict=True):
                    r = {"id": i, "key": key_of(exs[i]), "stage": stage, "raw": text,
                         **decide(stage, text, exs[i])}
                    if stage == "check":
                        r["for"] = msgs_key(done[i, "translate"]["messages"])
                    done[i, stage] = r
                    log.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"{stage} {start + len(part)}/{len(todo)}", flush=True)
    return done


def results(exs: dict[str, dict], done: dict) -> tuple[list[dict], Counter, Counter]:
    """(남은 영어 대화들, 데이터마다 남은 수, 뺀 까닭마다 수)."""
    kept, by_source, why = [], Counter(), Counter()
    for i, ex in exs.items():
        tr, ck = done.get((i, "translate")), done.get((i, "check"))
        if tr is not None and tr["why"]:
            reason = tr["why"]
        elif tr is None or ck is None:
            reason = "아직 안 함"
        else:
            reason = ck["why"]
        if reason:
            why[reason.split(":")[0]] += 1
            continue
        kept.append({k: v for k, v in ex.items() if k not in ("messages", "marks", "tools")}
                    | {"lang": "en", "messages": tr["messages"], "marks": ck["marks"]})
        by_source[ex.get("source", "?")] += 1
    return kept, by_source, why


def dropped(exs: dict[str, dict], done: dict, reason: str) -> str:
    """까닭이 reason으로 시작하는 빠진 대화를 한국어 원문과 옮긴 글로."""
    lines = []
    for i, ex in exs.items():
        tr, ck = done.get((i, "translate")) or {}, done.get((i, "check")) or {}
        why = tr.get("why") or ck.get("why")
        if not why or not why.startswith(reason):
            continue
        lines += [f"--- {ex.get('source', '')} [{ex.get('topic') or ''}] {why}", dialog(ex["messages"])]
        if tr.get("messages"):
            lines.append(dialog(tr["messages"]))
        if ck.get("marks"):
            lines.append(f"점수: {json.dumps(ck['marks'], ensure_ascii=False)}")
    return "\n".join(lines) + f"\n\n{reason}: {sum(x.startswith('---') for x in lines)}개\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="옮길 한국어 대화 (data/pibo.jsonl 등, PC에만 있음)")
    p.add_argument("--out", help="남은 영어 대화 (--dropped면 없어도 됨)")
    p.add_argument("--log", default="data/translate-log.jsonl")
    p.add_argument("--dropped", metavar="까닭", help="이 까닭으로 빠진 대화를 보여 주고 끝 (예: natural)")
    p.add_argument("--limit", type=int, help="앞에서 이만큼만 (먼저 해 보기)")
    p.add_argument("--teacher", help="정해서 쓸 선생 모델")
    args = p.parse_args()

    if not Path(args.data).exists():
        raise SystemExit(f"{args.data}가 없어요: 한국어 데이터(data-pibo)를 먼저 만드세요")
    rows = [json.loads(x) for x in Path(args.data).read_text().splitlines() if x.strip()]
    whole = len(rows)
    if args.limit:
        rows = rows[:args.limit]
    exs = {id_of(ex, n): ex for n, ex in enumerate(rows)}
    log_path = Path(args.log)
    if args.dropped:
        print(dropped(exs, load_log(log_path, exs), args.dropped))
        return
    if not args.out:
        p.error("--out이 필요해요")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    llm, load_s = None, 0.0

    def llm_ask(prompts, tokens, think):
        nonlocal llm, load_s
        from teacher import ask, load_teacher

        if llm is None:
            t = time.time()
            llm = load_teacher(args.teacher)
            load_s = time.time() - t
        return ask(llm, prompts, tokens, think)

    timing: dict = {}
    done = run(llm_ask, exs, log_path, timing)
    if timing:                                # 걸린 시간과, 전체를 하면 걸릴 시간
        first = next(iter(timing))
        timing[first][1] -= load_s
        hours = 0.0
        for stage, (n, sec) in timing.items():
            per = max(sec, 0.0) / n
            hours += per * (n / timing[first][0]) * whole / 3600
            print(f"시간: {stage} {n}개 {sec:.0f}초 (개당 {per:.2f}초)", flush=True)
        print(f"시간: 선생 모델 띄우기 {load_s:.0f}초. 전체 {whole}개를 하면 약 {hours:.1f}시간 "
              f"(이번 속도 그대로라면)", flush=True)
    kept, by_source, why = results(exs, done)
    total = Counter(ex.get("source", "?") for ex in exs.values())
    print("남은 것: " + ", ".join(f"{k} {by_source[k]}/{n}" for k, n in sorted(total.items())),
          flush=True)
    print("뺀 까닭: " + (", ".join(f"{k} {v}" for k, v in why.most_common(12)) or "없음"), flush=True)
    Path(args.out).write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in kept))
    print(f"{args.out}: {len(kept)}개", flush=True)


if __name__ == "__main__":
    main()
