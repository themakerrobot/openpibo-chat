"""원본 대화의 상대 말을 선생 모델이 파이보 말로 바꿔 쓰고, 다시 검사해 좋은 것만 남깁니다.

    python rewrite.py --data data/source.jsonl --out data/pibo.jsonl
    python rewrite.py --data data/source.jsonl --out data/try.jsonl --limit 200   # 먼저 조금 해 보기

1. 바꿔 쓰기 (prompts.REWRITE, 생각 켬): 학생과 나눌 만한 대화가 아니면 선생이 뺍니다.
   선생은 파이보 말만 쓰고 학생 말은 원본 그대로 씁니다 (가림 표시가 있는 대화는 미리 뺌).
   글자 검사(기호, 다른 나라 글자, 반말, 길이, 학생 말 되받아 읊기)에 걸리면 뺍니다.
2. 검사 (prompts.CHECK, 생각 켬): 자연스러움·알맞음·친구 말투가 4점 미만이거나, 어른 이야기·
   저급한 말, 지어낸 사람 경험, 틀린 사실, 안전 문제가 있으면 뺍니다. 고치지 않고 뺍니다.

판단과 선생의 답 원문은 --log(기본 data/rewrite-log.jsonl)에 대화 id마다 남기고, 다시 돌리면 이미
한 것은 건너뜁니다. 글자 검사 규칙을 고치면 남은 원문으로 다시 판단합니다 (선생에게 다시 묻지 않음). --limit으로 해 본 것도 같은 기록을 쓰므로 나중에 전체를 돌릴 때 다시 하지 않습니다.
원본: data/source.jsonl, 한 줄에 {"id", "source", "topic", "turns": [{"user", "other"}]}.
원본, 기록, 결과는 PC 밖으로 내보내지 않습니다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from prompts import MASK, check_prompt, decide_check, decide_rewrite, rewrite_prompt, to_messages

#: 이만큼씩 선생에게 보내고 기록합니다 (진행률이 로그에 남고, 끊겨도 이어 감)
CHUNK = 1000


def key_of(src: dict) -> str:
    """원본이 바뀌면 다시 하도록 원본 대화의 지문."""
    return hashlib.md5(json.dumps(src["turns"], ensure_ascii=False).encode()).hexdigest()[:12]


def turns_key(turns) -> str:
    return hashlib.md5(json.dumps(turns, ensure_ascii=False).encode()).hexdigest()[:12]


def load_log(path: Path, sources: list[dict]) -> dict[tuple[str, str], dict]:
    """{(id, "rewrite"|"check"): 기록}. 원본이 바뀐 기록은 버립니다. 선생의 답 원문(raw)이 남아
    있으면 지금 규칙으로 다시 판단합니다: 규칙을 고쳐도 선생에게 다시 묻지 않습니다. 바꿔 쓴
    대화가 달라졌으면 그 검사는 다시 합니다."""
    by_id = {s["id"]: s for s in sources}
    done = {}
    if path.exists():
        for row in path.read_text().splitlines():
            try:
                r = json.loads(row)
            except json.JSONDecodeError:      # 쓰다가 끊긴 마지막 줄: 그 대화는 다시 합니다
                continue
            src = by_id.get(r.get("id"))
            if src is not None and key_of(src) == r.get("key"):
                done[r["id"], r["stage"]] = r
    for (i, stage), r in list(done.items()):
        if stage == "rewrite" and "raw" in r:
            r["turns"], r["why"] = decide_rewrite(r["raw"], by_id[i])
    for (i, stage), r in list(done.items()):
        if stage != "check":
            continue
        rw = done.get((i, "rewrite"))
        if rw is None or not rw.get("turns") or ("for" in r and r["for"] != turns_key(rw["turns"])):
            del done[i, stage]
        elif "raw" in r:
            r["marks"], r["why"] = decide_check(r["raw"])
    return done


def run(llm_ask, sources: list[dict], log_path: Path,
        timing: dict | None = None) -> dict[tuple[str, str], dict]:
    """llm_ask(글 목록, 최대 토큰) → 답 목록. 바꿔 쓰기와 검사를 하고 기록을 돌려줍니다.
    timing을 주면 단계마다 [한 개수, 걸린 초]를 더해 둡니다."""
    done = load_log(log_path, sources)
    stages = [("rewrite", 3072, lambda s: rewrite_prompt(s),
               lambda s, text: dict(zip(("turns", "why"), decide_rewrite(text, s), strict=True))),
              ("check", 2048, lambda s: check_prompt(to_messages(done[s["id"], "rewrite"]["turns"])),
               lambda s, text: dict(zip(("marks", "why"), decide_check(text), strict=True)))]
    for stage, tokens, make, decide in stages:
        if stage == "rewrite":               # 학생 말에 가림 표시가 있으면 묻지 않고 뺌
            for s in sources:
                if (s["id"], stage) not in done and any(MASK.search(t["user"]) for t in s["turns"]):
                    done[s["id"], stage] = {"id": s["id"], "key": key_of(s), "stage": stage,
                                            "turns": None, "why": "학생 말에 가림 표시"}
        todo = [s for s in sources if (s["id"], stage) not in done
                and (stage == "rewrite" or done[s["id"], "rewrite"]["turns"])]
        print(f"{stage}: {len(todo)}개 할 차례", flush=True)
        for start in range(0, len(todo), CHUNK):
            part = todo[start:start + CHUNK]
            t = time.time()
            answers = llm_ask([make(s) for s in part], tokens)
            if timing is not None:
                n, sec = timing.get(stage, [0, 0.0])
                timing[stage] = [n + len(part), sec + time.time() - t]
            with log_path.open("a") as log:
                for s, text in zip(part, answers, strict=True):
                    r = {"id": s["id"], "key": key_of(s), "stage": stage, "raw": text,
                         **decide(s, text)}
                    if stage == "check":
                        r["for"] = turns_key(done[s["id"], "rewrite"]["turns"])
                    done[s["id"], stage] = r
                    log.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"{stage} {start + len(part)}/{len(todo)}", flush=True)
    return done


def results(sources: list[dict], done: dict) -> tuple[list[dict], Counter, Counter]:
    """(남은 학습 대화들, 데이터마다 남은 수, 뺀 까닭마다 수)."""
    kept, by_source, why = [], Counter(), Counter()
    for s in sources:
        r = done.get((s["id"], "rewrite"))
        c = done.get((s["id"], "check"))
        if r is not None and r["why"]:
            reason = r["why"]
        elif r is None or c is None:
            reason = "아직 안 함"
        else:
            reason = c["why"]
        if reason:
            why[reason.split(":")[0]] += 1
            continue
        kept.append({"kind": "chat", "source": s["source"], "topic": s.get("topic"), "id": s["id"],
                     "messages": to_messages(r["turns"]), "marks": c["marks"]})
        by_source[s["source"]] += 1
    return kept, by_source, why


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="원본 대화 (PC에만 있음)")
    p.add_argument("--out", required=True)
    p.add_argument("--log", default="data/rewrite-log.jsonl")
    p.add_argument("--limit", type=int, help="앞에서 이만큼만 (먼저 해 보기)")
    p.add_argument("--teacher", help="정해서 쓸 선생 모델 (없으면 목록에서 뜨는 것)")
    args = p.parse_args()

    if not Path(args.data).exists():
        raise SystemExit(f"원본 대화 {args.data}가 없어요: PC에서 먼저 만들어 두세요")
    sources = [json.loads(x) for x in Path(args.data).read_text().splitlines() if x.strip()]
    whole = len(sources)
    if args.limit:
        sources = sources[:args.limit]
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    llm, load_s = None, 0.0

    def llm_ask(prompts, tokens):
        nonlocal llm, load_s
        from teacher import ask, load_teacher

        if llm is None:
            t = time.time()
            llm = load_teacher(args.teacher)
            load_s = time.time() - t
        return ask(llm, prompts, tokens)

    timing: dict = {}
    done = run(llm_ask, sources, log_path, timing)
    if timing:                                # 걸린 시간과, 원본 전체를 하면 걸릴 시간
        first = next(iter(timing))
        timing[first][1] -= load_s            # 선생 모델 띄우는 시간은 빼고
        hours = 0.0
        for stage, (n, sec) in timing.items():
            per = max(sec, 0.0) / n
            share = n / timing[first][0]      # 검사는 바꿔 쓰기를 통과한 것만
            hours += per * share * whole / 3600
            print(f"시간: {stage} {n}개 {sec:.0f}초 (개당 {per:.2f}초)", flush=True)
        print(f"시간: 선생 모델 띄우기 {load_s:.0f}초. 원본 전체 {whole}개를 하면 약 {hours:.1f}시간 "
              f"(이번 속도 그대로라면)", flush=True)
    kept, by_source, why = results(sources, done)
    total = Counter(s["source"] for s in sources)
    print("남은 것: " + ", ".join(f"{k} {by_source[k]}/{n}" for k, n in sorted(total.items())),
          flush=True)
    print("뺀 까닭: " + (", ".join(f"{k} {v}" for k, v in why.most_common(12)) or "없음"), flush=True)
    Path(args.out).write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in kept))
    print(f"{args.out}: {len(kept)}개", flush=True)


if __name__ == "__main__":
    main()
