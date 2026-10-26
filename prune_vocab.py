"""어휘 줄이기: 파이보가 쓰지 않는 토큰을 빼서 임베딩(=출력층)을 줄입니다.

    python prune_vocab.py --model runs/qwen3-0.6b-school/final \\
        --out runs/qwen3-0.6b-school/small --data data/train-school.jsonl --check data/heldout-school.jsonl

Qwen3-0.6B는 어휘가 15만, Qwen3.5-0.8B는 25만이라 입력·출력이 함께 쓰는 임베딩이
Q4_K_M 파일의 1/3~2/5입니다 (122MiB, 199MiB). 출력층은 토큰마다 전부 읽으므로 속도에도
그만큼 듭니다. 남기는 토큰:

- 한글이 든 토큰과, 한글 글자의 바이트 조각으로만 된 토큰 (앞에 공백·기호 한 바이트까지):
  한국어 글은 무엇이든 원래와 똑같이 토큰으로 나뉩니다.
- 학습 데이터를 대화 템플릿과 도구 목록까지 펼쳐 실제로 나온 토큰 (영어, 숫자, 기호, JSON)
- 영어·숫자·기호만으로 된 토큰 중 BPE 병합 순서가 앞쪽(--ascii-merges)인 것: 병합이
  앞일수록 흔한 조각이라 학생이 섞어 말하는 흔한 영어 낱말("BTS", "AI")이 그대로 남습니다.
- 바이트 256개, 특수 토큰 전부, 그리고 위 토큰을 만드는 BPE 병합의 재료 토큰 (재귀로)

남긴 토큰끼리의 병합만 남기므로, 원래 토큰이 모두 남는 글은 원래와 똑같이 나뉘고
(재료가 다 남아 있어 BPE가 같은 길을 갑니다) 그렇지 않은 글(낯선 영어 낱말 등)은
더 잘게 나뉠 뿐 여전히 읽힙니다. 남긴 토큰의 로짓은 원래와 같습니다.

--check 파일(학습에 안 쓴 글)로 토큰이 하나라도 달라지는 말의 비율을 알려 줍니다.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

HANGUL = re.compile(r"[가-힣ㄱ-ㆎ]")
#: 한글 음절(U+AC00–D7A3: EA–ED로 시작)과 호환 자모(ㅋ, ㅎ: E3으로 시작)의 UTF-8 바이트
HANGUL_BYTES = set(range(0x80, 0xC0)) | {0xE3, 0xEA, 0xEB, 0xEC, 0xED}
#: llama.cpp 변환기가 토크나이저 해시로 정하는 이름: 어휘를 바꾸면 해시가 달라지므로 적어 둡니다
PRE = {"qwen3": "qwen2", "qwen3_5": "qwen35", "qwen3_5_text": "qwen35"}


def byte_decoder() -> dict[str, int]:
    """GPT-2 바이트 수준 BPE의 글자 → 바이트."""
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs, strict=True)}


def korean(token: str, bd: dict[str, int]) -> bool:
    """한글이 든 토큰이거나, (공백·기호 한 바이트 +) 한글 바이트 조각으로만 된 토큰."""
    raw = bytes(bd[c] for c in token)
    if HANGUL.search(raw.decode("utf-8", errors="ignore")):
        return True
    if raw[:1] and raw[0] < 0x80 and not chr(raw[0]).isalnum():
        raw = raw[1:]
    return bool(raw) and all(b in HANGUL_BYTES for b in raw)


def split_merge(m) -> tuple[str, str]:
    return tuple(m.split(" ", 1)) if isinstance(m, str) else tuple(m)


def keep_set(tok_json: dict, used: set[str], ascii_merges: int = 20000) -> set[str]:
    """남길 보통 토큰 (특수 토큰 빼고)."""
    vocab, bd = tok_json["model"]["vocab"], byte_decoder()
    parents, common = {}, set()
    for rank, m in enumerate(tok_json["model"]["merges"]):
        a, b = split_merge(m)
        parents.setdefault(a + b, (a, b))
        if rank < ascii_merges and all(bd[c] < 0x80 for c in a + b):
            common.add(a + b)
    keep = {t for t in vocab if korean(t, bd)} | ((used | common) & vocab.keys())
    keep |= {t for t in vocab if len(t) == 1}                        # 바이트 256개
    todo = list(keep)
    while todo:                                                      # 병합의 재료까지
        t = todo.pop()
        for p in parents.get(t, ()):
            if p not in keep:
                keep.add(p)
                todo.append(p)
    return keep


def corpus(paths: list[str]):
    """학습 예시들 (jsonl)."""
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                yield json.loads(line)


def used_tokens(tok, examples, render) -> set[str]:
    used = set()
    for ex in examples:
        used.update(tok.convert_ids_to_tokens(tok(render(ex), add_special_tokens=False)["input_ids"]))
    return used


def prune(model: Path, out: Path, data: list[str], ascii_merges: int = 20000) -> dict:
    import torch
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer

    from prompts import tools_for

    tok = AutoTokenizer.from_pretrained(model)
    tj = json.loads((model / "tokenizer.json").read_text())

    def render(ex):
        return tok.apply_chat_template(ex["messages"], tools=tools_for(ex), tokenize=False,
                                       enable_thinking=False)

    keep = keep_set(tj, used_tokens(tok, corpus(data), render), ascii_merges)
    vocab = tj["model"]["vocab"]
    normal = sorted(keep, key=vocab.get)
    added = sorted(tj["added_tokens"], key=lambda a: a["id"])
    old_ids = [vocab[t] for t in normal] + [a["id"] for a in added]
    new_id = {old: new for new, old in enumerate(old_ids)}

    tj["model"]["vocab"] = {t: i for i, t in enumerate(normal)}
    merges = [m for m in tj["model"]["merges"]
              if all(x in keep for x in split_merge(m)) and "".join(split_merge(m)) in keep]
    tj["model"]["merges"] = merges
    for a in added:
        a["id"] = new_id[a["id"]]
    out.mkdir(parents=True, exist_ok=True)
    for f in model.iterdir():                    # 템플릿, 설정 등은 그대로 옮기고 아래에서 고침
        if f.is_file() and f.suffix != ".safetensors" and f.name not in ("vocab.json", "merges.txt"):
            shutil.copy(f, out / f.name)
    (out / "tokenizer.json").write_text(json.dumps(tj, ensure_ascii=False))

    tc_path = out / "tokenizer_config.json"
    tc = json.loads(tc_path.read_text())
    if "added_tokens_decoder" in tc:
        tc["added_tokens_decoder"] = {str(new_id[int(k)]): v
                                      for k, v in tc["added_tokens_decoder"].items()}
    tc.pop("merges_file", None)
    tc.pop("vocab_file", None)
    tc_path.write_text(json.dumps(tc, ensure_ascii=False, indent=2))

    def remap(cfg: dict) -> None:
        for k in ("bos_token_id", "eos_token_id", "pad_token_id"):
            v = cfg.get(k)
            if isinstance(v, int):
                cfg[k] = new_id[v]
            elif isinstance(v, list):
                cfg[k] = [new_id[x] for x in v]

    cfg = json.loads((out / "config.json").read_text())
    old_size = cfg.get("vocab_size") or cfg["text_config"]["vocab_size"]
    for c in (cfg, cfg.get("text_config") or {}):
        if "vocab_size" in c:
            c["vocab_size"] = len(old_ids)
        remap(c)
    model_type = cfg.get("model_type")
    (out / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
    gen = out / "generation_config.json"
    if gen.exists():
        g = json.loads(gen.read_text())
        remap(g)
        gen.write_text(json.dumps(g, ensure_ascii=False, indent=2))

    index = torch.tensor(old_ids)
    for f in sorted(model.glob("*.safetensors")):
        tensors = load_file(str(f))
        for k, v in tensors.items():
            if v.dim() == 2 and v.shape[0] == old_size and ("embed_tokens" in k or "lm_head" in k):
                tensors[k] = v.index_select(0, index).contiguous()
        save_file(tensors, str(out / f.name), metadata={"format": "pt"})
    info = {"pre": PRE.get(model_type), "model_type": model_type, "old_vocab": old_size,
            "new_vocab": len(old_ids), "merges": len(merges), "old_ids": old_ids}
    (out / "pruned.json").write_text(json.dumps(info))
    return info


def check(model: Path, small: Path, path: str) -> tuple[int, int]:
    """학습에 안 쓴 글에서 (토큰이 달라진 말 수, 전체 말 수)."""
    from transformers import AutoTokenizer

    a, b = AutoTokenizer.from_pretrained(model), AutoTokenizer.from_pretrained(small)
    old_ids = json.loads((small / "pruned.json").read_text())["old_ids"]
    texts = [m["content"] for ex in corpus([path]) for m in ex["messages"]
             if m["role"] in ("user", "assistant") and m.get("content")]
    changed = sum([old_ids[i] for i in b(t, add_special_tokens=False)["input_ids"]]
                  != a(t, add_special_tokens=False)["input_ids"] for t in texts)
    return changed, len(texts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--data", nargs="+", required=True, help="펼쳐서 나온 토큰을 남길 학습 데이터")
    p.add_argument("--check", nargs="+", default=[],
                   help="학습에 안 쓴 글: 토큰이 달라지는 말의 비율을 셉니다 (파일마다)")
    p.add_argument("--ascii-merges", type=int, default=20000,
                   help="영어·숫자·기호 토큰은 BPE 병합 앞에서 이만큼 안에 든 것을 남김")
    args = p.parse_args()
    model, out = Path(args.model), Path(args.out)
    info = prune(model, out, args.data, args.ascii_merges)
    print(f"어휘 {info['old_vocab']} → {info['new_vocab']} (병합 {info['merges']}개) → {out}",
          flush=True)
    for path in args.check:
        changed, total = check(model, out, path)
        print(f"{path}: 학습에 안 쓴 말 {total}개 중 토큰이 달라진 것 {changed}개 "
              f"({changed / max(total, 1):.2%})", flush=True)


if __name__ == "__main__":
    main()
